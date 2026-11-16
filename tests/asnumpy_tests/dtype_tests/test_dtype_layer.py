# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

"""Tests for the dtype layer: the numpy<->ACL mapping, promotion, and astype.

np.result_type is the oracle throughout, per test_numpy_baseline.py.

Note on uint16/uint32/uint64: these map and round-trip correctly, but several aclnn kernels do not
accept them on Ascend 910 (even same-dtype `add(uint16, uint16)` raises). That is a kernel gap, not
a type-layer gap, so the promotion tests below exclude them and test the mapping separately.
"""

import itertools

import numpy as np
import pytest

import asnumpy as ap

# dtypes that both map cleanly AND have working aclnn kernels for the ops exercised here
PROMOTABLE = [
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.float16,
    np.float32,
    np.float64,
]

# every dtype the type layer claims to support, including ones whose kernels are missing
ALL_SUPPORTED = PROMOTABLE + [np.uint16, np.uint32, np.uint64]


def _arr(dtype, value=1):
    return ap.ndarray.from_numpy(np.full(4, value, dtype=dtype))


# ========== 1. numpy <-> ACL mapping is a true round trip ==========


@pytest.mark.parametrize("dtype", ALL_SUPPORTED)
def test_dtype_round_trips_bit_exact(dtype):
    """from_numpy -> to_numpy preserves both dtype and the exact bits."""
    host = np.arange(8).astype(dtype)
    out = ap.ndarray.from_numpy(host).to_numpy()
    assert out.dtype == np.dtype(dtype)
    assert np.array_equal(out, host)


def test_float16_round_trips_special_values():
    """float16 previously widened to float32 and overflowed the host buffer.

    Every value here was corrupted by the old hand-rolled bit conversion: it handled normals only,
    decoding +0.0 as 3.05e-05 and inf as 65536.
    """
    host = np.array(
        [0.0, -0.0, 1.0, -1.5, np.inf, -np.inf, np.nan, 6e-8, 65504.0], dtype=np.float16
    )
    out = ap.ndarray.from_numpy(host).to_numpy()
    assert out.dtype == np.float16
    # compare bit patterns so -0.0 and nan are checked exactly rather than by value
    assert np.array_equal(out.view(np.uint16), host.view(np.uint16))


def test_equivalent_dtypes_with_different_type_numbers_are_accepted():
    """np.longlong is 8-byte signed like np.int64 but has a distinct type number.

    The old identity-based (`dtype.is(...)`) mapping rejected it.
    """
    out = ap.ndarray.from_numpy(np.ones(4, np.longlong)).to_numpy()
    assert out.dtype == np.dtype(np.longlong)


@pytest.mark.parametrize(
    "bad, reason",
    [(">f4", "big-endian"), ("<f16", "longdouble")],
)
def test_unsupported_dtypes_are_rejected_clearly(bad, reason):
    """Unsupported dtypes must raise a typed error, not fall through to an opaque ACL failure."""
    with pytest.raises((TypeError, ValueError)):
        ap.ndarray.from_numpy(np.ones(2, dtype=bad))


# ========== 2. Binary op promotion matches NumPy ==========


@pytest.mark.parametrize("d1, d2", list(itertools.product(PROMOTABLE, PROMOTABLE)))
def test_add_dtype_matches_numpy(d1, d2):
    """ap.add's output dtype is exactly np.result_type's."""
    assert ap.add(_arr(d1), _arr(d2)).dtype == np.result_type(d1, d2)


@pytest.mark.parametrize("d1, d2", list(itertools.product(PROMOTABLE, PROMOTABLE)))
def test_add_dtype_is_argument_order_independent(d1, d2):
    """The regression this layer exists to prevent.

    add(int32, float32) used to give int32 while add(float32, int32) gave float32, because each op
    took x1's dtype and never looked at x2.
    """
    assert ap.add(_arr(d1), _arr(d2)).dtype == ap.add(_arr(d2), _arr(d1)).dtype


@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply", "divide"])
def test_mixed_dtype_binary_ops_match_numpy_values_and_dtype(op_name):
    """Values, not just dtypes: promotion must not silently truncate."""
    host1 = np.array([1, 2, 3, 4], dtype=np.int32)
    host2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    got = getattr(ap, op_name)(
        ap.ndarray.from_numpy(host1), ap.ndarray.from_numpy(host2)
    ).to_numpy()
    want = getattr(np, op_name)(host1, host2)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_maximum_does_not_demote():
    """maximum used to take x2's dtype whenever x1 was int16/int32/int64, demoting the result."""
    got = ap.maximum(_arr(np.int32, 5), _arr(np.int8, 3))
    assert got.dtype == np.result_type(np.int32, np.int8) == np.int32
    assert np.array_equal(got.to_numpy(), np.full(4, 5, np.int32))


def test_explicit_dtype_argument_still_wins():
    assert ap.add(_arr(np.int32), _arr(np.int32), dtype=np.float32).dtype == np.float32


def test_ldexp_does_not_promote_against_its_exponent():
    """numpy types ldexp as 'fi->f': the int exponent must not widen the result to float64."""
    x1 = np.array([1.0, 2.0], dtype=np.float32)
    x2 = np.array([1, 2], dtype=np.int32)
    got = ap.ldexp(ap.ndarray.from_numpy(x1), ap.ndarray.from_numpy(x2)).to_numpy()
    want = np.ldexp(x1, x2)
    assert got.dtype == want.dtype == np.float32
    np.testing.assert_allclose(got, want)


def test_ldexp_with_integer_x1_widens_to_float64():
    """numpy's ldexp has no 'ii->i' loop, so an integer x1 must widen rather than truncate.

    Pinning x1's dtype naively made ldexp(1, -1) return int32 0 (or raise) instead of 0.5.
    """
    x1 = np.array([1, 2], dtype=np.int32)
    x2 = np.array([-1, 3], dtype=np.int32)
    got = ap.ldexp(ap.ndarray.from_numpy(x1), ap.ndarray.from_numpy(x2)).to_numpy()
    want = np.ldexp(x1, x2)
    assert got.dtype == want.dtype == np.float64
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_true_divide_of_integers_is_float64():
    """numpy's true_divide has no integer loop, so int/int must not truncate.

    np.divide.types is ['ee->e','ff->f','dd->d','FF->F','DD->D'] -- taking the promoted operand
    type would give int32 [0,1,1] instead of float64 [0.5,1.0,1.5].
    """
    x1 = np.array([1, 2, 3], dtype=np.int32)
    x2 = np.array([2, 2, 2], dtype=np.int32)
    got = ap.divide(ap.ndarray.from_numpy(x1), ap.ndarray.from_numpy(x2)).to_numpy()
    want = np.divide(x1, x2)
    assert got.dtype == want.dtype == np.float64
    np.testing.assert_allclose(got, want)


@pytest.mark.parametrize("op_name", ["gcd", "lcm"])
def test_rational_ops_promote_rather_than_taking_x1_dtype(op_name):
    """gcd/lcm pinned x1's dtype while their operands were promoted, returning the wrong type."""
    x1 = np.full(4, 12, dtype=np.int32)
    x2 = np.full(4, 8, dtype=np.int64)
    got = getattr(ap, op_name)(ap.ndarray.from_numpy(x1), ap.ndarray.from_numpy(x2)).to_numpy()
    want = getattr(np, op_name)(x1, x2)
    assert got.dtype == want.dtype == np.int64
    assert np.array_equal(got, want)


def test_divmod_promotes_both_outputs():
    """divmod was the fourth hand-rolled binary op and kept x1's dtype."""
    x1 = np.array([7, 8], dtype=np.int32)
    x2 = np.array([2.0, 2.0], dtype=np.float32)
    q, r = ap.divmod(ap.ndarray.from_numpy(x1), ap.ndarray.from_numpy(x2))
    wq, wr = np.divmod(x1, x2)
    assert q.dtype == wq.dtype == np.float64
    assert r.dtype == wr.dtype == np.float64
    np.testing.assert_allclose(q.to_numpy(), wq)
    np.testing.assert_allclose(r.to_numpy(), wr)


# ========== 3. Python dtype API mirrors NumPy ==========


def test_dtype_api_is_numpy():
    """asnumpy has no dtype system of its own; these must be NumPy's objects."""
    assert ap.dtype is np.dtype
    assert ap.float32 is np.float32
    assert ap.promote_types is np.promote_types


@pytest.mark.parametrize("d1, d2", list(itertools.product(PROMOTABLE, PROMOTABLE)))
def test_result_type_matches_numpy_for_dtypes_and_arrays(d1, d2):
    want = np.result_type(d1, d2)
    assert ap.result_type(d1, d2) == want
    # arrays must unwrap to their dtype rather than transferring off device
    assert ap.result_type(_arr(d1), _arr(d2)) == want


@pytest.mark.parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
def test_can_cast_matches_numpy(casting):
    for d1, d2 in itertools.product(PROMOTABLE, PROMOTABLE):
        assert ap.can_cast(d1, d2, casting=casting) == np.can_cast(d1, d2, casting=casting)


# ========== 4. astype ==========


@pytest.mark.parametrize("target", PROMOTABLE)
def test_astype_produces_target_dtype(target):
    got = ap.ndarray.from_numpy(np.ones(4, np.int32)).astype(target)
    assert got.dtype == np.dtype(target)
    assert isinstance(got, ap.ndarray)
    np.testing.assert_allclose(got.to_numpy(), np.ones(4, target))


def test_astype_rejects_disallowed_cast():
    with pytest.raises(TypeError):
        ap.ndarray.from_numpy(np.ones(4, np.int32)).astype(np.int8, casting="safe")


def test_astype_copy_false_returns_self_when_dtype_matches():
    a = ap.ndarray.from_numpy(np.ones(4, np.int32))
    assert a.astype(np.int32, copy=False) is a
    # the default is a copy, as in NumPy
    assert a.astype(np.int32) is not a


def test_astype_positional_order_matches_numpy():
    """`x.astype(dt, "K")` must bind order, not copy -- the standard NumPy/CuPy idiom."""
    a = ap.ndarray.from_numpy(np.ones(4, np.int32))
    assert a.astype(np.float32, "K").dtype == np.float32
    with pytest.raises(ValueError):
        a.astype(np.float32, "F")  # asnumpy arrays are always C-contiguous


def test_astype_accepts_scalar_type_spellings_on_the_raw_core_type():
    """The _core binding must accept np.float32, not only np.dtype("float32").

    py::dtype's caster is a strict isinstance check, so a raw _core.ndarray (what ap.load returns)
    would otherwise reject the spelling NumPy users write.
    """
    from asnumpy._core import ndarray as core_ndarray

    raw = core_ndarray.from_numpy(np.ones(4, np.int32))
    assert raw.astype(np.float32).dtype == np.float32
    assert raw.astype(np.dtype("float32")).dtype == np.float32


@pytest.mark.parametrize("name", ["longlong", "ulonglong", "intp", "single", "double"])
def test_numpy_dtype_aliases_resolve(name):
    """ap.<alias> must agree with what from_numpy accepts: both go through normalized_num().

    Exercises the type layer via from_numpy/to_numpy (a plain memcpy) rather than an op, so a
    missing kernel for a given dtype does not masquerade as a dtype-layer failure.
    """
    alias = getattr(ap, name)
    assert alias is getattr(np, name)
    host = np.ones(2, dtype=alias)
    assert ap.ndarray.from_numpy(host).to_numpy().dtype == np.dtype(alias)
