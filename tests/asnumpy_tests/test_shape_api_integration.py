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

"""Integration tests for the shape-manipulation API.

Each feature is covered in depth by its own test module. These tests check that
the five pieces compose, and that none of them regressed another when merged.
"""

import numpy
import pytest

import asnumpy as ap


def _roundtrip(np_arr):
    return ap.ndarray.from_numpy(np_arr)


def test_all_shape_api_present():
    """Every symbol the five PRs added is reachable from the package."""
    for name in ("concatenate", "stack", "vstack", "hstack", "squeeze", "expand_dims"):
        assert hasattr(ap, name), f"asnumpy.{name} is missing"
        assert name in ap.__all__, f"asnumpy.{name} missing from __all__"

    a = _roundtrip(numpy.zeros((2, 3), dtype=numpy.float32))
    for method in ("reshape", "flatten", "ravel", "transpose", "swapaxes", "squeeze"):
        assert hasattr(a, method), f"ndarray.{method} is missing"
    assert hasattr(a, "T")
    assert a[0] is not None


def test_reshape_then_transpose():
    np_a = numpy.arange(24, dtype=numpy.float32)
    a = _roundtrip(np_a)
    got = a.reshape((2, 3, 4)).transpose((1, 0, 2)).to_numpy()
    expected = np_a.reshape((2, 3, 4)).transpose((1, 0, 2))
    numpy.testing.assert_array_equal(got, expected)


def test_concatenate_then_index():
    np_a = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
    np_b = numpy.array([[5, 6]], dtype=numpy.float32)
    joined = ap.concatenate([_roundtrip(np_a), _roundtrip(np_b)], axis=0)
    expected = numpy.concatenate([np_a, np_b], axis=0)
    numpy.testing.assert_array_equal(joined[2].to_numpy(), expected[2])
    numpy.testing.assert_array_equal(joined[1, 1].to_numpy(), expected[1, 1])


def test_stack_squeeze_roundtrip():
    np_a = numpy.array([1, 2, 3], dtype=numpy.float32)
    stacked = ap.stack([_roundtrip(np_a)])
    assert stacked.shape == (1, 3)
    numpy.testing.assert_array_equal(ap.squeeze(stacked).to_numpy(), np_a)


def test_expand_dims_then_concatenate():
    """expand_dims + concatenate must agree with stack."""
    np_a = numpy.array([1, 2, 3], dtype=numpy.float32)
    np_b = numpy.array([4, 5, 6], dtype=numpy.float32)
    a, b = _roundtrip(np_a), _roundtrip(np_b)

    via_expand = ap.concatenate([ap.expand_dims(a, 0), ap.expand_dims(b, 0)], axis=0)
    via_stack = ap.stack([a, b], axis=0)
    numpy.testing.assert_array_equal(via_expand.to_numpy(), via_stack.to_numpy())
    numpy.testing.assert_array_equal(via_expand.to_numpy(), numpy.stack([np_a, np_b]))


def test_transpose_then_flatten_sees_permuted_order():
    """Flatten after transpose must read the permuted order, not the original.

    This is the test that would fail if transpose returned a relabelled view
    over the original buffer instead of materializing the permutation.
    """
    np_a = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    got = _roundtrip(np_a).transpose().flatten().to_numpy()
    numpy.testing.assert_array_equal(got, np_a.T.flatten())


def test_vstack_hstack_against_numpy():
    np_a = numpy.array([1, 2, 3], dtype=numpy.float32)
    np_b = numpy.array([4, 5, 6], dtype=numpy.float32)
    a, b = _roundtrip(np_a), _roundtrip(np_b)
    numpy.testing.assert_array_equal(ap.vstack([a, b]).to_numpy(), numpy.vstack([np_a, np_b]))
    numpy.testing.assert_array_equal(ap.hstack([a, b]).to_numpy(), numpy.hstack([np_a, np_b]))


def test_chained_pipeline():
    """A longer chain touching all five features at once."""
    np_a = numpy.arange(12, dtype=numpy.float32)
    a = _roundtrip(np_a)

    result = ap.squeeze(ap.expand_dims(a.reshape((3, 4)).swapaxes(0, 1), axis=0))
    expected = numpy.squeeze(numpy.expand_dims(np_a.reshape((3, 4)).swapaxes(0, 1), axis=0))

    assert result.shape == expected.shape
    numpy.testing.assert_array_equal(result.to_numpy(), expected)


def test_existing_math_still_works():
    """The merged shape work must not have disturbed the math path."""
    np_a = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=numpy.float32)
    a = _roundtrip(np_a)
    numpy.testing.assert_allclose(ap.multiply(a, a).to_numpy(), np_a * np_a, rtol=1e-6)
    numpy.testing.assert_allclose(
        ap.multiply(a.transpose(), a.transpose()).to_numpy(), np_a.T * np_a.T, rtol=1e-6
    )


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.int32, numpy.int64])
def test_shape_ops_across_dtypes(dtype):
    np_a = numpy.arange(6, dtype=dtype).reshape(2, 3)
    a = _roundtrip(np_a)
    numpy.testing.assert_array_equal(a.transpose().to_numpy(), np_a.T)
    numpy.testing.assert_array_equal(a.reshape((3, 2)).to_numpy(), np_a.reshape(3, 2))
    numpy.testing.assert_array_equal(a[1].to_numpy(), np_a[1])
