# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

# pylint: disable=protected-access
import importlib
import importlib.util
import os

import numpy
import pytest

# Load _ufunc module directly to avoid triggering asnumpy.__init__
# which requires the CANN C++ extension (_core) not available in this env.
_UFUNC_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "src", "asnumpy", "_ufunc.py")
_spec = importlib.util.spec_from_file_location("asnumpy._ufunc", _UFUNC_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

Op = _mod.Op
_parse_loop_table = _mod._parse_loop_table
Ops = _mod.Ops
ufunc = _mod.ufunc
create_ufunc = _mod.create_ufunc
_dtype_kind = _mod._dtype_kind
_is_weak_scalar = _mod._is_weak_scalar
_get_max_array_dtype = _mod._get_max_array_dtype


class TestOp:
    @staticmethod
    def test_op_creation_with_char_codes():
        """Op correctly creates from dtype char codes via _parse_loop_table."""
        ops = _parse_loop_table(("f->f", lambda x: x), nin=1, nout=1)
        op = ops[0]
        assert op.in_types == (numpy.dtype("float32"),)
        assert op.out_types == (numpy.dtype("float32"),)

    @staticmethod
    def test_op_creation_with_bool_dtype():
        """Op correctly creates with bool output dtype via _parse_loop_table."""
        ops = _parse_loop_table(("f->?", lambda x: x), nin=1, nout=1)
        op = ops[0]
        assert op.in_types == (numpy.dtype("float32"),)
        assert op.out_types == (numpy.dtype("bool"),)

    @staticmethod
    def test_parse_loop_table_simple():
        """_parse_loop_table parses ('ff->f', routine) format."""

        def routine(x, y):
            return x

        ops = _parse_loop_table(("ff->f", routine), nin=2, nout=1)
        assert len(ops) == 1
        assert ops[0].in_types == (numpy.dtype("float32"), numpy.dtype("float32"))
        assert ops[0].out_types == (numpy.dtype("float32"),)

    @staticmethod
    def test_parse_loop_table_tuple():
        """_parse_loop_table parses (('ff->f', r1), ('dd->d', r2)) format."""

        def r1(x, y):
            return x

        def r2(x, y):
            return x

        ops = _parse_loop_table((("ff->f", r1), ("dd->d", r2)), nin=2, nout=1)
        assert len(ops) == 2
        assert ops[0].in_types == (numpy.dtype("float32"), numpy.dtype("float32"))
        assert ops[1].in_types == (numpy.dtype("float64"), numpy.dtype("float64"))

    @staticmethod
    def test_parse_loop_table_multiple_entries():
        """_parse_loop_table handles 3+ entries."""

        def r(x, y):
            return x

        ops = _parse_loop_table(
            (
                ("ee->e", r),
                ("ff->f", r),
                ("dd->d", r),
            ),
            nin=2,
            nout=1,
        )
        assert len(ops) == 3
        assert ops[0].in_types == (numpy.dtype("float16"), numpy.dtype("float16"))
        assert ops[1].in_types == (numpy.dtype("float32"), numpy.dtype("float32"))
        assert ops[2].in_types == (numpy.dtype("float64"), numpy.dtype("float64"))

    @staticmethod
    def test_parse_loop_table_input_mismatch_raises():
        """_parse_loop_table raises ValueError when input count mismatches nin."""
        with pytest.raises(ValueError, match="has 2 inputs, expected 1"):
            _parse_loop_table(("ff->f", lambda x, y: x), nin=1, nout=1)

    @staticmethod
    def test_parse_loop_table_output_mismatch_raises():
        """_parse_loop_table raises ValueError when output count mismatches nout."""
        with pytest.raises(ValueError, match="has 2 outputs, expected 1"):
            _parse_loop_table(("ff->ff", lambda x, y: (x, y)), nin=2, nout=1)

    @staticmethod
    def test_parse_loop_table_int_dtype():
        """_parse_loop_table handles integer dtype codes."""

        def r(x, y):
            return x

        ops = _parse_loop_table(("ii->i", r), nin=2, nout=1)
        assert ops[0].in_types == (numpy.dtype("int32"), numpy.dtype("int32"))
        assert ops[0].out_types == (numpy.dtype("int32"),)


class TestOps:
    @staticmethod
    def test_exact_match():
        """Exact match float32 input returns corresponding Op."""

        def r(x):
            return x

        ops = Ops([Op(["f"], ["f"], r)], nin=1, nout=1)
        result = ops.guess_routine([numpy.dtype("float32")])
        assert result is not None
        assert result.in_types == (numpy.dtype("float32"),)

    @staticmethod
    def test_no_match_returns_none():
        """No matching loop returns None."""

        def r(x):
            return x

        ops = Ops([Op(["f"], ["f"], r)], nin=1, nout=1)
        result = ops.guess_routine([numpy.dtype("int32")])
        assert result is None

    @staticmethod
    def test_first_match_wins():
        """First matching loop wins."""

        def r1(x):
            return "first"

        def r2(x):
            return "second"

        ops = Ops(
            [Op(["f"], ["f"], r1), Op(["d"], ["d"], r2)],
            nin=1,
            nout=1,
        )
        result = ops.guess_routine([numpy.dtype("float32")])
        assert result.routine is r1

    @staticmethod
    def test_caching():
        """Same input uses cache."""

        def r(x):
            return x

        ops = Ops([Op(["f"], ["f"], r)], nin=1, nout=1)
        result1 = ops.guess_routine([numpy.dtype("float32")])
        result2 = ops.guess_routine([numpy.dtype("float32")])
        assert result1 is result2

    @staticmethod
    def test_binary_op_match():
        """Binary op matches dual input dtypes."""

        def r(x, y):
            return x

        ops = Ops([Op(["f", "f"], ["f"], r)], nin=2, nout=1)
        result = ops.guess_routine([numpy.dtype("float32"), numpy.dtype("float32")])
        assert result is not None

    @staticmethod
    def test_out_dtype_filter():
        """When out_dtype specified, only loops matching output dtype return."""

        def r(x):
            return x

        ops = Ops(
            [Op(["f"], ["f"], r), Op(["d"], ["d"], r)],
            nin=1,
            nout=1,
        )
        result = ops.guess_routine([numpy.dtype("float32")], out_dtype=numpy.dtype("float64"))
        assert result is None

        result = ops.guess_routine([numpy.dtype("float32")], out_dtype=numpy.dtype("float32"))
        assert result is not None


class TestCreateUfunc:
    @staticmethod
    def test_create_unary_ufunc():
        """create_ufunc creates a unary ufunc."""
        u = create_ufunc(
            "test_neg",
            (("f->f", lambda x: x),),
        )
        assert isinstance(u, ufunc)
        assert u.name == "test_neg"
        assert u.__name__ == "test_neg"
        assert u.nin == 1
        assert u.nout == 1

    @staticmethod
    def test_create_binary_ufunc():
        """create_ufunc creates a binary ufunc."""
        u = create_ufunc(
            "test_add",
            (("ff->f", lambda x, y: x),),
        )
        assert u.nin == 2
        assert u.nout == 1

    @staticmethod
    def test_ufunc_has_ops_attribute():
        """ufunc exposes _ops for __array_ufunc__ check."""
        u = create_ufunc("test", (("ff->f", lambda x: x),))
        assert hasattr(u, "_ops")

    @staticmethod
    def test_ufunc_repr():
        """ufunc has a readable repr."""
        u = create_ufunc("my_ufunc", (("ff->f", lambda x: x),))
        assert "my_ufunc" in repr(u)

    @staticmethod
    def test_create_ufunc_with_fallback():
        """create_ufunc stores fallback on the ufunc."""

        def fallback(x):
            return x

        u = create_ufunc("test_fb", (("ff->f", lambda x: x),), fallback=fallback)
        assert hasattr(u, "_fallback")
        assert u._fallback is fallback


# ========== Integration tests (require CANN NPU) ==========
class TestUfuncAttributes:
    """Test ufunc object attributes on registered functions."""

    @staticmethod
    def test_sin_ufunc_attributes():
        """anp.sin should have ufunc attributes."""
        import asnumpy as anp

        assert hasattr(anp.sin, "nin")
        assert anp.sin.nin == 1
        assert hasattr(anp.sin, "nout")
        assert anp.sin.nout == 1
        assert anp.sin.name == "sin"

    @staticmethod
    def test_add_ufunc_attributes():
        """anp.add should have ufunc attributes."""
        import asnumpy as anp

        assert anp.add.nin == 2
        assert anp.add.nout == 1

    @staticmethod
    def test_equal_ufunc_attributes():
        """anp.equal should have ufunc attributes."""
        import asnumpy as anp

        assert anp.equal.nin == 2
        assert anp.equal.nout == 1


class TestUfuncCorrectness:
    """Test ufunc-registered functions produce correct results on NPU."""

    @staticmethod
    def _make_array(data, dtype=numpy.float32):
        """Create asnumpy array from plain data."""
        import asnumpy as anp

        return anp.ndarray.from_numpy(numpy.array(data, dtype=dtype))

    @staticmethod
    def test_sin_result():
        """ufunc-registered sin produces correct result."""
        import asnumpy as anp

        data = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float32)
        arr = TestUfuncCorrectness._make_array(data)
        result = anp.sin(arr)
        expected = numpy.sin(data)
        numpy.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)

    @staticmethod
    def test_add_result():
        """ufunc-registered add produces correct result."""
        import asnumpy as anp

        a = TestUfuncCorrectness._make_array([1.0, 2.0, 3.0])
        b = TestUfuncCorrectness._make_array([4.0, 5.0, 6.0])
        result = anp.add(a, b)
        numpy.testing.assert_allclose(result.to_numpy(), [5.0, 7.0, 9.0], atol=1e-5)

    @staticmethod
    def test_negative_result():
        """ufunc-registered negative produces correct result."""
        import asnumpy as anp

        arr = TestUfuncCorrectness._make_array([1.0, -2.0, 3.0])
        result = anp.negative(arr)
        numpy.testing.assert_allclose(result.to_numpy(), [-1.0, 2.0, -3.0], atol=1e-5)

    @staticmethod
    def test_equal_result():
        """ufunc-registered equal produces correct bool result."""
        import asnumpy as anp

        a = TestUfuncCorrectness._make_array([1.0, 2.0, 3.0])
        b = TestUfuncCorrectness._make_array([1.0, 0.0, 3.0])
        result = anp.equal(a, b)
        expected = numpy.array([True, False, True])
        numpy.testing.assert_array_equal(result.to_numpy(), expected)


class TestArrayUfuncProtocol:
    """Test __array_ufunc__ dispatch from NumPy to AsNumpy."""

    @staticmethod
    def _make_array(data, dtype=numpy.float32):
        import asnumpy as anp

        return anp.ndarray.from_numpy(numpy.array(data, dtype=dtype))

    @staticmethod
    def test_numpy_sin_dispatches():
        """np.sin(asnumpy_array) dispatches to anp.sin."""
        arr = TestArrayUfuncProtocol._make_array([0.0, 1.0, 2.0])
        result = numpy.sin(arr)
        assert isinstance(result, type(arr))

    @staticmethod
    def test_numpy_add_dispatches():
        """np.add(asnumpy_a, asnumpy_b) dispatches to anp.add."""
        a = TestArrayUfuncProtocol._make_array([1.0, 2.0])
        b = TestArrayUfuncProtocol._make_array([3.0, 4.0])
        result = numpy.add(a, b)
        assert isinstance(result, type(a))

    @staticmethod
    def test_numpy_equal_dispatches():
        """np.equal(asnumpy_a, asnumpy_b) dispatches to anp.equal."""
        a = TestArrayUfuncProtocol._make_array([1.0, 2.0])
        b = TestArrayUfuncProtocol._make_array([1.0, 0.0])
        result = numpy.equal(a, b)
        assert isinstance(result, type(a))

    @staticmethod
    def test_numpy_negative_dispatches():
        """np.negative(asnumpy_arr) dispatches to anp.negative."""
        arr = TestArrayUfuncProtocol._make_array([1.0, 2.0])
        result = numpy.negative(arr)
        assert isinstance(result, type(arr))

    @staticmethod
    def test_result_consistency():
        """np.sin(asnumpy_arr) should match anp.sin(asnumpy_arr)."""
        import asnumpy as anp

        arr = TestArrayUfuncProtocol._make_array([0.0, 1.0, 2.0])
        result_np = numpy.sin(arr).to_numpy()
        result_anp = anp.sin(arr).to_numpy()
        numpy.testing.assert_allclose(result_np, result_anp, atol=1e-6)


# ========== Weak scalar tests (pure Python, no CANN needed) ==========
class TestWeakScalar:
    """Test weak-scalar coercion helpers and ufunc dispatch logic."""

    # ---------- helper function tests ----------

    @staticmethod
    def test_numpy_scalar_not_weak():
        """NumPy scalar (np.float32(1)) is NOT weak."""
        assert _is_weak_scalar(numpy.float32(1.0)) is False
        assert _is_weak_scalar(numpy.int32(1)) is False

    @staticmethod
    def test_bool_not_weak():
        """Python bool is NOT weak."""
        assert _is_weak_scalar(True) is False
        assert _is_weak_scalar(False) is False

    @staticmethod
    def test_python_float_is_weak():
        """Python float IS weak."""
        assert _is_weak_scalar(1.5) is True
        assert _is_weak_scalar(0.0) is True

    @staticmethod
    def test_python_int_is_weak():
        """Python int IS weak."""
        assert _is_weak_scalar(42) is True
        assert _is_weak_scalar(0) is True

    @staticmethod
    def test_python_complex_is_weak():
        """Python complex IS weak."""
        assert _is_weak_scalar(1.0 + 2.0j) is True

    @staticmethod
    def test_dtype_kind_values():
        """_dtype_kind returns expected scores."""
        assert _dtype_kind(numpy.dtype("bool")) == 0
        assert _dtype_kind(numpy.dtype("int32")) == 1
        assert _dtype_kind(numpy.dtype("float32")) == 2
        assert _dtype_kind(numpy.dtype("float64")) == 2
        assert _dtype_kind(numpy.dtype("complex128")) == 2

    @staticmethod
    def test_get_max_array_dtype_none():
        """_get_max_array_dtype returns None when no array args."""
        assert _get_max_array_dtype([1, 2.0, 3 + 4j]) is None

    @staticmethod
    def test_get_max_array_dtype_mixed():
        """_get_max_array_dtype picks the highest-kind dtype."""
        a32 = numpy.array([1], dtype=numpy.float32)
        i64 = numpy.array([1], dtype=numpy.int64)
        assert _get_max_array_dtype([a32, i64]) == numpy.dtype("float32")

    @staticmethod
    def test_get_max_array_dtype_same_kind_float():
        """Among float dtypes, the highest precision (largest itemsize) wins."""
        a16 = numpy.array([1], dtype=numpy.float16)
        a32 = numpy.array([1], dtype=numpy.float32)
        a64 = numpy.array([1], dtype=numpy.float64)
        assert _get_max_array_dtype([a16, a32]) == numpy.dtype("float32")
        assert _get_max_array_dtype([a32, a16]) == numpy.dtype("float32")
        assert _get_max_array_dtype([a16, a64]) == numpy.dtype("float64")

    @staticmethod
    def test_get_max_array_dtype_same_kind_int():
        """Among int dtypes, the highest precision (largest itemsize) wins."""
        i8 = numpy.array([1], dtype=numpy.int8)
        i32 = numpy.array([1], dtype=numpy.int32)
        assert _get_max_array_dtype([i8, i32]) == numpy.dtype("int32")
        assert _get_max_array_dtype([i32, i8]) == numpy.dtype("int32")

    # ---------- coercion integration tests ----------

    @staticmethod
    def test_coerce_float_to_float32():
        """_coerce_weak_scalar converts Python float to float32."""
        result = _mod._coerce_weak_scalar(1.5, numpy.dtype("float32"))
        assert isinstance(result, numpy.ndarray)
        assert result.dtype == numpy.dtype("float32")
        assert result == 1.5

    @staticmethod
    def test_coerce_int_to_float32():
        """_coerce_weak_scalar converts Python int to float32."""
        result = _mod._coerce_weak_scalar(42, numpy.dtype("float32"))
        assert result.dtype == numpy.dtype("float32")

    @staticmethod
    def test_coerce_complex_to_float_promotes_to_complex128():
        """Complex scalar coerced to real dtype promotes to complex128."""
        result = _mod._coerce_weak_scalar(1.0 + 0j, numpy.dtype("float32"))
        assert numpy.issubdtype(result.dtype, numpy.complexfloating)

    # ---------- ufunc dispatch with weak scalars (needs _ndarray mock) ----------

    @staticmethod
    def test_ufunc_coerces_scalar_dtype():
        """Verify that weak scalar changes in_dtypes before dispatch."""
        create_ufunc(
            "test_ws",
            (("f->f", lambda x: x), ("d->d", lambda x: x)),
        )
        # Manually test the coercion logic from ufunc.__call__
        arr = numpy.array([1.0, 2.0], dtype=numpy.float32)
        max_dtype = _get_max_array_dtype([arr, 0.5])
        assert max_dtype == numpy.dtype("float32")
        # After coercion, the scalar should have float32 dtype
        coerced = _mod._coerce_weak_scalar(0.5, max_dtype)
        assert coerced.dtype == numpy.dtype("float32")

    @staticmethod
    def test_ufunc_int_scalar_kind_le_float():
        """int kind (1) <= float32 kind (2), so coercion happens."""
        arr = numpy.array([1.0], dtype=numpy.float32)
        scalar_dtype = numpy.dtype(int)  # int -> int64
        assert _dtype_kind(scalar_dtype) <= _dtype_kind(arr.dtype)

    @staticmethod
    def test_ufunc_float_scalar_kind_eq_float():
        """float kind (2) == float32 kind (2), so coercion happens."""
        arr = numpy.array([1.0], dtype=numpy.float32)
        scalar_dtype = numpy.dtype(float)  # float -> float64
        assert _dtype_kind(scalar_dtype) == _dtype_kind(arr.dtype)

    @staticmethod
    def test_no_arrays_no_coercion():
        """Without arrays, _get_max_array_dtype returns None."""
        assert _get_max_array_dtype([1.0, 2.0]) is None
