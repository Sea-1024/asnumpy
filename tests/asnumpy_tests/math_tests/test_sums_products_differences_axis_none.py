"""全局归约（axis=None）的 NumPy 契约测试（dtype 提升、标量 dtype）。"""

import numpy

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ---------------------------------------------------------------------------
# sum / nansum（全局标量）
# ---------------------------------------------------------------------------


@testing.for_dtypes([numpy.bool_, numpy.int8, numpy.int16, numpy.int32, numpy.int64])
@testing.numpy_asnumpy_allclose()
def test_sum_axis_none_signed_contract(xp, dtype):
    a = _create_array(xp, [[100, 100], [50, 25]], dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64])
@testing.numpy_asnumpy_allclose()
def test_sum_axis_none_unsigned_contract(xp, dtype):
    a = _create_array(xp, [[100, 100]], dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_sum_axis_none_float_contract(xp, dtype):
    a = _create_array(xp, [[1.5, 2.5], [0.5, 0.5]], dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_sum_axis_none_float16_contract(xp, dtype):
    a = _create_array(xp, [[1.0, 2.0], [3.0, 4.0]], dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.complex64])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_sum_axis_none_complex_contract(xp, dtype):
    a = _create_array(xp, [[1 + 2j, 3 - 1j], [0.5j, -2]], dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.float32, numpy.int64])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_sum_axis_none_explicit_dtype(xp, dtype):
    a = _create_array(xp, [[100, 100]], numpy.int8)
    return xp.sum(a, dtype=dtype)


@testing.for_dtypes([numpy.uint64])
@testing.numpy_asnumpy_allclose()
def test_sum_axis_none_explicit_uint64_dtype(xp, dtype):
    a = _create_array(xp, [[100, 100]], numpy.uint8)
    return xp.sum(a, dtype=dtype)


@testing.for_dtypes([numpy.int8])
@testing.numpy_asnumpy_allclose()
def test_sum_axis_none_empty(xp, dtype):
    a = _create_array(xp, numpy.zeros((0,), dtype=dtype), dtype)
    return xp.sum(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_nansum_axis_none_float_contract(xp, dtype):
    a = _create_array(xp, [[1.0, numpy.nan], [2.0, 3.0]], dtype)
    return xp.nansum(a)


@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_nansum_axis_none_float16_contract(xp, dtype):
    a = _create_array(xp, [[1.0, numpy.nan]], dtype)
    return xp.nansum(a)


@testing.for_dtypes([numpy.int8])
@testing.numpy_asnumpy_allclose()
def test_nansum_axis_none_integer_contract(xp, dtype):
    a = _create_array(xp, [[100, 100]], dtype)
    return xp.nansum(a)


# ---------------------------------------------------------------------------
# prod / nanprod（全局标量）
# ---------------------------------------------------------------------------


@testing.for_dtypes([numpy.bool_, numpy.int8, numpy.int64])
@testing.numpy_asnumpy_allclose()
def test_prod_axis_none_signed_contract(xp, dtype):
    a = _create_array(xp, [[2, 3], 1], dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.float16])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_prod_axis_none_float16_contract(xp, dtype):
    a = _create_array(xp, [[2.0, 3.0]], dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_prod_axis_none_float64_contract(xp, dtype):
    a = _create_array(xp, [[2.0, 3.0], [1.0, 1.0]], dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.complex64])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_prod_axis_none_complex_contract(xp, dtype):
    a = _create_array(xp, [[2 + 1j, 3]], dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.uint8, numpy.uint64])
@testing.numpy_asnumpy_allclose()
def test_prod_axis_none_unsigned_contract(xp, dtype):
    a = _create_array(xp, [[3, 4]], dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_prod_axis_none_empty(xp, dtype):
    a = _create_array(xp, numpy.zeros((0,), dtype=dtype), dtype)
    return xp.prod(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-6, atol=1e-6)
def test_nanprod_axis_none_float_contract(xp, dtype):
    a = _create_array(xp, [[2.0, numpy.nan], [3.0, 2.0]], dtype)
    return xp.nanprod(a)


@testing.for_dtypes([numpy.int8])
@testing.numpy_asnumpy_allclose()
def test_nanprod_axis_none_integer_contract(xp, dtype):
    a = _create_array(xp, [[3, 4]], dtype)
    return xp.nanprod(a)
