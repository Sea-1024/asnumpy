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

"""求和、乘积与累积算子测试

针对已记录的 CANN 算子限制进行精准标注。
"""

import numpy
import pytest

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 求和 (Sum) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sum_basic(xp, dtype):
    """记录：必须指定 axis 和 keepdims"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.sum(a, axis=0, keepdims=False)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sum_axis_none_basic(xp, dtype):
    """全局归约（axis=None）返回带 dtype 的 NumPy 标量（原 xfail：aclnnFlatten 不支持 float64）"""
    a = _create_array(xp, [1.0, 2.0], dtype)
    return xp.sum(a)


# ========== 2. 乘积 (Prod) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_prod_basic(xp, dtype):
    """记录：必须指定 axis 和 keepdims"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.prod(a, axis=0, keepdims=False)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_prod_axis_none_xfail(xp, dtype):
    a = _create_array(xp, [1.0, 2.0], dtype)
    return xp.prod(a)


# ========== 3. 归约 Dtype 限制 (XFAIL) ==========


@pytest.mark.xfail(
    reason="[FIXABLE] dtype promotion: sum/prod keeps original dtype, NumPy promotes", strict=True
)
@testing.for_dtypes([numpy.int8, numpy.int16, numpy.int32, numpy.uint8])
def test_sum_int_mismatch_xfail(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    return xp.sum(a, axis=0, keepdims=False)


@pytest.mark.xfail(reason="[FIXABLE] aclnnSum/Prod unsupported dtypes (float16/uints)", strict=True)
@testing.for_dtypes([numpy.float16, numpy.uint16, numpy.uint32, numpy.uint64])
def test_sum_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1, 1], dtype)
    return xp.sum(a, axis=0, keepdims=False)


# ========== 4. NaN 系列归约 (Nansum, Nanprod) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_nansum_basic(xp, dtype):
    data = [[1.0, numpy.nan], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.nansum(a, axis=0, keepdims=False)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_nanprod_basic(xp, dtype):
    data = [[1.0, numpy.nan], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.nanprod(a, axis=0, keepdims=False)


@pytest.mark.xfail(
    reason="[FIXABLE] nansum/nanprod unsupported dtypes (float16/64/uints/complex)", strict=True
)
@testing.for_dtypes([numpy.float16, numpy.float64, numpy.uint16, numpy.uint32, numpy.complex64])
def test_nan_reduction_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    return xp.nansum(a, axis=0, keepdims=False)


# ========== 5. 累积运算 (Cumsum, Cumprod) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_cumsum_basic(xp, dtype):
    """记录：必须指定 axis 参数"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.cumsum(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_cumprod_basic(xp, dtype):
    """记录：必须指定 axis 参数"""
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.cumprod(a, axis=0)


@pytest.mark.xfail(
    reason="[FIXABLE] dtype promotion: cumulative ops keep int dtype, NumPy promotes", strict=True
)
@testing.for_dtypes([numpy.int8, numpy.int16, numpy.int32, numpy.uint8])
def test_cumsum_int_mismatch_xfail(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    return xp.cumsum(a, axis=0)


@pytest.mark.xfail(
    reason="[FIXABLE] cumprod/cumsum unsupported dtypes (float16/uints/complex)", strict=True
)
@testing.for_dtypes([numpy.float16, numpy.uint16, numpy.uint32, numpy.complex64])
def test_cumulative_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    return xp.cumprod(a, axis=0)


# ========== 6. NaN 系列累积 (Nancumsum, Nancumprod) ==========


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_nancumsum_basic(xp, dtype):
    data = [1.0, numpy.nan, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.nancumsum(a, axis=0)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose()
def test_nancumprod_basic(xp, dtype):
    data = [1.0, numpy.nan, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.nancumprod(a, axis=0)


@pytest.mark.xfail(
    reason="[FIXABLE] nancumsum/nancumprod unsupported dtypes (float64/uints/complex)", strict=True
)
@testing.for_dtypes([numpy.float64, numpy.uint16, numpy.uint32, numpy.complex64])
def test_nan_cumulative_unsupported_xfail(xp, dtype):
    a = _create_array(xp, [1.0], dtype)
    return xp.nancumsum(a, axis=0)
