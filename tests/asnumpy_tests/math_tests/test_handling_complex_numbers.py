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

"""复数运算算子测试

当前阶段仅测试：real
记录问题：代码强制将所有复数（包括 complex128）的输出类型硬编码为 ACL_FLOAT (float32)，
导致 complex128 -> float64 的精度下降。
"""

import numpy

from asnumpy import testing

# ========== 辅助函数 ==========


def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. 实部提取 (Real) ==========


@testing.for_dtypes([numpy.complex64])
@testing.numpy_asnumpy_allclose()
def test_real_complex64_basic(xp, dtype):
    """测试 complex64 -> float32 链路"""
    data = [1.0 + 2.0j, -3.5 + 4.5j, 0.0 + 0.0j]
    a = _create_array(xp, data, dtype)
    return xp.real(a)


@testing.for_dtypes([numpy.complex128])
@testing.numpy_asnumpy_allclose()
def test_real_complex128(xp, dtype):
    """complex128 提取实部应返回 float64"""
    data = [1.23456789012345 + 0.5j]
    a = _create_array(xp, data, dtype)
    return xp.real(a)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_allclose()
def test_real_non_complex(xp, dtype):
    """NumPy 允许对实数调 real (返回自身)"""
    a = _create_array(xp, [1, 2], dtype)
    return xp.real(a)
