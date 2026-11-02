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

import numpy
import pytest

from asnumpy import testing


def _create_array(xp, data, dtype):
    """Helper function: create array"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_reshape_basic(xp, dtype):
    """Test basic reshape operation"""
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a.reshape((3, 2))


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_reshape_to_1d(xp, dtype):
    """Test reshape to 1D"""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return a.reshape((4,))


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_reshape_with_negative_one(xp, dtype):
    """Test reshape with -1 dimension inference"""
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a.reshape((-1, 2))


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_flatten(xp, dtype):
    """Test flatten operation"""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return a.flatten()


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_ravel(xp, dtype):
    """Test ravel operation"""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return a.ravel()


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_reshape_3d(xp, dtype):
    """Test reshape with 3D arrays"""
    a = _create_array(xp, numpy.arange(24, dtype=dtype).reshape(2, 3, 4), dtype)
    return a.reshape((4, 6))


def test_reshape_invalid_size(xp=None):
    """Test that reshape with invalid size raises error"""
    import asnumpy as ap
    a = ap.ndarray.from_numpy(numpy.array([[1, 2], [3, 4]], dtype=numpy.float32))
    with pytest.raises((ValueError, RuntimeError)):
        a.reshape((3, 3))
