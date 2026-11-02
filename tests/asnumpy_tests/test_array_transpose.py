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

"""Transpose and axis-swap tests."""

import numpy
import pytest

from asnumpy import testing


def _create_array(xp, data, dtype):
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_2d(xp, dtype):
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a.transpose()


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_T_property(xp, dtype):
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a.T


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_3d_default(xp, dtype):
    """Default transpose reverses all axes: (2,3,4) -> (4,3,2)."""
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.transpose()


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_3d_explicit_axes(xp, dtype):
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.transpose((1, 0, 2))


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_varargs(xp, dtype):
    """NumPy accepts a.transpose(1, 0, 2) as well as a tuple."""
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.transpose(2, 0, 1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_negative_axes(xp, dtype):
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.transpose((-2, -3, -1))


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_transpose_1d_is_identity(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    return a.transpose()


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_swapaxes_2d(xp, dtype):
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a.swapaxes(0, 1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_swapaxes_3d(xp, dtype):
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.swapaxes(0, 2)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_swapaxes_negative(xp, dtype):
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.swapaxes(-1, -3)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_double_transpose_roundtrip(xp, dtype):
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a.transpose().transpose()


def test_transpose_shape():
    """Shape must actually change, not just the contents."""
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4))
    assert a.transpose().shape == (4, 3, 2)
    assert a.transpose((1, 0, 2)).shape == (3, 2, 4)
    assert a.swapaxes(0, 1).shape == (3, 2, 4)
    assert a.T.shape == (4, 3, 2)


def test_transpose_repeated_axis_raises():
    """A repeated axis would silently drop another one."""
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2, 3, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        a.transpose((0, 0, 1))


def test_transpose_wrong_length_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    with pytest.raises(ValueError):
        a.transpose((0, 1, 2))


def test_swapaxes_out_of_range_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    with pytest.raises(ValueError):
        a.swapaxes(0, 5)
