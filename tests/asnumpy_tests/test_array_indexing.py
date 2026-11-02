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

"""Basic integer indexing tests."""

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
def test_getitem_1d(xp, dtype):
    a = _create_array(xp, [10, 20, 30, 40], dtype)
    return a[2]


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_getitem_2d_row(xp, dtype):
    """a[i] on a 2-D array keeps the trailing axis."""
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a[1]


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_getitem_2d_scalar(xp, dtype):
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a[1, 2]


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_getitem_negative_index(xp, dtype):
    a = _create_array(xp, [10, 20, 30, 40], dtype)
    return a[-1]


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_getitem_negative_2d(xp, dtype):
    a = _create_array(xp, [[1, 2, 3], [4, 5, 6]], dtype)
    return a[-1, -2]


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_getitem_3d_partial(xp, dtype):
    """Indexing fewer axes than ndim keeps the remaining ones."""
    a = _create_array(xp, numpy.arange(24).reshape(2, 3, 4), dtype)
    return a[1, 2]


def test_getitem_out_of_bounds_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.array([1, 2, 3], dtype=numpy.float32))
    with pytest.raises(IndexError):
        a[5]
    with pytest.raises(IndexError):
        a[-4]


def test_getitem_too_many_indices_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.array([1, 2, 3], dtype=numpy.float32))
    with pytest.raises(IndexError):
        a[0, 0]


def test_getitem_slice_rejected():
    """Slices are not supported yet; they must raise, not mis-index."""
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.array([1, 2, 3], dtype=numpy.float32))
    with pytest.raises(TypeError):
        a[0:2]
