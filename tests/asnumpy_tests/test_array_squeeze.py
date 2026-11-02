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

"""squeeze and expand_dims tests."""

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
def test_squeeze_all(xp, dtype):
    a = _create_array(xp, numpy.arange(3).reshape(1, 3, 1), dtype)
    return xp.squeeze(a)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_specific_axis(xp, dtype):
    a = _create_array(xp, numpy.arange(3).reshape(1, 3, 1), dtype)
    return xp.squeeze(a, axis=0)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_negative_axis(xp, dtype):
    a = _create_array(xp, numpy.arange(3).reshape(1, 3, 1), dtype)
    return xp.squeeze(a, axis=-1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_multiple_axes(xp, dtype):
    a = _create_array(xp, numpy.arange(3).reshape(1, 3, 1), dtype)
    return xp.squeeze(a, axis=(0, 2))


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_no_unit_axes_is_noop(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return xp.squeeze(a)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_method(xp, dtype):
    """a.squeeze() must work as a method, not only as a free function."""
    a = _create_array(xp, numpy.arange(3).reshape(1, 3, 1), dtype)
    return a.squeeze()


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_front(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    return xp.expand_dims(a, axis=0)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_back(xp, dtype):
    """axis == ndim is valid: the new axis goes one past the last."""
    a = _create_array(xp, [1, 2, 3], dtype)
    return xp.expand_dims(a, axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_middle(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return xp.expand_dims(a, axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_expand_dims_negative(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    return xp.expand_dims(a, axis=-1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_squeeze_expand_dims_roundtrip(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    return xp.squeeze(xp.expand_dims(a, axis=0))


def test_squeeze_shapes():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.arange(3, dtype=numpy.float32).reshape(1, 3, 1))
    assert ap.squeeze(a).shape == (3,)
    assert ap.squeeze(a, axis=0).shape == (3, 1)
    assert ap.expand_dims(a, axis=0).shape == (1, 1, 3, 1)


def test_squeeze_all_unit_axes_gives_0d():
    """A fully length-1 array squeezes to 0-d, as in NumPy."""
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.ones((1, 1, 1), dtype=numpy.float32))
    assert ap.squeeze(a).shape == ()
    assert numpy.squeeze(numpy.ones((1, 1, 1), dtype=numpy.float32)).shape == ()


def test_squeeze_non_unit_axis_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.squeeze(a, axis=1)


def test_squeeze_axis_out_of_range_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((1, 3), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.squeeze(a, axis=5)


def test_expand_dims_out_of_range_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((3,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.expand_dims(a, axis=3)
