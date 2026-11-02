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

"""Array concatenation and stacking tests."""

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
def test_concatenate_1d(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5], dtype)
    return xp.concatenate([a, b])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_concatenate_2d_axis0(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5, 6]], dtype)
    return xp.concatenate([a, b], axis=0)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_concatenate_2d_axis1(xp, dtype):
    """Axis-1 joins are the interleaving case, not a flat copy."""
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5], [6]], dtype)
    return xp.concatenate([a, b], axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_concatenate_negative_axis(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5], [6]], dtype)
    return xp.concatenate([a, b], axis=-1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_concatenate_three_arrays(xp, dtype):
    a = _create_array(xp, [1, 2], dtype)
    b = _create_array(xp, [3], dtype)
    c = _create_array(xp, [4, 5, 6], dtype)
    return xp.concatenate([a, b, c])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_concatenate_3d_middle_axis(xp, dtype):
    a = _create_array(xp, numpy.arange(12).reshape(2, 3, 2), dtype)
    b = _create_array(xp, numpy.arange(8).reshape(2, 2, 2), dtype)
    return xp.concatenate([a, b], axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_stack_1d(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    return xp.stack([a, b])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_stack_axis1(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    return xp.stack([a, b], axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_stack_2d_axis1(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5, 6], [7, 8]], dtype)
    return xp.stack([a, b], axis=1)


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_vstack_1d(xp, dtype):
    """1-D inputs must be promoted to rows, giving (2, 3) not (6,)."""
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5, 6], dtype)
    return xp.vstack([a, b])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_vstack_2d(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5, 6]], dtype)
    return xp.vstack([a, b])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_hstack_1d(xp, dtype):
    a = _create_array(xp, [1, 2, 3], dtype)
    b = _create_array(xp, [4, 5], dtype)
    return xp.hstack([a, b])


@testing.for_dtypes([numpy.float32, numpy.int32])
@testing.numpy_asnumpy_array_equal()
def test_hstack_2d(xp, dtype):
    a = _create_array(xp, [[1, 2], [3, 4]], dtype)
    b = _create_array(xp, [[5], [6]], dtype)
    return xp.hstack([a, b])


def test_concatenate_shape_mismatch_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2, 3), dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.zeros((2, 4), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.concatenate([a, b], axis=0)


def test_concatenate_dtype_mismatch_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2,), dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.zeros((2,), dtype=numpy.int32))
    with pytest.raises(ValueError):
        ap.concatenate([a, b])


def test_concatenate_axis_out_of_range_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.concatenate([a], axis=3)


def test_concatenate_empty_sequence_raises():
    import asnumpy as ap

    with pytest.raises(ValueError):
        ap.concatenate([])


def test_stack_shape_mismatch_raises():
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2,), dtype=numpy.float32))
    b = ap.ndarray.from_numpy(numpy.zeros((3,), dtype=numpy.float32))
    with pytest.raises(ValueError):
        ap.stack([a, b])


def test_concatenate_rejects_host_arrays():
    """Host data must be transferred explicitly, not implicitly."""
    import asnumpy as ap

    a = ap.ndarray.from_numpy(numpy.zeros((2,), dtype=numpy.float32))
    with pytest.raises(TypeError):
        ap.concatenate([a, numpy.zeros((2,), dtype=numpy.float32)])
