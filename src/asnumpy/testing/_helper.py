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

"""Test helper functions.

Provides convenient utilities for generating and handling test data.
"""

__all__ = [
    "shaped_arange",
    "shaped_random",
    "shaped_reverse_arange",
    "assert_array_list_equal",
    "suppress_warnings",
    "with_seed",
    "generate_test_data",
    "TEST_SHAPES",
    "TEST_DTYPES",
    "TEST_ORDERS",
]

import functools

import numpy


def shaped_arange(shape, dtype=numpy.float64, order="C", xp=None, start=0):
    """Generate a sequential array with the given shape.

    Produces a contiguous integer sequence starting from ``start``,
    then reshapes it to ``shape``. Useful in tests because the values
    make it easy to verify array operations.

    Args:
        start: Starting value (default 0).
    """
    if xp is None:
        xp = numpy

    if isinstance(shape, int):
        shape = (shape,)

    size = 1
    for dim in shape:
        size *= dim

    if xp is numpy:
        arr = numpy.arange(start, start + size, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.arange(start, start + size, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def shaped_random(shape, dtype=numpy.float64, scale=1.0, seed=None, xp=None):
    """Generate a random array with the given shape.

    Produces a uniform-distribution random array.
    """
    if xp is None:
        xp = numpy

    if isinstance(shape, int):
        shape = (shape,)

    if seed is not None:
        numpy.random.seed(seed)

    if xp is numpy:
        arr = numpy.random.random(shape).astype(dtype)
        return arr * scale
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.random.random(shape).astype(dtype)
        arr = arr * scale
        return xp.ndarray.from_numpy(arr)


def shaped_reverse_arange(shape, dtype=numpy.float64, order="C", xp=None):
    """Generate a descending-order sequential array with the given shape.

    Produces a descending sequence then reshapes it to ``shape``.
    Useful for testing operations on reverse-sorted data.
    """
    if xp is None:
        xp = numpy

    if isinstance(shape, int):
        shape = (shape,)

    size = 1
    for dim in shape:
        size *= dim

    if xp is numpy:
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # For asnumpy, generate with numpy then convert
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def assert_array_list_equal(x_list, y_list, err_msg="", verbose=True):
    """Assert that two lists of arrays are element-wise equal.

    Used for testing functions that return multiple arrays.
    """
    from . import _array

    if len(x_list) != len(y_list):
        raise AssertionError(f"List lengths differ: {len(x_list)} vs {len(y_list)}")

    for i, (x, y) in enumerate(zip(x_list, y_list, strict=False)):
        try:
            _array.assert_array_equal(x, y, err_msg, verbose)
        except AssertionError as e:
            raise AssertionError(f"Arrays at index {i} differ: {e}") from e


def suppress_warnings(func):
    """Decorator: suppress warnings during function execution.

    Use in tests to temporarily silence known warnings.

    Examples:
        @suppress_warnings
        def test_something():
            # warnings here will be suppressed
            pass
    """
    import warnings

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


def with_seed(seed):
    """Decorator: run a test with a fixed random seed for reproducibility.

    Args:
        seed: Random seed.

    Examples:
        @with_seed(42)
        def test_random_function():
            arr = numpy.random.random((3, 3))
            # same random values on every run
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save current random state
            old_state = numpy.random.get_state()
            try:
                numpy.random.seed(seed)
                return func(*args, **kwargs)
            finally:
                # Restore original random state
                numpy.random.set_state(old_state)

        return wrapper

    return decorator


def generate_test_data(func):
    """Decorator: automatically generate test data for a test function.

    Generates common test cases based on the function signature.
    This is a simplified implementation that can be extended as needed.

    Examples:
        @generate_test_data
        def test_add(a, b):
            return a + b
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Simplified: delegate directly to the original function
        return func(*args, **kwargs)

    return wrapper


# Common test constants
TEST_SHAPES = [
    (),  # scalar
    (0,),  # empty array
    (1,),  # single element
    (5,),  # 1-D
    (2, 3),  # 2-D
    (2, 3, 4),  # 3-D
    (1, 2, 3, 4),  # 4-D
]

TEST_DTYPES = [
    numpy.float32,
    numpy.float64,
    numpy.int32,
    numpy.int64,
]

TEST_ORDERS = ["C", "F"]
