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

"""Shared helpers for interop tests — NumPy/AsNumpy conversion and assertion utilities."""

from __future__ import annotations

import numpy as np


def to_numpy(value):
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def assert_numpy_equal(actual, expected):
    actual_np = to_numpy(actual)
    expected_np = np.asarray(expected)
    if actual_np.shape != expected_np.shape:
        raise AssertionError(f"shape mismatch: {actual_np.shape} != {expected_np.shape}")
    if actual_np.dtype != expected_np.dtype:
        raise AssertionError(f"dtype mismatch: {actual_np.dtype} != {expected_np.dtype}")
    np.testing.assert_array_equal(actual_np, expected_np)


def assert_numpy_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    actual_np = to_numpy(actual)
    expected_np = np.asarray(expected)
    if actual_np.shape != expected_np.shape:
        raise AssertionError(f"shape mismatch: {actual_np.shape} != {expected_np.shape}")
    if actual_np.dtype != expected_np.dtype:
        raise AssertionError(f"dtype mismatch: {actual_np.dtype} != {expected_np.dtype}")
    np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, equal_nan=True)
