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

"""Baseline tests locking NumPy 2.x as the required runtime version."""

import numpy as np


def test_numpy_major_version_is_two_or_newer():
    major = int(np.__version__.split(".", 1)[0])
    if major < 2:
        raise AssertionError(
            f"NumPy {np.__version__} does not meet the baseline requirement (>= 2.0). "
            f"Upgrade: pip install 'numpy>=2.0'"
        )


def test_numpy_result_type_is_the_dtype_oracle():
    # int32 + float scalar → float64
    if np.result_type(np.array([1], dtype=np.int32), 1.5) != np.dtype("float64"):
        raise AssertionError("np.result_type(int32_array, float_scalar) should be float64")

    # float32 + float64 scalar → float64 (NumPy safe-casting rule)
    if np.result_type(np.array([1], dtype=np.float32), np.float64(1)) != np.dtype("float64"):
        raise AssertionError("np.result_type(float32_array, float64_scalar) should be float64")
