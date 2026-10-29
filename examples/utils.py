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


import numpy as np


def create_arrays(shape: tuple[int, ...], dtype: np.dtype):
    """创建asnumpy和numpy测试数组"""
    # numpy 基准数据
    m1_np = np.random.normal(0, 1, shape).astype(dtype)
    m2_np = np.random.normal(0, 1, shape).astype(dtype)

    # asnumpy测试数据 - 从 numpy 转换
    import asnumpy as ap

    m1_asnp = ap.ndarray.from_numpy(m1_np)
    m2_asnp = ap.ndarray.from_numpy(m2_np)

    return m1_asnp, m2_asnp, m1_np, m2_np


def calculate_stable_metric(times: list, trim_ratio: float = 0.1) -> float:
    """
    统计策略：取中段最快速度
    1. 排序去除最慢的 10% (受系统调度影响的数据)
    2. 取剩余数据的最小值 (代表硬件峰值性能)
    """
    if not times:
        return 0.0

    sorted_times = sorted(times)
    keep_count = int(len(sorted_times) * (1.0 - trim_ratio))
    if keep_count < 1:
        keep_count = 1

    valid_times = sorted_times[:keep_count]
    return min(valid_times)
