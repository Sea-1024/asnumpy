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

import gc
import time

import numpy as np
from loguru import logger
from utils import calculate_stable_metric, create_arrays

import asnumpy as ap


def bench_vdot(vdot_func, m1, m2, warmup: int, iterations: int, is_npu: bool = False) -> list:
    """
    Benchmark function for vdot operation.
    Memory-optimized: explicitly delete intermediate variables to avoid OOM.
    """
    # 1. Warmup phase
    for _ in range(warmup):
        res = vdot_func(m1, m2)
        del res  # Release immediately

    # 2. Benchmark phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        # Execute computation
        res = vdot_func(m1, m2)
        end = time.perf_counter()
        times.append(end - start)

        # Key optimization: explicitly delete result objects to prevent memory accumulation
        del res

    # Force garbage collection
    if is_npu:
        gc.collect()

    return times


def run_test_case(
    shape: tuple[int, ...], dtype: np.dtype = np.float32, warmup: int = 40, iterations: int = 400
) -> dict[str, float]:
    """Run a single test case"""
    logger.info(f"{'=' * 50}")
    logger.info(f"Test shape: {shape}")

    m1_asnp, m2_asnp, m1_np, m2_np = create_arrays(shape, dtype)

    try:
        # --- Benchmark AsNumpy ---
        asnp_times = bench_vdot(
            ap.vdot, m1_asnp, m2_asnp, warmup=warmup, iterations=iterations, is_npu=True
        )

        # --- Benchmark NumPy ---
        np_times = bench_vdot(
            np.vdot, m1_np, m2_np, warmup=warmup, iterations=iterations, is_npu=False
        )

        # Calculate statistics
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0

        # Verify result consistency
        result_asnp = ap.vdot(m1_asnp, m2_asnp).to_numpy()
        result_np = np.vdot(m1_np, m2_np)

        max_diff = np.max(np.abs(result_asnp - result_np))
        max_val = np.max(np.abs(result_np))
        rel_diff = max_diff / max_val if max_val > 0 else max_diff

        if rel_diff < 1e-4:
            logger.info(
                f"Verification passed: results are consistent (relative diff: {rel_diff:.2e})"
            )
        else:
            logger.warning(f"Results differ (max relative diff: {rel_diff:.2e})")

        return {
            "shape": shape,
            "asnumpy_metric": metric_asnp,
            "numpy_metric": metric_np,
            "speedup": speedup,
            "relative_diff": rel_diff,
        }

    finally:
        # Explicitly clean up large objects for current shape
        del m1_asnp, m2_asnp, m1_np, m2_np
        gc.collect()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("README example code performance benchmark")
    logger.info("Test operation: vdot (vector dot product)")
    logger.info(
        "Statistics strategy: after warmup, take mid-segment fastest speed (exclude slowest 10%)"
    )
    logger.info("=" * 70)

    # Test configuration
    shapes = [
        (500, 500),  # Medium scale
        (1000, 1000),  # Large scale test
        (2000, 2000),  # Larger scale test
        (3000, 3000),  # Extra large scale test
    ]
    dtype = np.dtype(np.float32)

    # Adjusted parameters: reduced iterations to fit NPU memory limits
    warmup_iterations = 40
    test_iterations = 400

    logger.info("\nConfiguration:")
    logger.info(f"  Data type: {dtype}")
    logger.info(f"  Warmup iterations: {warmup_iterations}")
    logger.info(f"  Test iterations: {test_iterations}")
    logger.info("  Statistics method: sort, exclude slowest 10%, take minimum")
    logger.info(f"\n{'=' * 70}\n")

    results = []
    for shape in shapes:
        try:
            result = run_test_case(shape, dtype, warmup_iterations, test_iterations)
            results.append(result)
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Output summary results
    print(f"{'Shape':<15} | {'Data Size':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'Speedup':<10}")
    print(f"{'':15} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 85)

    for result in results:
        shape_str = str(result["shape"])
        data_size = np.prod(result["shape"])
        data_size_str = f"{data_size:,}"
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"

        print(
            f"{shape_str:<15} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}"
        )

    print("-" * 85)

    # Statistics
    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)

        logger.info(f"\n{'=' * 70}")
        logger.info("Performance statistics:")
        logger.info(f"  Average speedup: {avg_speedup:.2f}x")
        logger.info(f"  Maximum speedup: {max_speedup:.2f}x")
        logger.info(f"{'=' * 70}")

    logger.info("\nNotes:")
    logger.info("  • Using float32 data type for NPU compatibility")
    logger.info("  • vdot flattens input arrays and computes vector dot product")
    logger.info("  • Iterations adjusted to ensure memory safety")
    logger.info(
        "  • Uses 'mid-segment fastest speed' algorithm to showcase NPU's true compute capability"
    )
    logger.info("=" * 70)
