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
from utils import calculate_stable_metric

import asnumpy as ap


def bench_linspace(
    linspace_func,
    start: float,
    end: float,
    steps: int,
    dtype: np.dtype,
    warmup: int,
    iterations: int,
    is_npu: bool = False,
) -> list:
    """
    Benchmark function for linspace operation.
    Memory-optimized: explicitly delete intermediate variables to avoid OOM.
    """
    # 1. Warmup phase
    for _ in range(warmup):
        res = linspace_func(start, end, steps, dtype=dtype)
        del res  # Release immediately

    # 2. Benchmark phase
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()

        # Execute computation
        res = linspace_func(start, end, steps, dtype=dtype)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        # Key optimization: explicitly delete result objects to prevent memory accumulation
        del res

    # Force garbage collection
    if is_npu:
        gc.collect()

    return times


def run_test_case(
    steps: int,
    start: float = 0.0,
    end: float = 100.0,
    dtype: np.dtype = np.float32,
    warmup: int = 40,
    iterations: int = 400,
) -> dict[str, float]:
    """Run a single test case"""
    logger.info(f"{'=' * 50}")
    logger.info(f"Test steps: {steps:,} (range: [{start}, {end}])")

    try:
        # --- Benchmark AsNumpy ---
        asnp_times = bench_linspace(
            ap.linspace, start, end, steps, dtype, warmup=warmup, iterations=iterations, is_npu=True
        )

        # --- Benchmark NumPy ---
        np_times = bench_linspace(
            np.linspace,
            start,
            end,
            steps,
            dtype,
            warmup=warmup,
            iterations=iterations,
            is_npu=False,
        )

        # Calculate statistics
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0

        # Verify result consistency
        result_asnp = ap.linspace(start, end, steps, dtype=dtype).to_numpy()
        result_np = np.linspace(start, end, steps, dtype=dtype)

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
            "steps": steps,
            "asnumpy_metric": metric_asnp,
            "numpy_metric": metric_np,
            "speedup": speedup,
            "relative_diff": rel_diff,
        }

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        # Force garbage collection
        gc.collect()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("README example code performance benchmark")
    logger.info("Test operation: linspace (create evenly spaced sequence)")
    logger.info(
        "Statistics strategy: after warmup, take mid-segment fastest speed (exclude slowest 10%)"
    )
    logger.info("=" * 70)

    # Test configuration - step counts correspond to data sizes of other examples
    test_steps = [
        250000,  # Corresponds to (500, 500) data size
        1000000,  # Corresponds to (1000, 1000) data size
        4000000,  # Corresponds to (2000, 2000) data size
        9000000,  # Corresponds to (3000, 3000) data size
    ]
    dtype = np.dtype(np.float32)
    start_val = 0.0
    end_val = 100.0

    # Adjusted parameters: reduced iterations to fit NPU memory limits
    warmup_iterations = 40
    test_iterations = 400

    logger.info("\nConfiguration:")
    logger.info(f"  Data type: {dtype}")
    logger.info(f"  Start value: {start_val}")
    logger.info(f"  End value: {end_val}")
    logger.info(f"  Warmup iterations: {warmup_iterations}")
    logger.info(f"  Test iterations: {test_iterations}")
    logger.info("  Statistics method: sort, exclude slowest 10%, take minimum")
    logger.info(f"\n{'=' * 70}\n")

    results = []
    for steps in test_steps:
        result = run_test_case(steps, start_val, end_val, dtype, warmup_iterations, test_iterations)
        if result is not None:
            results.append(result)

    # Output summary results
    print(f"{'Steps':<15} | {'Data Size':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'Speedup':<10}")
    print(f"{'':15} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 85)

    for result in results:
        steps_str = f"{result['steps']:,}"
        data_size_str = f"{result['steps']:,}"
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"

        print(
            f"{steps_str:<15} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}"
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
    logger.info("  • linspace generates evenly spaced 1D array within specified range")
    logger.info("  • Iterations adjusted to ensure memory safety")
    logger.info(
        "  • Uses 'mid-segment fastest speed' algorithm to showcase NPU's true compute capability"
    )
    logger.info("=" * 70)
