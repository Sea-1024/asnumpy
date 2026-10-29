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


def create_arrays(shape: tuple[int, ...], dtype: np.dtype):
    """Create asnumpy and numpy test arrays"""
    # NumPy baseline data
    if dtype == np.bool_:
        # For bool type, generate random boolean values (~50% True, ~50% False)
        m1_np = np.random.rand(*shape) > 0.5
    else:
        # For other types, generate random data
        m1_np = np.random.rand(*shape).astype(dtype)

    # AsNumpy test data - converted from NumPy
    m1_asnp = ap.ndarray.from_numpy(m1_np)

    return m1_asnp, m1_np


def bench_all(all_func, m1, warmup: int, iterations: int, is_npu: bool = False) -> list:
    """
    Benchmark function for all reduction operation.
    Memory-optimized: explicitly delete intermediate variables to avoid OOM.
    """
    # 1. Warmup phase
    for _ in range(warmup):
        res = all_func(m1)
        del res  # Release immediately

    # 2. Benchmark phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()

        # Execute computation
        res = all_func(m1)
        end = time.perf_counter()
        times.append(end - start)

        # Explicitly delete result objects (all returns scalar, but keep consistent)
        del res

    # Force garbage collection
    if is_npu:
        gc.collect()

    return times


def run_test_case(
    shape: tuple[int, ...],
    dtype: np.dtype,
    warmup: int = 40,
    iterations: int = 400,
) -> dict[str, float]:
    """Run a single test case"""
    logger.info(f"{'=' * 50}")
    logger.info(f"Test shape: {shape}, dtype: {dtype}")

    m1_asnp, m1_np = create_arrays(shape, dtype)

    try:
        # --- Benchmark AsNumpy ---
        asnp_times = bench_all(ap.all, m1_asnp, warmup=warmup, iterations=iterations, is_npu=True)

        # --- Benchmark NumPy ---
        np_times = bench_all(np.all, m1_np, warmup=warmup, iterations=iterations, is_npu=False)

        # Calculate statistics
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0

        # Verify result consistency
        result_asnp = ap.all(m1_asnp)
        # all() returns a scalar, may need conversion
        if hasattr(result_asnp, "to_numpy"):
            result_asnp = result_asnp.to_numpy()
        result_np = np.all(m1_np)

        # For bool results, compare directly
        if result_asnp == result_np:
            logger.info(f"Verification passed: results are consistent (result: {result_np})")
        else:
            logger.warning(f"Results differ (NumPy: {result_np}, AsNumpy: {result_asnp})")

        return {
            "shape": shape,
            "dtype": str(dtype),
            "asnumpy_metric": metric_asnp,
            "numpy_metric": metric_np,
            "speedup": speedup,
        }

    finally:
        # Explicitly clean up large objects for current shape
        del m1_asnp, m1_np
        gc.collect()


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("README example code performance benchmark")
    logger.info("Test operation: all (reduction - check if all elements are True)")
    logger.info(
        "Statistics strategy: after warmup, take mid-segment fastest speed (exclude slowest 10%)"
    )
    logger.info("=" * 70)

    # Test configuration - multiple shapes and data types
    test_configs: list[tuple[tuple[int, ...], np.dtype]] = [
        # bool type tests
        ((1000, 1000), np.bool_),
        ((2000, 2000), np.bool_),
        ((3000, 3000), np.bool_),
        # float32 type tests
        ((1000, 1000), np.float32),
        ((2000, 2000), np.float32),
        ((3000, 3000), np.float32),
    ]

    # Adjusted parameters: reduced iterations to fit NPU memory limits
    warmup_iterations = 40
    test_iterations = 400

    logger.info("\nConfiguration:")
    logger.info(f"  Warmup iterations: {warmup_iterations}")
    logger.info(f"  Test iterations: {test_iterations}")
    logger.info("  Statistics method: sort, exclude slowest 10%, take minimum")
    logger.info(f"\n{'=' * 70}\n")

    results = []
    for shape, dtype in test_configs:
        try:
            result = run_test_case(shape, dtype, warmup_iterations, test_iterations)
            results.append(result)
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback

            traceback.print_exc()

    # Output summary results
    print("\n" + "=" * 95)
    print("Test results summary (based on mid-segment fastest speed)")
    print("-" * 95)
    print(
        f"{'Shape':<15} | {'Type':<10} | {'Data Size':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'Speedup':<10}"
    )
    print(f"{'':15} | {'':10} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 95)

    for result in results:
        shape_str = str(result["shape"])
        dtype_str = result["dtype"]
        data_size = np.prod(result["shape"])
        data_size_str = f"{data_size:,}"
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"

        print(
            f"{shape_str:<15} | {dtype_str:<10} | {data_size_str:<12} "
            f"| {asnp_time:<12} | {np_time:<12} | {speedup_str}"
        )

    print("-" * 95)

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
    logger.info("  • Testing both bool and float32 data types")
    logger.info("  • all() is a reduction operation that returns a scalar result")
    logger.info("  • Iterations adjusted to ensure memory safety")
    logger.info(
        "  • Uses 'mid-segment fastest speed' algorithm to showcase NPU's true compute capability"
    )
    logger.info("=" * 70)
