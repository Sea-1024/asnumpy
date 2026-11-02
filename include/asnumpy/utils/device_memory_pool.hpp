/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once

#include <acl/acl.h>
#include <cstddef>

namespace asnumpy {

/**
 * @brief Size-bucketed cache in front of aclrtMalloc / aclrtFree.
 *
 * aclrtMalloc is expensive and its cost is not proportional to the request: a
 * 64 MB allocation measured 399 us while a 4 MB one measured 15 us, because the
 * driver rounds large requests up to a huge-page-backed pool. Every operator
 * allocates at least one output, so that cost lands on every call.
 *
 * Returning freed blocks to a cache instead of the driver makes a repeated
 * allocate/free cycle of a given size cost a hash lookup.
 *
 * Blocks are matched on exact byte size, never split or coalesced. That keeps
 * the bookkeeping trivial and cannot hand back a block smaller than requested,
 * which is the failure mode that matters here.
 *
 * Cached memory stays resident. The cache is capped (default 256 MB) and evicts
 * largest-first once a release would exceed the cap, so a long-lived process
 * does not grow without bound.
 *
 * Thread-safe. Deliberately not tied to a device context: callers are expected
 * to have an active context, and the pool is only ever reached through the
 * allocate/free pair below. Clear() must be called before a device reset,
 * because the pointers it holds are invalid afterwards.
 */
class DeviceMemoryPool {
  public:
    /**
     * @brief Allocate at least @p bytes of device memory.
     *
     * Falls back to aclrtMalloc on a cache miss. @p bytes == 0 returns nullptr
     * without touching the driver, matching aclrtMalloc's own behavior for the
     * zero-size case being useless to call.
     *
     * @throws std::runtime_error If aclrtMalloc fails.
     */
    static void* Allocate(size_t bytes);

    /**
     * @brief Return a block obtained from Allocate() to the cache.
     *
     * @p bytes must be the same value that was passed to Allocate(). A null
     * pointer or zero size is a no-op, so callers need not special-case empty
     * tensors.
     */
    static void Release(void* ptr, size_t bytes) noexcept;

    /**
     * @brief Free every cached block back to the driver.
     *
     * Must be called before aclrtResetDevice or aclFinalize; cached pointers do
     * not survive either.
     */
    static void Clear() noexcept;

    /** @brief Bytes currently held in the cache. */
    static size_t CachedBytes() noexcept;

    /** @brief Allocations served from the cache since start. */
    static size_t HitCount() noexcept;

    /** @brief Allocations that had to reach the driver. */
    static size_t MissCount() noexcept;
};

} // namespace asnumpy
