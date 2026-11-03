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

#include <asnumpy/utils/device_memory_pool.hpp>

#include <fmt/core.h>
#include <map>
#include <mutex>
#include <vector>

namespace asnumpy {

namespace {

// One free list per exact byte size. A std::map keyed on size keeps eviction
// (largest first) cheap without a second index.
using FreeList = std::map<size_t, std::vector<void*>>;

struct PoolState {
    std::mutex mutex;
    FreeList free_lists;
    size_t cached_bytes = 0;
    size_t hits = 0;
    size_t misses = 0;
    // Set by Clear(). Once the CANN runtime has been torn down there is nothing
    // left that can free a block, so Release() must drop rather than cache.
    bool closed = false;
};

PoolState& state() {
    // Function-local static: constructed on first use, so the pool does not
    // depend on static-initialization order relative to CANN's own globals.
    static PoolState instance;
    return instance;
}

// 256 MB. Large enough to cover the working set of a typical chain of
// operators, small enough that holding it does not meaningfully change the
// memory a process is seen to use.
constexpr size_t kMaxCachedBytes = 256ull * 1024ull * 1024ull;

} // namespace

void* DeviceMemoryPool::Allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }

    {
        auto& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto it = s.free_lists.find(bytes);
        if (it != s.free_lists.end() && !it->second.empty()) {
            void* ptr = it->second.back();
            it->second.pop_back();
            s.cached_bytes -= bytes;
            ++s.hits;
            return ptr;
        }
        ++s.misses;
    }

    void* ptr = nullptr;
    auto error = aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (error != ACL_SUCCESS) {
        auto message = aclGetRecentErrMsg();
        std::string detail = message ? std::string(" - ") + message : "";
        throw std::runtime_error(
            fmt::format("aclrtMalloc({} bytes) failed, error = {}{}", bytes, static_cast<int>(error), detail));
    }
    return ptr;
}

void DeviceMemoryPool::Release(void* ptr, size_t bytes) noexcept {
    if (ptr == nullptr || bytes == 0) {
        return;
    }

    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);

    // After Clear() the runtime may already be gone; calling aclrtFree would be
    // at best a no-op and at worst a crash during interpreter shutdown. Leaking
    // the block is the only safe option, and the process is ending anyway.
    if (s.closed) {
        return;
    }

    // Evict largest-first until this block fits under the cap. Evicting by
    // descending size frees the most bytes per driver call.
    while (s.cached_bytes + bytes > kMaxCachedBytes && !s.free_lists.empty()) {
        auto largest = std::prev(s.free_lists.end());
        auto& list = largest->second;
        if (list.empty()) {
            s.free_lists.erase(largest);
            continue;
        }
        void* victim = list.back();
        list.pop_back();
        size_t victim_bytes = largest->first;
        s.cached_bytes -= victim_bytes;
        aclrtFree(victim);
        if (list.empty()) {
            s.free_lists.erase(largest);
        }
    }

    // If a single block is larger than the whole cap, it cannot be cached
    // without evicting everything and still overflowing; hand it straight back.
    if (bytes > kMaxCachedBytes) {
        aclrtFree(ptr);
        return;
    }

    s.free_lists[bytes].push_back(ptr);
    s.cached_bytes += bytes;
}

void DeviceMemoryPool::Clear() noexcept {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    for (auto& [bytes, list] : s.free_lists) {
        for (void* ptr : list) {
            aclrtFree(ptr);
        }
    }
    s.free_lists.clear();
    s.cached_bytes = 0;
    s.closed = true;
}

size_t DeviceMemoryPool::CachedBytes() noexcept {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.cached_bytes;
}

size_t DeviceMemoryPool::HitCount() noexcept {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.hits;
}

size_t DeviceMemoryPool::MissCount() noexcept {
    auto& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return s.misses;
}

} // namespace asnumpy
