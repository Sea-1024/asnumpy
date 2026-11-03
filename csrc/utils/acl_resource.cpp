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

#include "asnumpy/utils/acl_resource.hpp"
#include <acl/acl.h>
#include <spdlog/spdlog.h>
#include "asnumpy/utils/device_memory_pool.hpp"
#include "asnumpy/utils/status_handler.hpp"

namespace asnumpy {

// ============================================================================
// AclWorkspace
// ============================================================================

AclWorkspace::AclWorkspace(uint64_t size) : size_(size) {
    if (size_ > 0ULL) {
        // Through the pool: workspaces are allocated and freed around every
        // operator call, which is exactly the churn the pool exists to absorb.
        ptr_ = DeviceMemoryPool::Allocate(static_cast<size_t>(size_));
        spdlog::info("AclWorkspace allocated {} bytes", size_);
    }
}

AclWorkspace::~AclWorkspace() {
    if (ptr_) {
        DeviceMemoryPool::Release(ptr_, static_cast<size_t>(size_));
    }
}

AclWorkspace::AclWorkspace(AclWorkspace&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

AclWorkspace& AclWorkspace::operator=(AclWorkspace&& other) noexcept {
    if (this != &other) {
        // Free current resource
        if (ptr_) {
            DeviceMemoryPool::Release(ptr_, static_cast<size_t>(size_));
        }
        // Take ownership of other resource
        ptr_ = other.ptr_;
        size_ = other.size_;
        // Clear other object
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

} // namespace asnumpy
