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
#include <aclnn/aclnn_base.h>
#include <cstdint>
#include <utility>

namespace asnumpy {

/**
 * @brief RAII wrapper class for managing CANN workspace memory
 *
 * Automatically manages workspace allocation and deallocation to ensure exception safety.
 * Copy is prohibited, only move semantics are allowed.
 *
 * Example usage:
 * @code
 * AclWorkspace workspace(workspaceSize);  // Allocate workspace
 * // Use workspace.get() to get the pointer
 * // Automatically freed when goes out of scope
 * @endcode
 */
class AclWorkspace {
  public:
    /**
     * @brief Constructor, allocates workspace of specified size
     * @param size workspace size in bytes, if 0 no allocation is made
     * @throws std::runtime_error if memory allocation fails
     */
    explicit AclWorkspace(uint64_t size);

    /**
     * @brief Destructor, automatically frees workspace
     */
    ~AclWorkspace();

    // Disable copy
    AclWorkspace(const AclWorkspace&) = delete;
    AclWorkspace& operator=(const AclWorkspace&) = delete;

    // Allow move
    AclWorkspace(AclWorkspace&& other) noexcept;
    AclWorkspace& operator=(AclWorkspace&& other) noexcept;

    /**
     * @brief Get workspace pointer
     * @return workspace address, nullptr if not allocated
     */
    void* get() const noexcept { return ptr_; }

    /**
     * @brief Get workspace size
     * @return workspace size in bytes
     */
    uint64_t size() const noexcept { return size_; }

    /**
     * @brief Check if workspace is allocated
     * @return true if allocated
     */
    bool valid() const noexcept { return ptr_ != nullptr; }

  private:
    void* ptr_ = nullptr;
    uint64_t size_ = 0;
};

/**
 * @brief RAII guard for aclIntArray handles
 *
 * Destroys the handle on scope exit even when op checks (ACLNN_CHECK / ACL_RT_CHECK) throw,
 * so error paths no longer leak the descriptor.
 */
class AclIntArrayGuard {
  public:
    explicit AclIntArrayGuard(aclIntArray* ptr) noexcept : ptr_(ptr) {}
    ~AclIntArrayGuard() { aclDestroyIntArray(ptr_); }

    // Disable copy and move
    AclIntArrayGuard(const AclIntArrayGuard&) = delete;
    AclIntArrayGuard& operator=(const AclIntArrayGuard&) = delete;

    aclIntArray* get() const noexcept { return ptr_; }

  private:
    aclIntArray* ptr_;
};

/**
 * @brief RAII guard for aclScalar handles
 *
 * Destroys the handle on scope exit even when op checks (ACLNN_CHECK / ACL_RT_CHECK) throw.
 */
class AclScalarGuard {
  public:
    explicit AclScalarGuard(aclScalar* ptr) noexcept : ptr_(ptr) {}
    ~AclScalarGuard() { aclDestroyScalar(ptr_); }

    // Disable copy and move
    AclScalarGuard(const AclScalarGuard&) = delete;
    AclScalarGuard& operator=(const AclScalarGuard&) = delete;

    aclScalar* get() const noexcept { return ptr_; }

  private:
    aclScalar* ptr_;
};

} // namespace asnumpy
