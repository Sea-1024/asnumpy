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
#include <algorithm>
#include <cstdint>
#include <fmt/core.h>
#include <iostream>
#include <memory>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

/**
 * @brief Refcounted owner of one device allocation.
 *
 * Split out of NPUArray so several arrays can share one allocation. A view
 * holds a shared_ptr to the same DeviceBuffer, so the memory outlives any
 * individual array that references it -- `a.reshape(...)` stays valid after
 * `a` is destroyed.
 *
 * The allocation is returned to asnumpy::DeviceMemoryPool on destruction, not
 * to the driver; see device_memory_pool.hpp for why that matters.
 */
class DeviceBuffer {
  public:
    DeviceBuffer(void* ptr, size_t bytes) : ptr_(ptr), bytes_(bytes) {}
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() const noexcept { return ptr_; }
    size_t bytes() const noexcept { return bytes_; }

  private:
    void* ptr_;
    size_t bytes_;
};

class NPUArray {
  public:
    aclTensor* tensorPtr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    py::dtype dtype;
    aclDataType aclDtype;
    size_t tensorSize;

  private:
    /**
     * @brief Storage shared with any views of this array.
     *
     * Null for zero-element arrays, which have no allocation.
     */
    std::shared_ptr<DeviceBuffer> buffer;

    /**
     * @brief Byte offset of this array's first element within `buffer`.
     *
     * Nonzero only for views produced by Reshape/Flatten/Ravel/GetItem, all of
     * which select a C-contiguous region and so keep every downstream
     * assumption (dense, contiguous) intact.
     */
    int64_t byteOffset;

    /**
     * @brief Build an array that shares `src`'s storage at `offset_bytes`.
     *
     * Private because it deliberately skips allocation: the caller must have
     * verified that the region is contiguous and in bounds.
     */
    NPUArray(const NPUArray& src, const std::vector<int64_t>& new_shape, int64_t offset_bytes);

  public:
    /**
     * @brief The logical start of this array's data on the device.
     *
     * This is `buffer + byteOffset`, not the allocation base, so an operator
     * never needs to know that its input is a view.
     */
    void* device_address() const {
        return buffer ? static_cast<char*>(buffer->data()) + byteOffset : nullptr;
    }

    /**
     * @brief Constructor to create an empty NPUArray from shape and data type
     * @param shape Tensor shape
     * @param dtype Data type
     */
    NPUArray(const std::vector<int64_t>& shape, py::dtype dtype);

    /**
     * @brief Constructor to create an empty NPUArray from shape and ACL data type
     * @param shape Tensor shape
     * @param acl_type ACL data type constant
     */
    NPUArray(const std::vector<int64_t>& shape, aclDataType acl_type);

    // Copy constructor - deep copy
    NPUArray(const NPUArray& other);

    // Move constructor
    NPUArray(NPUArray&& other) noexcept;

    // Copy assignment
    NPUArray& operator=(const NPUArray& other);

    // Move assignment
    NPUArray& operator=(NPUArray&& other) noexcept;

    //Destructor
    ~NPUArray();

    /**
     * @brief Create an NPUArray from a NumPy array
     *
     * This static method creates an NPUArray from a NumPy array
     * and copies the data from host memory to NPU device memory.
     *
     * @param host_data Input NumPy array.
     * @return NPUArray Created NPUArray.
     */
    static NPUArray FromNumpy(py::array host_data);

    /**
     * @brief Copy data from NPU to host and return a NumPy array
     * @return py::array Returned NumPy array
     */
    py::array ToNumpy() const;

    /**
     * @brief Create a view
     * @return std::unique_ptr<NPUArray> Returned view object
     */
    std::unique_ptr<NPUArray> View() const;

    /**
     * @brief Calculate the total size of the array.
     * @param shape Vector containing the dimensions of the array, defining its shape.
     * @return int64_t Total number of elements in the array.
     */
    static int64_t GetShapeSize(const std::vector<int64_t>& shape);

    /**
     * @brief Get the byte size of the specified aclDataType
     * @param dataType ACL data type.
     * @return int64_t Byte size of the data type.
     */
    static int64_t GetDataTypeSize(aclDataType dataType);

    /**
     * @brief Convert py::dtype to aclDataType
     * @param dtype Input py::dtype.
     * @return aclDataType Converted aclDataType.
     */
    static aclDataType GetACLDataType(py::dtype dtype);

    /**
     * @brief Convert aclDataType to py::dtype
     * @param acl_type Input aclDataType.
     * @return py::dtype Converted py::dtype.
     */
    static py::dtype GetPyDtype(aclDataType acl_type);

    /**
     * @brief Reshape array to new shape.
     *
     * Returns a view: no device memory is allocated and no bytes are copied,
     * matching NumPy and CuPy. The result shares storage with this array, so a
     * write through either would be visible through both -- the rule NumPy
     * documents for views. Nothing in asnumpy can currently write in place, but
     * the contract is stated here because adding __setitem__ or out= later must
     * not silently break it.
     *
     * @return NPUArray Reshaped view over the same storage.
     * @throws std::invalid_argument If the shapes hold different element counts,
     *         or more than one dimension is -1. Surfaces to Python as ValueError.
     */
    NPUArray Reshape(const std::vector<int64_t>& new_shape) const;

    /**
     * @brief Flatten array to 1D.
     *
     * A view, not a copy -- NumPy's flatten() copies but its ravel() does not;
     * asnumpy has no F-order arrays, so both are the same zero-copy operation
     * here. See Reshape.
     *
     * @return NPUArray 1-D view over the same storage.
     */
    NPUArray Flatten() const;

    /**
     * @brief Ravel array (alias for flatten).
     * @return NPUArray 1-D view over the same storage.
     */
    NPUArray Ravel() const;

    /**
     * @brief Permute the array's axes.
     *
     * Mirrors numpy.ndarray.transpose. `axes` must be a permutation of
     * [0, ndim). When empty, the axes are reversed, which is the NumPy default
     * and what the `.T` property uses. Negative entries wrap.
     *
     * @param axes Target axis order, or empty for a full reversal.
     * @return NPUArray Newly allocated array with permuted axes. Transpose
     *         materializes the permutation rather than returning a strided
     *         view, because downstream operators assume C-contiguous input.
     * @throws std::invalid_argument If `axes` is not a permutation of the
     *         array's axes. Surfaces to Python as ValueError.
     */
    NPUArray Transpose(const std::vector<int64_t>& axes) const;

    /**
     * @brief Interchange two axes.
     *
     * @param axis1 First axis; negative values wrap.
     * @param axis2 Second axis; negative values wrap.
     * @return NPUArray Newly allocated array with the two axes swapped.
     * @throws std::invalid_argument If either axis is out of range.
     */
    NPUArray SwapAxes(int64_t axis1, int64_t axis2) const;

    /**
     * @brief Index the array with a leading run of integer indices.
     *
     * Mirrors NumPy basic integer indexing: `indices` must not be longer than
     * ndim, and each entry selects along the corresponding leading axis.
     * Negative indices wrap, as in NumPy. The remaining trailing axes are kept,
     * so a[i] on a (2,3) array returns a (3,) array and a[i, j] returns a 0-d
     * array. Because asnumpy arrays are dense and C-contiguous, the selection is
     * one contiguous device block, copied out rather than viewed.
     *
     * @param indices Integer index per leading axis.
     * @return NPUArray Newly allocated array holding the selected sub-block.
     * @throws std::out_of_range If too many indices are given or an index is out
     *         of bounds. Surfaces to Python as IndexError.
     */
    NPUArray GetItem(const std::vector<int64_t>& indices) const;

    // The former declarations View/Copy/Fill/ToList/ToArray/UpdateContiguity/
    // CalculateSize were removed: none was ever defined or bound, so each was a
    // link-time trap for anyone who called it. Their intended behavior is now
    // provided by Reshape (zero-copy), the copy constructor, and ToNumpy.
};

std::vector<int64_t> GetBroadcastShape(const NPUArray& a, const NPUArray& b);
