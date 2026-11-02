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
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

class NPUArray {
  public:
    aclTensor* tensorPtr;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    py::dtype dtype;
    aclDataType aclDtype;
    size_t tensorSize;

  private:
    void* devicePtr;

  public:
    /**
     * @brief Constructor to create an empty NPUArray from shape and data type
     * @param shape Tensor shape
     * @param dtype Data type
     */
    NPUArray(const std::vector<int64_t>& shape, py::dtype dtype);
    void* device_address() const { return devicePtr; }

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
     * @brief Reshape array to new shape
     * @param new_shape Target shape
     * @return NPUArray Reshaped array
     */
    NPUArray Reshape(const std::vector<int64_t>& new_shape) const;

    /**
     * @brief Flatten array to 1D
     * @return NPUArray Flattened array
     */
    NPUArray Flatten() const;

    /**
     * @brief Ravel array (alias for flatten)
     * @return NPUArray Flattened array
     */
    NPUArray Ravel() const;

    // -- Not implemented yet --
    /**
     * @brief Update contiguity flags of the NPUArray
     */
    void UpdateContiguity();

    /**
     * @brief Calculate the size of the tensor
     */
    void CalculateSize();
    /**
     * @brief Fill with specified value
     * @param value Value to fill
     */
    void Fill(const py::object& value);

    /**
     * @brief Temporarily retained interface, no effect
     * @return py::array Returned array
     */
    py::array ToArray();

    /**
     * @brief Return a Python list
     * @return py::list Returned Python list
     */
    py::list ToList() const;

    /**
     * @brief Create a deep copy
     * @return std::unique_ptr<NPUArray> Returned deep copy object
     */
    std::unique_ptr<NPUArray> Copy() const;
};

std::vector<int64_t> GetBroadcastShape(const NPUArray& a, const NPUArray& b);
