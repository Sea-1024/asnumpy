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

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <optional>
#include <utility>

namespace asnumpy {
NPUArray Prod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype = std::nullopt);
py::object Prod(const NPUArray& a, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Sum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype = std::nullopt);
py::object Sum(const NPUArray& a, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Nanprod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype = std::nullopt);
py::object Nanprod(const NPUArray& a, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Nansum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype = std::nullopt);
py::object Nansum(const NPUArray& a, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Cumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Cumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Nancumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Nancumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype = std::nullopt);

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axis);
} // namespace asnumpy
