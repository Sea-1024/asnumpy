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

add_library(utils OBJECT npu_array.cpp npu_scalar.cpp status_handler.cpp acl_resource.cpp cast.cpp dtype_promotion.cpp device_memory_pool.cpp)

target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(utils PUBLIC SPDLOG_FMT_EXTERNAL)
target_link_libraries(utils PUBLIC fmt::fmt spdlog::spdlog ascend_sdk pybind11::pybind11)
