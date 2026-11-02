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
 ******************************************************************************/

#include <asnumpy/array/basic.hpp>
#include <asnumpy/array/manipulation.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace asnumpy;

void bind_array(pybind11::module_& array) {
    array.doc() = "array module of asnumpy";
    array.def("zeros", &Zeros, py::arg("shape"), py::arg("dtype"));
    array.def("zeros_like", &Zeros_like, py::arg("other"), py::arg("dtype"));
    array.def("full", &Full, py::arg("shape"), py::arg("value"), py::arg("dtype"));
    array.def("full_like", &Full_like, py::arg("other"), py::arg("value"), py::arg("dtype"));
    array.def("empty", &Empty, py::arg("shape"), py::arg("dtype"));
    array.def("empty_like", &EmptyLike, py::arg("prototype"), py::arg("dtype") = py::none());
    array.def("eye", &Eye, py::arg("n"), py::arg("dtype"));
    array.def("ones", &Ones, py::arg("shape"), py::arg("dtype"));
    array.def("ones_like", &ones_like, py::arg("other"), py::arg("dtype"));
    array.def("identity", &Identity, py::arg("n"), py::arg("dtype"));
    array.def("linspace", &Linspace, py::arg("start"), py::arg("end"), py::arg("steps") = 50,
              py::arg("dtype") = py::none());
    array.def("concatenate", &Concatenate, py::arg("arrays"), py::arg("axis") = 0);
    array.def("stack", &Stack, py::arg("arrays"), py::arg("axis") = 0);
    array.def("squeeze", &Squeeze, py::arg("array"), py::arg("axes") = std::vector<int64_t>{});
    array.def("expand_dims", &ExpandDims, py::arg("array"), py::arg("axis"));
}
