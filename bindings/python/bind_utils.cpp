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

#include <asnumpy/utils/cast.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <utility>

void bind_utils(pybind11::module_& utils) {
    pybind11::class_<NPUArray>(utils, "ndarray")
        .def(py::init<const std::vector<int64_t>&, py::dtype>(), py::arg("shape"), py::arg("dtype"),
             "Constructs an empty NPUArray with the given shape and dtype.")
        // One-argument construction must remain a deep copy.
        .def(py::init<const NPUArray&>(), "Copy constructor for NPUArray")
        // Keep _move required and keyword-only: ndarray(other) selects the copy constructor, while
        // ndarray(other, _move=True) selects this consuming constructor.
        .def(
            py::init([](NPUArray& other, bool move) {
                if (!move) {
                    // The project's exception translator maps std::invalid_argument to
                    // Python ValueError; py::value_error would be caught as std::runtime_error.
                    throw std::invalid_argument("_move must be true for the internal consuming constructor");
                }
                return NPUArray(std::move(other));
            }),
            py::arg("other"), py::kw_only(), py::arg("_move"),
            "Internal consuming constructor; the source must not be used afterwards")
        .def("to_numpy", &NPUArray::ToNumpy)
        .def_static("from_numpy", &NPUArray::FromNumpy, py::arg("host_data"))
        .def(
            "astype",
            // Takes py::object, not py::dtype: py::dtype's caster is a strict isinstance check with
            // no conversion, so it would reject the scalar-type spellings NumPy users write
            // (np.float32) and accept only np.dtype("float32"). Normalizing here keeps direct
            // _core.ndarray users -- e.g. whatever ap.load() returns -- working.
            [](const NPUArray& self, const py::object& dtype) {
                return asnumpy::CastTo(self, NPUArray::GetACLDataType(py::dtype::from_args(dtype)));
            },
            py::arg("dtype"), "Cast the array to the given dtype on device, returning a new array.")
        .def_property_readonly("shape",
                               [](const NPUArray& self) {
                                   py::tuple shape_tuple(self.shape.size());
                                   for (size_t i = 0; i < self.shape.size(); ++i) {
                                       shape_tuple[i] = self.shape[i];
                                   }
                                   return shape_tuple;
                               })
        .def_property_readonly("dtype", [](const NPUArray& self) { return self.dtype; })
        .def_property_readonly("aclDtype", [](const NPUArray& self) { return static_cast<int>(self.aclDtype); })
        .def_property_readonly("ndim", [](const NPUArray& self) { return self.shape.size(); })
        .def_property_readonly("itemsize",
                               [](const NPUArray& self) { return NPUArray::GetDataTypeSize(self.aclDtype); })
        .def_property_readonly(
            "nbytes", [](const NPUArray& self) { return self.tensorSize * NPUArray::GetDataTypeSize(self.aclDtype); })
        .def_property_readonly("strides", [](const NPUArray& self) {
            auto itemsize = NPUArray::GetDataTypeSize(self.aclDtype);
            py::tuple byte_strides(self.strides.size());
            for (size_t i = 0; i < self.strides.size(); ++i) {
                byte_strides[i] = self.strides[i] * itemsize;
            }
            return byte_strides;
        })
        .def("reshape", &NPUArray::Reshape, py::arg("new_shape"), "Reshape the array to a new shape")
        .def("flatten", &NPUArray::Flatten, "Flatten the array to 1D")
        .def("ravel", &NPUArray::Ravel, "Ravel the array (alias for flatten)")
        .def(
            "__getitem__",
            [](const NPUArray& self, const py::object& key) {
                // a[i] arrives as a bare int; a[i, j] arrives as a tuple. Only
                // integer indexing is supported here -- slices and index arrays
                // are rejected rather than silently mis-indexed.
                std::vector<int64_t> indices;
                auto push = [&indices](const py::handle& item) {
                    if (py::isinstance<py::bool_>(item) || !py::isinstance<py::int_>(item)) {
                        throw py::type_error("only integer indices are supported");
                    }
                    indices.push_back(item.cast<int64_t>());
                };
                if (py::isinstance<py::tuple>(key)) {
                    for (const auto& item : key.cast<py::tuple>()) {
                        push(item);
                    }
                } else {
                    push(key);
                }
                return self.GetItem(indices);
            },
            py::arg("key"), "Index the array with integer indices over the leading axes")
        .def("transpose", &NPUArray::Transpose, py::arg("axes") = std::vector<int64_t>{},
             "Permute the array's axes; empty axes reverses them")
        .def("swapaxes", &NPUArray::SwapAxes, py::arg("axis1"), py::arg("axis2"), "Interchange two axes")
        .def_property_readonly(
            "T", [](const NPUArray& self) { return self.Transpose({}); },
            "The transposed array (all axes reversed)");
    utils.def("broadcast_shape", &GetBroadcastShape, py::arg("a"), py::arg("b"));
}
