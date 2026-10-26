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

#include <asnumpy/math/arithmetic_operations.hpp>
#include <asnumpy/math/exponents_and_logarithms.hpp>
#include <asnumpy/math/extrema_finding.hpp>
#include <asnumpy/math/floating_point_routines.hpp>
#include <asnumpy/math/handling_complex_numbers.hpp>
#include <asnumpy/math/hyperbolic_functions.hpp>
#include <asnumpy/math/miscellaneous.hpp>
#include <asnumpy/math/other_special_functions.hpp>
#include <asnumpy/math/rational_routines.hpp>
#include <asnumpy/math/rounding.hpp>
#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/math/trigonometric_functions.hpp>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace asnumpy;

namespace asnumpy {

void bind_trigonometric_functions(py::module_& math);
void bind_hyperbolic_functions(py::module_& math);
void bind_rounding(py::module_& math);
void bind_sums_products_differences(py::module_& math);
void bind_exponents_and_logarithms(py::module_& math);
void bind_other_special_functions(py::module_& math);
void bind_floating_point_routines(py::module_& math);
void bind_rational_routines(py::module_& math);
void bind_arithmetic_operations(py::module_& math);
void bind_handling_complex_numbers(py::module_& math);
void bind_miscellaneous(py::module_& math);
void bind_extrema_finding(py::module_& math);

} // namespace asnumpy

void bind_math(py::module_& math) {
    math.doc() = "math module of asnumpy";
    bind_trigonometric_functions(math);
    bind_hyperbolic_functions(math);
    bind_rounding(math);
    bind_sums_products_differences(math);
    bind_exponents_and_logarithms(math);
    bind_other_special_functions(math);
    bind_floating_point_routines(math);
    bind_rational_routines(math);
    bind_arithmetic_operations(math);
    bind_handling_complex_numbers(math);
    bind_miscellaneous(math);
    bind_extrema_finding(math);
}

namespace asnumpy {
void bind_trigonometric_functions(py::module_& math) {
    math.def("sin", &Sin, py::arg("x"));
    math.def("cos", &Cos, py::arg("x"));
    math.def("tan", &Tan, py::arg("x"));
    math.def("arcsin", &Arcsin, py::arg("x"));
    math.def("arccos", &Arccos, py::arg("x"));
    math.def("arctan", &Arctan, py::arg("x"));
    math.def("arctan2", &Arctan2, py::arg("x1"), py::arg("x2"));
    math.def("hypot", &Hypot, py::arg("x1"), py::arg("x2"));
    math.def("radians", &Radians, py::arg("x"));
    math.def("deg2rad", &Radians, py::arg("x"));
    math.def("degrees", &Degrees, py::arg("x"));
    math.def("rad2deg", &Degrees, py::arg("x"));
}

void bind_miscellaneous(py::module_& math) {
    math.def("absolute", &Absolute, py::arg("x"));
    math.def("fabs", &Fabs, py::arg("x"));
    math.def("sign", &Sign, py::arg("x"));
    math.def("heaviside", &Heaviside, py::arg("x1"), py::arg("x2"));
    math.def("clip", py::overload_cast<const NPUArray&, const NPUArray&, const NPUArray&>(&Clip), py::arg("a"),
             py::arg("a_min"), py::arg("a_max"));
    math.def("clip", py::overload_cast<const NPUArray&, float, float>(&Clip), py::arg("a"), py::arg("a_min"),
             py::arg("a_max"));
    math.def("clip", py::overload_cast<const NPUArray&, float, const NPUArray&>(&Clip), py::arg("a"), py::arg("a_min"),
             py::arg("a_max"));
    math.def("clip", py::overload_cast<const NPUArray&, const NPUArray&, float>(&Clip), py::arg("a"), py::arg("a_min"),
             py::arg("a_max"));
    math.def("nan_to_num", &Nan_to_num, py::arg("x"), py::arg("nan"), py::arg("posinf"), py::arg("neginf"));
    math.def("sqrt", &Sqrt, py::arg("x"));
    math.def("square", &Square, py::arg("x"));
    math.def("relu", &Relu, py::arg("x"), py::arg("dtype") = py::none());
    math.def("gelu", &Gelu, py::arg("x"), py::arg("dtype") = py::none());
}

void bind_arithmetic_operations(py::module_& math) {
    math.def("add", &Add, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("reciprocal", &Reciprocal, py::arg("x"), py::arg("dtype") = py::none());
    math.def("positive", &Positive, py::arg("x"), py::arg("dtype") = py::none());
    math.def("negative", &Negative, py::arg("x"), py::arg("dtype") = py::none());
    math.def("multiply", &Multiply, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("divide", &Divide, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("true_divide", &TrueDivide, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("subtract", &Subtract, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("floor_divide", &FloorDivide, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("float_power", &FloatPower, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("fmod", &Fmod, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("mod", &Mod, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("modf", &Modf, py::arg("x"));
    math.def("remainder", &Remainder, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("divmod", &Divmod, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("power", py::overload_cast<const NPUArray&, const NPUArray&, std::optional<py::dtype>>(&Power),
             py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("power", py::overload_cast<const py::object&, const NPUArray&, std::optional<py::dtype>>(&Power),
             py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("power", py::overload_cast<const NPUArray&, const py::object&, std::optional<py::dtype>>(&Power),
             py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
}

void bind_sums_products_differences(py::module_& math) {
    math.def("prod", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Prod), py::arg("a"),
             py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    math.def("prod", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Prod), py::arg("a"),
             py::arg("dtype") = py::none());
    math.def("sum", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Sum), py::arg("a"),
             py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    math.def("sum", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Sum), py::arg("a"),
             py::arg("dtype") = py::none());
    math.def("nanprod", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Nanprod),
             py::arg("a"), py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    math.def("nanprod", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Nanprod), py::arg("a"),
             py::arg("dtype") = py::none());
    math.def("nansum", py::overload_cast<const NPUArray&, int64_t, bool, std::optional<py::dtype>>(&Nansum),
             py::arg("a"), py::arg("axis"), py::arg("keepdims"), py::arg("dtype") = py::none());
    math.def("nansum", py::overload_cast<const NPUArray&, std::optional<py::dtype>>(&Nansum), py::arg("a"),
             py::arg("dtype") = py::none());
    math.def("cumprod", &Cumprod, py::arg("a"), py::arg("axis"), py::arg("dtype") = py::none());
    math.def("cumsum", &Cumsum, py::arg("a"), py::arg("axis"), py::arg("dtype") = py::none());
    math.def("nancumprod", &Nancumprod, py::arg("a"), py::arg("axis"), py::arg("dtype") = py::none());
    math.def("nancumsum", &Nancumsum, py::arg("a"), py::arg("axis"), py::arg("dtype") = py::none());
    math.def("cross", &Cross, py::arg("a"), py::arg("b"), py::arg("axis"));
}

void bind_exponents_and_logarithms(py::module_& math) {
    math.def("exp", &Exp, py::arg("x"));
    math.def("expm1", &Expm1, py::arg("x"));
    math.def("exp2", &Exp2, py::arg("x"));
    math.def("log", &Log, py::arg("x"));
    math.def("log10", &Log10, py::arg("x"));
    math.def("log2", &Log2, py::arg("x"));
    math.def("log1p", &Log1p, py::arg("x"));
    math.def("logaddexp", &Logaddexp, py::arg("x1"), py::arg("x2"));
    math.def("logaddexp2", &Logaddexp2, py::arg("x1"), py::arg("x2"));
}

void bind_handling_complex_numbers(py::module_& math) { math.def("real", &Real, py::arg("x")); }

void bind_floating_point_routines(py::module_& math) {
    math.def("signbit", &Signbit, py::arg("x"));
    math.def("ldexp", &Ldexp, py::arg("x1"), py::arg("x2"));
    math.def("copysign", &Copysign, py::arg("x1"), py::arg("x2"));
}

void bind_hyperbolic_functions(py::module_& math) {
    math.def("sinh", &Sinh, py::arg("x"), py::arg("dtype") = py::none());
    math.def("cosh", &Cosh, py::arg("x"), py::arg("dtype") = py::none());
    math.def("tanh", &Tanh, py::arg("x"), py::arg("dtype") = py::none());
    math.def("arcsinh", &Arcsinh, py::arg("x"), py::arg("dtype") = py::none());
    math.def("arccosh", &Arccosh, py::arg("x"), py::arg("dtype") = py::none());
    math.def("arctanh", &Arctanh, py::arg("x"), py::arg("dtype") = py::none());
}

void bind_other_special_functions(py::module_& math) {
    math.def("sinc", &Sinc, py::arg("x"), py::arg("dtype") = py::none());
}

void bind_rational_routines(py::module_& math) {
    math.def("gcd", &Gcd, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("lcm", &Lcm, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
}

void bind_rounding(py::module_& math) {
    math.def("around", &Around, py::arg("x"), py::arg("decimals"), py::arg("dtype") = py::none());
    math.def("round_", &Round_, py::arg("x"), py::arg("decimals"), py::arg("dtype") = py::none());
    math.def("rint", &Rint, py::arg("x"), py::arg("dtype") = py::none());
    math.def("fix", &Fix, py::arg("x"), py::arg("dtype") = py::none());
    math.def("floor", &Floor, py::arg("x"), py::arg("dtype") = py::none());
    math.def("ceil", &Ceil, py::arg("x"), py::arg("dtype") = py::none());
    math.def("trunc", &Trunc, py::arg("x"), py::arg("dtype") = py::none());
}

void bind_extrema_finding(py::module_& math) {
    math.def("maximum", &Maximum, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("minimum", &Minimum, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("fmax", &Fmax, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("fmin", &Fmin, py::arg("x1"), py::arg("x2"), py::arg("dtype") = py::none());
    math.def("max", py::overload_cast<const NPUArray&, int64_t, bool>(&Max), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("max", py::overload_cast<const NPUArray&>(&Max), py::arg("a"));
    math.def("amax", py::overload_cast<const NPUArray&, int64_t, bool>(&Max), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("amax", py::overload_cast<const NPUArray&>(&Max), py::arg("a"));
    math.def("nanmax", py::overload_cast<const NPUArray&, int64_t, bool>(&Nanmax), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("nanmax", py::overload_cast<const NPUArray&>(&Nanmax), py::arg("a"));
    math.def("nanmin", py::overload_cast<const NPUArray&, int64_t, bool>(&Nanmin), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("nanmin", py::overload_cast<const NPUArray&>(&Nanmin), py::arg("a"));
    math.def("min", py::overload_cast<const NPUArray&, int64_t, bool>(&Min), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("min", py::overload_cast<const NPUArray&>(&Min), py::arg("a"));
    math.def("amin", py::overload_cast<const NPUArray&, int64_t, bool>(&Min), py::arg("a"), py::arg("axis"),
             py::arg("keepdims"));
    math.def("amin", py::overload_cast<const NPUArray&>(&Min), py::arg("a"));
}

} // namespace asnumpy
