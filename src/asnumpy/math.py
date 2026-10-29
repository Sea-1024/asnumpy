# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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


import numpy as np

from ._core.math import (
    absolute as _absolute,
)
from ._core.math import (
    add as _add,
)
from ._core.math import (
    amax as _amax,
)
from ._core.math import (
    amin as _amin,
)
from ._core.math import (
    arccos as _arccos,
)
from ._core.math import (
    arccosh as _arccosh,
)
from ._core.math import (
    arcsin as _arcsin,
)
from ._core.math import (
    arcsinh as _arcsinh,
)
from ._core.math import (
    arctan as _arctan,
)
from ._core.math import (
    arctan2 as _arctan2,
)
from ._core.math import (
    arctanh as _arctanh,
)
from ._core.math import (
    around as _around,
)
from ._core.math import (
    ceil as _ceil,
)
from ._core.math import (
    clip as _clip,
)
from ._core.math import (
    copysign as _copysign,
)
from ._core.math import (
    cos as _cos,
)
from ._core.math import (
    cosh as _cosh,
)
from ._core.math import (
    cross as _cross,
)
from ._core.math import (
    cumprod as _cumprod,
)
from ._core.math import (
    cumsum as _cumsum,
)
from ._core.math import (
    degrees as _degrees,
)
from ._core.math import (
    divide as _divide,
)
from ._core.math import (
    divmod as _divmod,
)
from ._core.math import (
    exp as _exp,
)
from ._core.math import (
    exp2 as _exp2,
)
from ._core.math import (
    expm1 as _expm1,
)
from ._core.math import (
    fabs as _fabs,
)
from ._core.math import (
    fix as _fix,
)
from ._core.math import (
    float_power as _float_power,
)
from ._core.math import (
    floor as _floor,
)
from ._core.math import (
    floor_divide as _floor_divide,
)
from ._core.math import (
    fmax as _fmax,
)
from ._core.math import (
    fmin as _fmin,
)
from ._core.math import (
    fmod as _fmod,
)
from ._core.math import (
    gcd as _gcd,
)
from ._core.math import (
    gelu as _gelu,
)
from ._core.math import (
    heaviside as _heaviside,
)
from ._core.math import (
    hypot as _hypot,
)
from ._core.math import (
    lcm as _lcm,
)
from ._core.math import (
    ldexp as _ldexp,
)
from ._core.math import (
    log as _log,
)
from ._core.math import (
    log1p as _log1p,
)
from ._core.math import (
    log2 as _log2,
)
from ._core.math import (
    log10 as _log10,
)
from ._core.math import (
    logaddexp as _logaddexp,
)
from ._core.math import (
    logaddexp2 as _logaddexp2,
)
from ._core.math import (
    max as _max,
)
from ._core.math import (
    maximum as _maximum,
)
from ._core.math import (
    min as _min,
)
from ._core.math import (
    minimum as _minimum,
)
from ._core.math import (
    mod as _mod,
)
from ._core.math import (
    modf as _modf,
)
from ._core.math import (
    multiply as _multiply,
)
from ._core.math import (
    nan_to_num as _nan_to_num,
)
from ._core.math import (
    nancumprod as _nancumprod,
)
from ._core.math import (
    nancumsum as _nancumsum,
)
from ._core.math import (
    nanmax as _nanmax,
)
from ._core.math import (
    nanmin as _nanmin,
)
from ._core.math import (
    nanprod as _nanprod,
)
from ._core.math import (
    nansum as _nansum,
)
from ._core.math import (
    negative as _negative,
)
from ._core.math import (
    positive as _positive,
)
from ._core.math import (
    power as _power,
)
from ._core.math import (
    prod as _prod,
)
from ._core.math import (
    rad2deg as _rad2deg,
)
from ._core.math import (
    radians as _radians,
)
from ._core.math import (
    real as _real,
)
from ._core.math import (
    reciprocal as _reciprocal,
)
from ._core.math import (
    relu as _relu,
)
from ._core.math import (
    remainder as _remainder,
)
from ._core.math import (
    rint as _rint,
)
from ._core.math import (
    round_ as _round_,
)
from ._core.math import (
    sign as _sign,
)
from ._core.math import (
    signbit as _signbit,
)
from ._core.math import (
    sin as _sin,
)
from ._core.math import (
    sinc as _sinc,
)
from ._core.math import (
    sinh as _sinh,
)
from ._core.math import (
    sqrt as _sqrt,
)
from ._core.math import (
    square as _square,
)
from ._core.math import (
    subtract as _subtract,
)
from ._core.math import (
    sum as _sum,
)
from ._core.math import (
    tan as _tan,
)
from ._core.math import (
    tanh as _tanh,
)
from ._core.math import (
    true_divide as _true_divide,
)
from ._core.math import (
    trunc as _trunc,
)
from ._types import ArrayLike, AxisOptional, DTypeLike
from ._ufunc import create_ufunc as _create_ufunc
from .utils import _convert_dtype, ndarray


# Trigonometric functions (ufunc-registered)
def _sin_fallback(x, dtype=None):
    """Fallback for sin: delegate to _core for non-registered dtypes (e.g. int)."""
    if dtype is not None:
        raise TypeError("sin() does not support the 'dtype' parameter")
    return ndarray(_sin(x))


sin = _create_ufunc(
    "sin",
    (("f->f", _sin, False), ("e->e", _sin, False), ("d->d", _sin, False)),
    doc="Compute sine element-wise.",
    fallback=_sin_fallback,
)


def cos(x: ArrayLike) -> ndarray:
    return ndarray(_cos(x))


def tan(x: ArrayLike) -> ndarray:
    return ndarray(_tan(x))


def arcsin(x: ArrayLike) -> ndarray:
    return ndarray(_arcsin(x))


def arccos(x: ArrayLike) -> ndarray:
    return ndarray(_arccos(x))


def arctan(x: ArrayLike) -> ndarray:
    return ndarray(_arctan(x))


def arctan2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_arctan2(x1, x2))


def hypot(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_hypot(x1, x2))


def radians(x: ArrayLike) -> ndarray:
    return ndarray(_radians(x))


def deg2rad(x: ArrayLike) -> ndarray:
    return ndarray(_radians(x))


def degrees(x: ArrayLike) -> ndarray:
    return ndarray(_degrees(x))


def rad2deg(x: ArrayLike) -> ndarray:
    return ndarray(_rad2deg(x))


# Miscellaneous functions
def absolute(x: ArrayLike) -> ndarray:
    return ndarray(_absolute(x))


def fabs(x: ArrayLike) -> ndarray:
    return ndarray(_fabs(x))


def sign(x: ArrayLike) -> ndarray:
    return ndarray(_sign(x))


def heaviside(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_heaviside(x1, x2))


def clip(a: ArrayLike, a_min: ArrayLike | float, a_max: ArrayLike | float) -> ndarray:
    return ndarray(_clip(a, a_min, a_max))


def nan_to_num(
    x: ArrayLike,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> ndarray:
    return ndarray(_nan_to_num(x, nan, posinf, neginf))


def sqrt(x: ArrayLike) -> ndarray:
    return ndarray(_sqrt(x))


def square(x: ArrayLike) -> ndarray:
    return ndarray(_square(x))


def relu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_relu(x, _convert_dtype(dtype)))


def gelu(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_gelu(x, _convert_dtype(dtype)))


# Arithmetic operations (ufunc-registered)
def _add_fallback(x1, x2, dtype=None):
    """Fallback for add with dtypes not in loop table (e.g. int32)."""
    return ndarray(_add(x1, x2, _convert_dtype(dtype)))


add = _create_ufunc(
    "add",
    (("ff->f", _add), ("dd->d", _add)),
    fallback=_add_fallback,
    doc="Add arguments element-wise.",
)


def reciprocal(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_reciprocal(x, _convert_dtype(dtype)))


def positive(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_positive(x, _convert_dtype(dtype)))


def _negative_fallback(x, dtype=None):
    """Fallback for negative with dtypes not in loop table (e.g. int64, float16)."""
    return ndarray(_negative(x, _convert_dtype(dtype)))


negative = _create_ufunc(
    "negative",
    (("f->f", _negative), ("d->d", _negative), ("i->i", _negative)),
    doc="Numerical negative, element-wise.",
    fallback=_negative_fallback,
)


def multiply(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_multiply(x1, x2, _convert_dtype(dtype)))


def divide(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_divide(x1, x2, _convert_dtype(dtype)))


def true_divide(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_true_divide(x1, x2, _convert_dtype(dtype)))


def subtract(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_subtract(x1, x2, _convert_dtype(dtype)))


def floor_divide(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_floor_divide(x1, x2, _convert_dtype(dtype)))


def float_power(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_float_power(x1, x2, _convert_dtype(dtype)))


def fmod(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_fmod(x1, x2, _convert_dtype(dtype)))


def mod(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_mod(x1, x2, _convert_dtype(dtype)))


def modf(x: ArrayLike) -> tuple:
    frac, inte = _modf(x)
    return (ndarray(frac), ndarray(inte))


def remainder(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_remainder(x1, x2, _convert_dtype(dtype)))


def divmod(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> tuple:
    res1, res2 = _divmod(x1, x2)
    _type = _convert_dtype(dtype)
    return (ndarray(res1, _type), ndarray(res2, _type))


def power(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_power(x1, x2, _convert_dtype(dtype)))


# Sums, products, differences
def prod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    if axis is None:
        return _prod(a)  # type: ignore[no-any-return]
    return ndarray(_prod(a, axis, keepdims, _convert_dtype(dtype)))


def sum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    if axis is None:
        return _sum(a)  # type: ignore[no-any-return]
    return ndarray(_sum(a, axis, keepdims, _convert_dtype(dtype)))


def nanprod(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    if axis is None:
        return _nanprod(a)  # type: ignore[no-any-return]
    return ndarray(_nanprod(a, axis, keepdims, _convert_dtype(dtype)))


def nansum(
    a: ArrayLike,
    axis: AxisOptional = None,
    keepdims: bool = False,
    dtype: DTypeLike = None,
) -> ndarray | float:
    if axis is None:
        return _nansum(a)  # type: ignore[no-any-return]
    return ndarray(_nansum(a, axis, keepdims, _convert_dtype(dtype)))


def cumprod(a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_cumprod(a, axis, _convert_dtype(dtype)))


def cumsum(a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_cumsum(a, axis, _convert_dtype(dtype)))


def nancumprod(a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_nancumprod(a, axis, _convert_dtype(dtype)))


def nancumsum(a: ArrayLike, axis: AxisOptional = None, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_nancumsum(a, axis, _convert_dtype(dtype)))


def cross(a: ArrayLike, b: ArrayLike, axis: AxisOptional = None) -> ndarray:
    return ndarray(_cross(a, b, axis))


# Exponents and logarithms
def exp(x: ArrayLike) -> ndarray:
    return ndarray(_exp(x))


def expm1(x: ArrayLike) -> ndarray:
    return ndarray(_expm1(x))


def exp2(x: ArrayLike) -> ndarray:
    return ndarray(_exp2(x))


def log(x: ArrayLike) -> ndarray:
    return ndarray(_log(x))


def log10(x: ArrayLike) -> ndarray:
    return ndarray(_log10(x))


def log2(x: ArrayLike) -> ndarray:
    return ndarray(_log2(x))


def log1p(x: ArrayLike) -> ndarray:
    return ndarray(_log1p(x))


def logaddexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logaddexp(x1, x2))


def logaddexp2(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_logaddexp2(x1, x2))


# Handling complex numbers
def real(x: ArrayLike) -> ndarray:
    return ndarray(_real(x))


# Floating point routines
def signbit(x: ArrayLike) -> ndarray:
    result = ndarray(_signbit(x))
    # CANN's aclnnSignbit does not handle IEEE 754 negative zero (-0.0).
    # Detect -0.0 via numpy and patch the result.
    import numpy as np

    np_x = x.to_numpy()  # type: ignore[union-attr]
    neg_zero_mask = np.signbit(np_x) & (np_x == 0)
    if neg_zero_mask.any():
        np_result = result.to_numpy().astype(np.bool_)
        np_result |= neg_zero_mask
        return ndarray(ndarray.from_numpy(np_result))
    return result


def ldexp(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_ldexp(x1, x2))


def copysign(x1: ArrayLike, x2: ArrayLike) -> ndarray:
    return ndarray(_copysign(x1, x2))


# Hyperbolic functions
def sinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_sinh(x, _convert_dtype(dtype)))


def cosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_cosh(x, _convert_dtype(dtype)))


def tanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_tanh(x, _convert_dtype(dtype)))


def arcsinh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arcsinh(x, _convert_dtype(dtype)))


def arccosh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arccosh(x, _convert_dtype(dtype)))


def arctanh(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_arctanh(x, _convert_dtype(dtype)))


# Other special functions
def sinc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_sinc(x, _convert_dtype(dtype)))


# Rational routines
def gcd(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_gcd(x1, x2, _convert_dtype(dtype)))


def lcm(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_lcm(x1, x2, _convert_dtype(dtype)))


# Rounding
def around(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_around(x, decimals, _convert_dtype(dtype)))


def round_(x: ArrayLike, decimals: int = 0, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_round_(x, decimals, _convert_dtype(dtype)))


def rint(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_rint(x, _convert_dtype(dtype)))


def fix(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_fix(x, _convert_dtype(dtype)))


def floor(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    converted_dtype = _convert_dtype(dtype)
    if converted_dtype is None:
        host = x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)
        if np.issubdtype(host.dtype, np.integer) or np.issubdtype(host.dtype, np.bool_):
            return ndarray.from_numpy(np.asarray(np.floor(host)))
    return ndarray(_floor(x, converted_dtype))


def ceil(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_ceil(x, _convert_dtype(dtype)))


def trunc(x: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_trunc(x, _convert_dtype(dtype)))


# Extrema finding
def maximum(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_maximum(x1, x2, _convert_dtype(dtype)))


def minimum(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_minimum(x1, x2, _convert_dtype(dtype)))


def fmax(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_fmax(x1, x2, _convert_dtype(dtype)))


def fmin(x1: ArrayLike, x2: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    return ndarray(_fmin(x1, x2, _convert_dtype(dtype)))


def max(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _max(a)  # type: ignore[no-any-return]
    return ndarray(_max(a, axis, keepdims))


def amax(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _amax(a)  # type: ignore[no-any-return]
    return ndarray(_amax(a, axis, keepdims))


def nanmax(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _nanmax(a)  # type: ignore[no-any-return]
    return ndarray(_nanmax(a, axis, keepdims))


def nanmin(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _nanmin(a)  # type: ignore[no-any-return]
    return ndarray(_nanmin(a, axis, keepdims))


def min(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _min(a)  # type: ignore[no-any-return]
    return ndarray(_min(a, axis, keepdims))


def amin(a: ArrayLike, axis: AxisOptional = None, keepdims: bool = False) -> ndarray | float:
    if axis is None:
        return _amin(a)  # type: ignore[no-any-return]
    return ndarray(_amin(a, axis, keepdims))
