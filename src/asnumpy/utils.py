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

import operator
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from loguru import logger

from ._core import broadcast_shape as _broadcast_shape
from ._core import ndarray as _ndarray


class ndarray(_ndarray):
    @overload
    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None: ...

    @overload
    def __init__(self, other: _ndarray) -> None: ...

    def __init__(self, shape_or_array, dtype=None):
        if isinstance(shape_or_array, ndarray):
            # Public ndarray and its subclasses retain deep-copy semantics, so an
            # explicit ndarray(existing_array) never consumes the source array.
            super().__init__(shape_or_array)
        elif isinstance(shape_or_array, _ndarray):
            # A base _ndarray that is not a public ndarray is a private one-shot
            # result produced by _core; wrapping consumes it via move semantics.
            super().__init__(shape_or_array, _move=True)
        elif isinstance(shape_or_array, (Sequence, int)):
            if dtype is None:
                raise ValueError("dtype must be specified when initializing with shape")
            shape = shape_or_array if isinstance(shape_or_array, Sequence) else (shape_or_array,)
            super().__init__(shape, np.dtype(dtype))
        else:
            raise TypeError(f"Unsupported type for initialization: {type(shape_or_array)}")

    def __repr__(self) -> str:
        return f"ndarray(shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def shape(self) -> tuple:
        return super().shape  # type: ignore[no-any-return]

    @property
    def dtype(self) -> np.dtype:
        return super().dtype  # type: ignore[no-any-return]

    @property
    def acl_dtype(self) -> int:
        return super().aclDtype  # type: ignore[no-any-return]

    @property
    def ndim(self) -> int:
        return super().ndim  # type: ignore[no-any-return]

    @property
    def itemsize(self) -> int:
        return super().itemsize  # type: ignore[no-any-return]

    @property
    def nbytes(self) -> int:
        return super().nbytes  # type: ignore[no-any-return]

    @property
    def strides(self) -> tuple:
        return super().strides  # type: ignore[no-any-return]

    @classmethod
    def from_numpy(cls, host_data: np.ndarray) -> "ndarray":
        base_obj = _ndarray.from_numpy(host_data)
        return cls(base_obj)

    def to_numpy(self) -> np.ndarray:
        return super().to_numpy()  # type: ignore[no-any-return]

    __array_priority__ = 100.0

    def __array_ufunc__(self, ufunc_obj, method, *inputs, **kwargs):
        """Implement NumPy ufunc protocol.

        When a NumPy ufunc (e.g. np.sin, np.add) is called with an asnumpy
        ndarray, this method dispatches to the corresponding asnumpy ufunc.

        Args:
            ufunc_obj: The numpy.ufunc being applied.
            method: The ufunc method ('__call__', 'outer', 'reduce', etc.).
            *inputs: Input arrays.
            **kwargs: Additional keyword arguments (out, dtype, etc.).

        Returns:
            Result from the asnumpy ufunc, or NotImplemented if unhandled.
        """
        import asnumpy as anp

        if method not in ("__call__",):
            return NotImplemented

        name = getattr(ufunc_obj, "__name__", None)
        if name is None:
            return NotImplemented

        func = getattr(anp, name, None)
        if func is None or not hasattr(func, "_ops"):
            return NotImplemented

        out = kwargs.pop("out", None)
        dtype = kwargs.pop("dtype", None)

        # Unsupported numpy ufunc kwargs — return NotImplemented so NumPy
        # falls back to its own implementation rather than silently ignoring them.
        _UNSUPPORTED_UFUNC_KWARGS = frozenset(
            ("where", "casting", "subok", "order", "signature", "extobj"),
        )
        if _UNSUPPORTED_UFUNC_KWARGS & kwargs.keys():
            return NotImplemented

        try:
            return func(*inputs, dtype=dtype, out=out)
        except TypeError as exc:
            # Only return NotImplemented for expected dispatch failures
            # (e.g. no matching loop for given dtypes). Internal bugs
            # should propagate so they are not silently masked.
            if "no matching loop" in str(exc):
                return NotImplemented
            raise

    def __array_function__(self, func, types, args, kwargs):
        """Implement NumPy array function protocol.

        When a NumPy function (e.g. numpy.sin, numpy.add) is called with
        an asnumpy ndarray, this method dispatches to the corresponding
        asnumpy function.

        Args:
            func: The numpy function being called.
            types: Collection of types involved in the call.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Result from the asnumpy function, or NotImplemented if unhandled.
        """
        import asnumpy as anp

        name = func.__name__
        anp_func = getattr(anp, name, None)

        if anp_func is None or anp_func is func:
            return NotImplemented

        try:
            return anp_func(*args, **kwargs)
        except TypeError as exc:
            if "no matching loop" in str(exc):
                return NotImplemented
            raise

    def astype(
        self,
        dtype,
        order: str = "K",
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "unsafe",
        subok: bool = True,
        copy: bool = True,
    ) -> "ndarray":
        """Copy of the array, cast to the given dtype.

        The parameter order matches :meth:`numpy.ndarray.astype` positionally, so
        ``x.astype(dt, "K")`` binds ``order`` as a NumPy user expects.

        Args:
            dtype: Target dtype.
            order: Memory layout. Only ``"K"``/``"C"`` are meaningful: asnumpy arrays are always
                dense and C-contiguous, so ``"F"``/``"A"`` raise rather than silently lie.
            casting: Casting rule, checked against :func:`numpy.can_cast`. Defaults to
                ``"unsafe"``, matching NumPy.
            subok: Accepted for signature compatibility; asnumpy has no ndarray subclasses to
                preserve, so only the default is allowed.
            copy: If False, return self when the dtype already matches instead of copying.
        """
        dtype = np.dtype(dtype)
        if order not in ("K", "C"):
            raise ValueError(
                f"order={order!r} is not supported: asnumpy arrays are always C-contiguous"
            )
        if not subok:
            raise ValueError("subok=False is not supported")
        if not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                f"Cannot cast array data from {self.dtype!r} to {dtype!r} "
                f"according to the rule '{casting}'"
            )
        if dtype == self.dtype:
            # CastTo would deep-copy on a dtype match, and re-wrapping copies again -- two device
            # copies for what is at most one. Short-circuit to a single copy (or none).
            return ndarray(self) if copy else self
        return ndarray(super().astype(dtype))

    def reshape(self, new_shape) -> "ndarray":
        """Return a reshaped array."""
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        return ndarray(super().reshape(list(new_shape)))

    def flatten(self) -> "ndarray":
        """Return a flattened 1D array."""
        return ndarray(super().flatten())

    def ravel(self) -> "ndarray":
        """Return a flattened array (alias for flatten)."""
        return ndarray(super().ravel())

    def transpose(self, *axes) -> "ndarray":
        """Return a view of the array with axes permuted.

        Accepts NumPy's three spellings: ``a.transpose()`` reverses the axes,
        ``a.transpose(1, 0)`` takes them varargs-style, and
        ``a.transpose((1, 0))`` takes a single sequence.
        """
        if not axes:
            order: Sequence[int] = ()
        elif len(axes) == 1 and axes[0] is None:
            order = ()
        elif len(axes) == 1 and isinstance(axes[0], Sequence):
            order = axes[0]
        else:
            order = axes
        return ndarray(super().transpose([int(a) for a in order]))

    def swapaxes(self, axis1: int, axis2: int) -> "ndarray":
        """Return a copy of the array with two axes interchanged."""
        return ndarray(super().swapaxes(axis1, axis2))

    @property
    def T(self) -> "ndarray":
        """The array with all axes reversed."""
        return ndarray(super().T)

    def squeeze(self, axis=None) -> "ndarray":
        """Remove axes of length one.

        Imported lazily to avoid a circular import: asnumpy.array imports
        ndarray from this module.
        """
        from .array import squeeze as _squeeze_fn

        return _squeeze_fn(self, axis)

    def __getitem__(self, key) -> "ndarray":
        """Index the array with integer indices over the leading axes.

        Supports ``a[i]`` and ``a[i, j]`` with NumPy's negative-index wrapping.
        Slices and index arrays are not supported yet and raise TypeError.
        """
        # Validate here rather than in C++: the module's exception translator
        # maps every std::runtime_error (which pybind11's py::type_error derives
        # from) to RuntimeError, so a TypeError raised in the binding would
        # reach Python as RuntimeError.
        items = key if isinstance(key, tuple) else (key,)
        for item in items:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                raise TypeError(
                    f"only integer indices are supported, got {type(item).__name__}"
                )
        return ndarray(super().__getitem__(key))


@logger.catch(reraise=True)
def broadcast_shape(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple:
    logger.debug(f"Broadcasting shapes {shape_a}, {shape_b}")
    return _broadcast_shape(shape_a, shape_b)  # type: ignore[no-any-return]


@logger.catch(reraise=True)
def _convert_dtype(dtype):
    """Convert dtype parameter to appropriate format if needed"""
    logger.debug(f"Converting dtype {dtype}")
    if dtype is None:
        return None
    if not isinstance(dtype, np.dtype):
        return np.dtype(dtype)
    return dtype


@logger.catch(reraise=True)
def _convert_size(size: int | Sequence[int]) -> Sequence[int]:
    """Convert size from int to tuple"""
    logger.debug(f"Converting size {size}")
    if isinstance(size, int):
        return (size,)
    return size


@logger.catch(reraise=True)
def _normalize_shape(shape: int | Sequence[int]) -> list[int]:
    """Normalize a shape argument to a list and reject negative dimensions."""
    logger.debug(f"Normalizing shape {shape}")

    if isinstance(shape, (int, np.integer)):
        normalized = [operator.index(shape)]
    else:
        normalized = [operator.index(dim) for dim in shape]

    if any(dim < 0 for dim in normalized):
        raise ValueError("negative dimensions are not allowed")

    return normalized


def _ensure_ndarray(a) -> ndarray:
    """Coerce an ArrayLike argument to a public ndarray (host inputs are copied to device)."""
    if isinstance(a, ndarray):
        return a
    return ndarray.from_numpy(np.asarray(a))
