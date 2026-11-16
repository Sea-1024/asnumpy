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

"""Declarative ufunc registry for AsNumpy.

Provides a CuPy-style dtype loop table mechanism for dispatching to
pre-compiled CANN ACLNN operators, replacing scattered if-chain dispatch
with a declarative registration pattern.
"""

from __future__ import annotations

import numpy as np

# Dtype char code -> np.dtype mapping
_DTYPE_CHAR_MAP = {
    "?": np.bool_,
    "b": np.int8,
    "B": np.uint8,
    "h": np.int16,
    "H": np.uint16,
    "i": np.int32,
    "I": np.uint32,
    "l": np.int64,
    "L": np.uint64,
    "e": np.float16,
    "f": np.float32,
    "d": np.float64,
    "F": np.complex64,
    "D": np.complex128,
}


def _char_to_dtype(ch: str) -> np.dtype:
    """Convert a dtype char code to numpy dtype."""
    if ch in _DTYPE_CHAR_MAP:
        return np.dtype(_DTYPE_CHAR_MAP[ch])
    raise ValueError(f"Unknown dtype char code: {ch!r}")


def _dtype_kind(dtype: np.dtype) -> int:
    """Return kind score for weak-scalar coercion rules.

    Lower scores can be coerced to higher dtypes without promotion.
    bool=0, int=1, float/complex=2, other=3.
    """
    if dtype == np.dtype("bool"):
        return 0
    if np.issubdtype(dtype, np.integer):
        return 1
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return 2
    return 3


def _is_weak_scalar(value) -> bool:
    """Check if *value* is a Python native scalar subject to coercion.

    - ``bool`` is **never** weak (matches NumPy semantics).
    - NumPy scalars (``np.float32(1)``) are **never** weak.
    - Only plain ``int``, ``float``, ``complex`` are weak.
    """
    if isinstance(value, (bool, np.generic)):
        return False
    return isinstance(value, (int, float, complex))


def _get_max_array_dtype(args: list) -> np.dtype | None:
    """Return the highest-kind, highest-precision dtype among array-like arguments.

    Tie-breaking within the same kind (e.g. float16 vs float32) uses
    ``dtype.itemsize`` so that the most precise dtype wins.
    """
    array_dtypes: list[np.dtype] = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            array_dtypes.append(arg.dtype)
        elif hasattr(arg, "dtype") and not _is_weak_scalar(arg):
            array_dtypes.append(arg.dtype)
    if not array_dtypes:
        return None
    return max(array_dtypes, key=lambda d: (_dtype_kind(d), d.itemsize))


def _coerce_weak_scalar(value, target_dtype: np.dtype) -> np.ndarray:
    """Wrap a weak scalar as a 0-d NumPy array with *target_dtype*.

    If *value* is complex and *target_dtype* is real, fall back to
    ``complex128`` to avoid a casting error.
    """
    if isinstance(value, complex) and not np.issubdtype(target_dtype, np.complexfloating):
        target_dtype = np.dtype("complex128")
    return np.array(value, dtype=target_dtype)


def _parse_type_sig(sig: str) -> tuple[list[np.dtype], list[np.dtype]]:
    """Parse a type signature string like 'ff->f' into (in_dtypes, out_dtypes)."""
    parts = sig.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid type signature: {sig!r}, expected 'in->out' format")
    in_str, out_str = parts
    in_dtypes = [_char_to_dtype(ch) for ch in in_str]
    out_dtypes = [_char_to_dtype(ch) for ch in out_str]
    return in_dtypes, out_dtypes


class Op:
    """A single entry in the dtype loop table.

    Attributes:
        in_types: Tuple of input numpy dtypes this loop accepts.
        out_types: Tuple of output numpy dtypes this loop produces.
        routine: The callable to invoke (typically a _core function).
        accepts_dtype: Whether *routine* accepts a ``dtype`` keyword argument.
    """

    __slots__ = ("in_types", "out_types", "routine", "accepts_dtype")

    def __init__(
        self,
        in_types: list[np.dtype],
        out_types: list[np.dtype],
        routine,
        accepts_dtype: bool = True,
    ):
        self.in_types = tuple(in_types)
        self.out_types = tuple(out_types)
        self.routine = routine
        self.accepts_dtype = accepts_dtype


def _is_single_entry(loop_table) -> bool:
    """Check if *loop_table* is a single (sig, routine[, accepts_dtype]) entry."""
    return (
        isinstance(loop_table, tuple)
        and len(loop_table) in (2, 3)
        and isinstance(loop_table[0], str)
        and "->" in loop_table[0]
        and callable(loop_table[1])
    )


def _parse_loop_table(loop_table, nin: int, nout: int) -> list[Op]:
    """Parse a declarative loop table into a list of Op objects.

    Each entry can be:
    - A 2-tuple ``(type_sig_str, routine)`` — ``accepts_dtype`` defaults to True.
    - A 3-tuple ``(type_sig_str, routine, accepts_dtype)``.

    Args:
        loop_table: Either:
            - A single entry, e.g. ('ff->f', _add)
            - A tuple of entries, e.g. (('ff->f', _add), ('dd->d', _add))
        nin: Number of inputs (validated against type signature).
        nout: Number of outputs (validated against type signature).

    Returns:
        List of Op objects.

    Raises:
        ValueError: If any entry's input/output count does not match
            ``nin``/``nout``, or if the entry format is invalid.
    """
    # Detect single-entry format: ('ff->f', routine) vs multi-entry: (('ff->f', r1), ...)
    if _is_single_entry(loop_table):
        loop_table = (loop_table,)

    ops: list[Op] = []
    for entry in loop_table:
        if isinstance(entry, tuple) and len(entry) in (2, 3):
            sig = entry[0]
            routine = entry[1]
            accepts_dtype = entry[2] if len(entry) == 3 else True
            if isinstance(sig, str) and "->" in sig:
                in_dtypes, out_dtypes = _parse_type_sig(sig)
                if len(in_dtypes) != nin:
                    raise ValueError(
                        f"Type signature {sig!r} has {len(in_dtypes)} inputs, "
                        f"expected {nin} (from first entry)"
                    )
                if len(out_dtypes) != nout:
                    raise ValueError(
                        f"Type signature {sig!r} has {len(out_dtypes)} outputs, expected {nout}"
                    )
                ops.append(Op(in_dtypes, out_dtypes, routine, accepts_dtype=accepts_dtype))
            else:
                raise ValueError(f"Invalid loop table entry: {entry!r}")
        else:
            raise ValueError(f"Invalid loop table entry format: {entry!r}")
    return ops


class Ops:
    """Collection of Op entries with dtype-based dispatch.

    Uses exact equality (``actual == declared``) to match input dtypes
    against registered loop entries. First matching entry wins.
    Results are cached for performance.
    """

    def __init__(self, ops: list[Op], nin: int, nout: int):
        self.ops = tuple(ops)
        self.nin = nin
        self.nout = nout
        self._cache: dict[tuple, Op | None] = {}

    def guess_routine(
        self,
        in_dtypes: list[np.dtype],
        out_dtype: np.dtype | None = None,
    ) -> Op | None:
        """Find the best matching loop entry via exact equality, with caching.

        Args:
            in_dtypes: Actual input dtypes.
            out_dtype: If user explicitly specified an output dtype, find
                a loop whose out_types match it.

        Returns:
            Matching Op, or None if no loop matches.
        """
        key = (tuple(in_dtypes), out_dtype)
        if key in self._cache:
            return self._cache[key]

        result = self._search(in_dtypes, out_dtype)
        self._cache[key] = result
        return result

    def _search(
        self,
        in_dtypes: list[np.dtype],
        out_dtype: np.dtype | None,
    ) -> Op | None:
        """Linear search through ops list for a matching entry."""
        for op in self.ops:
            if out_dtype is not None:
                if len(op.out_types) != 1 or op.out_types[0] != out_dtype:
                    continue

            if len(in_dtypes) != len(op.in_types):
                continue

            if all(
                actual == declared for actual, declared in zip(in_dtypes, op.in_types, strict=False)
            ):
                return op

        return None


class ufunc:
    """AsNumpy ufunc object, modeled after numpy.ufunc.

    Attributes:
        name: The ufunc name.
        nin: Number of input arguments.
        nout: Number of output arguments.
        _ops: The Ops dispatch table.
    """

    def __init__(
        self,
        name: str,
        ops: Ops,
        doc: str = "",
        default_casting: str = "same_kind",
        fallback=None,
    ):
        self.name = name
        self.__name__ = name
        self.nin = ops.nin
        self.nout = ops.nout
        self._ops = ops
        self.__doc__ = doc
        self._default_casting = default_casting
        self._fallback = fallback

    def __repr__(self) -> str:
        return f"<ufunc '{self.name}'>"

    def __call__(self, *args, **kwargs):
        from .utils import ndarray as _ndarray

        dtype = kwargs.pop("dtype", None)
        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        out = kwargs.pop("out", None)

        if kwargs:
            raise TypeError(
                f"ufunc '{self.name}() got unexpected keyword argument(s): {', '.join(kwargs)}"
            )

        # Extract dtypes from inputs
        in_dtypes = []
        processed_args = list(args)
        for arg in processed_args:
            if isinstance(arg, _ndarray):
                in_dtypes.append(arg.dtype)
            elif isinstance(arg, np.ndarray):
                in_dtypes.append(arg.dtype)
            elif isinstance(arg, (bool, int, float, complex)):
                in_dtypes.append(np.dtype(type(arg)))
            else:
                in_dtypes.append(np.asarray(arg).dtype)

        # --- Weak-scalar coercion ---
        # Python native scalars (int, float, complex) must NOT trigger dtype
        # promotion.  Coerce them to match the highest-kind array dtype so
        # that loop-table exact-match succeeds.
        max_array_dtype = _get_max_array_dtype(processed_args)
        if max_array_dtype is not None:
            for i, arg in enumerate(processed_args):
                if _is_weak_scalar(arg):
                    scalar_dtype = np.dtype(type(arg))
                    if _dtype_kind(scalar_dtype) <= _dtype_kind(max_array_dtype):
                        coerced = _coerce_weak_scalar(arg, max_array_dtype)
                        processed_args[i] = coerced
                        in_dtypes[i] = coerced.dtype

        # Dispatch to matching loop
        op = self._ops.guess_routine(in_dtypes, dtype)

        if op is None:
            # Try fallback if available
            fallback = self._fallback
            if fallback is not None:
                result = fallback(*processed_args, dtype=dtype)
                return self._write_out(out, result)

            raise TypeError(
                f"ufunc '{self.name}' has no matching loop for input dtypes "
                f"{[str(d) for d in in_dtypes]}"
            )

        # Call the core routine
        if dtype is not None and not op.accepts_dtype:
            raise TypeError(f"ufunc '{self.name}' does not support the 'dtype' parameter")
        if op.accepts_dtype:
            result = op.routine(*processed_args, dtype)
        else:
            result = op.routine(*processed_args)

        # Wrap in ndarray if needed
        if isinstance(result, _ndarray):
            result_array = result
        elif isinstance(result, tuple):
            result_array = tuple(_ndarray(r) if not isinstance(r, _ndarray) else r for r in result)
        else:
            result_array = _ndarray(result)

        return self._write_out(out, result_array)

    @staticmethod
    def _write_out(out, result):
        """Write *result* into *out* buffer if specified, otherwise return result."""
        if out is None:
            return result
        out_targets = out if isinstance(out, tuple) else (out,)
        results = result if isinstance(result, tuple) else (result,)
        for dst, src in zip(out_targets, results, strict=True):
            dst[...] = src
        return out if not isinstance(out, tuple) else tuple(out)


def create_ufunc(
    name: str,
    loop_table: tuple,
    doc: str = "",
    default_casting: str = "same_kind",
    fallback=None,
) -> ufunc:
    """Create a ufunc from a declarative dtype loop table.

    Args:
        name: ufunc name (e.g. 'add').
        loop_table: Tuple of (type_sig, routine) entries.
            type_sig uses NumPy dtype char codes: 'ff->f' means
            (float32, float32) -> float32.
        doc: Docstring for the ufunc.
        default_casting: NumPy casting rule (default: 'same_kind').
        fallback: Optional callable invoked when no loop matches.

    Returns:
        A ufunc instance.

    Example:
        >>> add = create_ufunc('add', (('ff->f', _add), ('dd->d', _add)))
    """
    if not loop_table:
        raise ValueError(f"Loop table for '{name}' must not be empty")

    # Normalize single-entry format, consistent with _parse_loop_table
    if _is_single_entry(loop_table):
        loop_table = (loop_table,)

    # Infer nin from first entry's type signature
    first_sig = loop_table[0][0]

    in_str, _ = first_sig.split("->")
    nin = len(in_str)
    nout = 1  # all current ops produce 1 output

    ops_list = _parse_loop_table(loop_table, nin, nout)

    u = ufunc(name, Ops(ops_list, nin, nout), doc, default_casting, fallback=fallback)
    return u
