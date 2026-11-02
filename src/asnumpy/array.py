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

from loguru import logger

from ._core.array import (
    concatenate as _concatenate,
)
from ._core.array import (
    empty as _empty,
)
from ._core.array import (
    expand_dims as _expand_dims,
)
from ._core.array import (
    empty_like as _empty_like,
)
from ._core.array import (
    eye as _eye,
)
from ._core.array import (
    full as _full,
)
from ._core.array import (
    full_like as _full_like,
)
from ._core.array import (
    identity as _identity,
)
from ._core.array import (
    linspace as _linspace,
)
from ._core.array import (
    ones as _ones,
)
from ._core.array import (
    ones_like as _ones_like,
)
from ._core.array import (
    squeeze as _squeeze,
)
from ._core.array import (
    stack as _stack,
)
from ._core.array import (
    zeros as _zeros,
)
from ._core.array import (
    zeros_like as _zeros_like,
)
from ._types import ArrayLike, DTypeLike, ScalarLike, ShapeLike
from .utils import _convert_dtype, _normalize_shape, ndarray


@logger.catch(reraise=True)
def zeros(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros array shape={shape}, dtype={dtype}")
    return ndarray(_zeros(_normalize_shape(shape), _convert_dtype(dtype)))


@logger.catch(reraise=True)
def zeros_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros_like array other={other}, dtype={dtype}")
    return ndarray(_zeros_like(other, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def full(shape: ShapeLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full array shape={shape}, value={value}, dtype={dtype}")
    return ndarray(_full(_normalize_shape(shape), value, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def full_like(other: ArrayLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full_like array other={other}, value={value}, dtype={dtype}")
    return ndarray(_full_like(other, value, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def empty(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty array shape={shape}, dtype={dtype}")
    return ndarray(_empty(_normalize_shape(shape), _convert_dtype(dtype)))


@logger.catch(reraise=True)
def empty_like(prototype: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty_like array prototype={prototype}, dtype={dtype}")
    return ndarray(_empty_like(prototype, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def eye(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating eye array n={n}, dtype={dtype}")
    return ndarray(_eye(n, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def ones(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones array shape={shape}, dtype={dtype}")
    return ndarray(_ones(_normalize_shape(shape), _convert_dtype(dtype)))


@logger.catch(reraise=True)
def ones_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones_like array other={other}, dtype={dtype}")
    return ndarray(_ones_like(other, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def identity(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating identity array n={n}, dtype={dtype}")
    return ndarray(_identity(n, _convert_dtype(dtype)))


@logger.catch(reraise=True)
def linspace(
    start: ScalarLike,
    end: ScalarLike,
    steps: int = 50,
    dtype: DTypeLike = None,
) -> ndarray:
    logger.debug(f"Creating linspace array start={start}, end={end}, steps={steps}, dtype={dtype}")
    return ndarray(_linspace(start, end, steps, _convert_dtype(dtype)))


def _as_array_sequence(arrays, func_name: str) -> list:
    """Normalize the array sequence accepted by concatenate/stack.

    NumPy accepts any sequence here, and callers commonly pass a tuple. The
    binding takes a std::vector<NPUArray>, so the elements must already be
    device arrays -- host data is rejected rather than implicitly transferred.
    """
    if isinstance(arrays, ndarray):
        raise TypeError(f"{func_name}: expected a sequence of arrays, got a single ndarray")
    seq = list(arrays)
    if not seq:
        raise ValueError(f"{func_name}: need at least one array")
    for item in seq:
        if not isinstance(item, ndarray):
            raise TypeError(
                f"{func_name}: expected asnumpy ndarray elements, got {type(item).__name__}; "
                "transfer host data with ndarray.from_numpy() first"
            )
    return seq


@logger.catch(reraise=True)
def concatenate(arrays, axis: int = 0) -> ndarray:
    """Join a sequence of arrays along an existing axis."""
    logger.debug(f"Concatenating arrays axis={axis}")
    return ndarray(_concatenate(_as_array_sequence(arrays, "concatenate"), axis))


@logger.catch(reraise=True)
def stack(arrays, axis: int = 0) -> ndarray:
    """Join a sequence of arrays along a new axis."""
    logger.debug(f"Stacking arrays axis={axis}")
    return ndarray(_stack(_as_array_sequence(arrays, "stack"), axis))


@logger.catch(reraise=True)
def vstack(arrays) -> ndarray:
    """Stack arrays in sequence vertically (row wise).

    1-D inputs are promoted to 2-D rows first, matching numpy.vstack, so
    vstack of two (3,) arrays gives (2, 3) rather than (6,).
    """
    seq = _as_array_sequence(arrays, "vstack")
    promoted = [a.reshape((1, a.shape[0])) if a.ndim == 1 else a for a in seq]
    return ndarray(_concatenate(promoted, 0))


@logger.catch(reraise=True)
def hstack(arrays) -> ndarray:
    """Stack arrays in sequence horizontally (column wise).

    1-D inputs concatenate along axis 0, matching numpy.hstack.
    """
    seq = _as_array_sequence(arrays, "hstack")
    axis = 0 if seq[0].ndim == 1 else 1
    return ndarray(_concatenate(seq, axis))


@logger.catch(reraise=True)
def squeeze(a: ndarray, axis=None) -> ndarray:
    """Remove axes of length one from `a`."""
    logger.debug(f"Squeezing array axis={axis}")
    if axis is None:
        axes: list[int] = []
    elif isinstance(axis, int):
        axes = [axis]
    else:
        axes = [int(ax) for ax in axis]
    return ndarray(_squeeze(a, axes))


@logger.catch(reraise=True)
def expand_dims(a: ndarray, axis: int) -> ndarray:
    """Insert a new axis of length one at `axis`."""
    logger.debug(f"Expanding dims axis={axis}")
    return ndarray(_expand_dims(a, axis))
