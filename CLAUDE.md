# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AsNumpy is a NumPy-compatible library for Huawei Ascend NPU (910B) acceleration. It provides a three-layer architecture:
- **Python frontend** (`src/asnumpy/`) - NumPy-compatible API
- **Pybind11 bindings** (`bindings/python/`) - C++ to Python interface
- **C++ core** (`csrc/`, `include/`) - CANN ACLNN operator wrappers

The core data structure is `NPUArray`, which mirrors `numpy.ndarray` but executes operations on the NPU via CANN runtime.

## Build System

Uses **scikit-build-core** (CMake + Python build frontend), not traditional setuptools.

### Environment Setup

CANN SDK path must be set before building:
```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
```

### Build Commands

**Development build (editable install)**
```bash
pip install -e .
```

**Full rebuild**
```bash
pip uninstall asnumpy -y && pip install -e .
```

**Production wheel build**
```bash
python -m build
pip install dist/*.whl
```

The build system uses scikit-build-core's editable mode with "redirect" strategy, so code changes in `src/asnumpy/*.py` are reflected immediately without rebuild. C++ changes require `pip install -e .` again.

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Module
```bash
# Math module tests
pytest tests/asnumpy_tests/math_tests/

# Logic module tests
pytest tests/asnumpy_tests/logic_tests/
```

### Run Single Test File or Function
```bash
# Single file
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py

# Single function
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py::test_sinc_basic
```

### Parallel Testing
```bash
# Use pytest-xdist for parallel execution
pytest tests/ -n auto
```

### Test Options
```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Multi-NPU tests (requires multiple NPU devices)
pytest --multi-npu

# Specify NPU device
pytest --npu-id 1
```

## Code Quality

### Linting and Formatting

**Run all pre-commit hooks manually**
```bash
pre-commit run --all-files
```

**Individual tools**
```bash
# Python linting (ruff)
ruff check src/ tests/

# Python formatting (ruff)
ruff format src/ tests/

# Type checking (mypy)
mypy src/

# C++ formatting (clang-format)
clang-format -i csrc/**/*.cpp include/**/*.hpp
```

Pre-commit hooks run automatically on `git commit` if installed via `pre-commit install`.

## Architecture

### Three-Layer Design

1. **Python Frontend** (`src/asnumpy/`)
   - `__init__.py` - Main exports (122+ functions)
   - `math.py` - Mathematical functions (80+ ops: sin, cos, exp, log, etc.)
   - `linalg/` - Linear algebra
   - `logic.py` - Logical operations (16 functions)
   - `array.py` - Array creation functions
   - `random/` - Random sampling
   - `statistics.py` - Statistical functions
   - `nn.py` - Neural network ops (softmax)
   - `utils.py` - Core `ndarray` class wrapper

2. **Pybind11 Bindings** (`bindings/python/`)
   - `module.cpp` - Main module definition
   - `bind_math.cpp` - Math function bindings
   - `bind_logic.cpp` - Logic function bindings
   - `bind_array.cpp` - Array creation bindings
   - `bind_utils.cpp` - NPUArray class binding
   - Each `bind_*.cpp` exposes C++ functions to Python

3. **C++ Core** (`csrc/`, `include/`)
   - `utils/npu_array.cpp` - Core `NPUArray` class (RAII, device memory, CANN tensor management)
   - `cann/driver.cpp` - Device initialization and lifecycle
   - `math/` - Math operator implementations
   - `logic/` - Logic operator implementations
   - `dtypes/` - Data type system and promotion rules

### NPUArray Design

`NPUArray` is the core data structure with:
- **dtype** (py::dtype) - Element type
- **shape** (std::vector<int64_t>) - Dimensions
- **strides** (std::vector<int64_t>) - Memory layout, **in elements, not bytes**. Byte strides exist only at the Python boundary.
- **tensorPtr** (aclTensor*) - CANN tensor handle, pointing at `buffer + byteOffset`
- **buffer** (std::shared_ptr\<DeviceBuffer\>, private) - Refcounted device allocation
- **byteOffset** (int64_t, private) - Start of this array's data within `buffer`

**Shared storage, zero-copy views.** `Reshape`, `Flatten`, `Ravel` and `GetItem` return
views: they share `buffer` with their source and allocate nothing. Semantics match NumPy,
so a view and its source alias the same memory. Nothing can currently write in place (there
is no `__setitem__` and `multiply()` rejects `out=`), but any future in-place API must
honor that rule. `Transpose` is the deliberate exception — it materializes, because a
permuted array is not C-contiguous and downstream operators assume they are.

**Memory pooling.** Allocations go through `asnumpy::DeviceMemoryPool`
(`include/asnumpy/utils/device_memory_pool.hpp`), never `aclrtMalloc` directly. The pool
caches freed blocks by exact byte size (capped at 256 MB, evicting largest-first), because
the driver's allocation cost is not proportional to the request — 64 MB measured 399 µs
against 15 µs for 4 MB, and every operator allocates at least one output. If you add a
place that needs device memory, go through the pool and call `Clear()` before any device
reset. Workspace allocations inside individual operator bodies still use their own
`aclrtMalloc`/`aclrtFree` pairs and are not pooled.

**RAII resource management**: the destructor calls `aclDestroyTensor`; `DeviceBuffer`'s
destructor returns the allocation to the pool when the last referencing array dies. No
manual cleanup needed, and a view stays valid after the array it was made from is gone.

**Data transfer**:
- `FromNumpy(np.ndarray)` → NPU (ACL_MEMCPY_HOST_TO_DEVICE)
- `ToNumpy()` → CPU (ACL_MEMCPY_DEVICE_TO_HOST)

### Directory Structure

```
csrc/
├── array/          # Array creation ops
├── cann/           # CANN driver and device management
├── dtypes/         # Data type system
├── linalg/         # Linear algebra ops
├── logic/          # Logical ops
├── math/           # Math ops (largest module)
├── nn/             # Neural network ops
├── random/         # Random sampling
├── sorting/        # Sorting ops
├── statistics/     # Statistical ops
└── utils/          # NPUArray, ACL resources, error handling

include/asnumpy/
├── array/
├── cann/
├── dtypes/
├── linalg/
├── logic/
├── math/
├── nn/
├── random/
├── sorting/
├── statistics/
└── utils/

bindings/python/
├── module.cpp      # PYBIND11_MODULE(_core, ...)
├── bind_*.cpp      # Per-module bindings

src/asnumpy/
├── __init__.py     # Main exports
├── math.py         # Math function wrappers
├── logic.py        # Logic function wrappers
├── array.py        # Array creation wrappers
├── utils.py        # ndarray class
└── testing/        # Test utilities and decorators
```

## Development Workflow

### Adding a New Operator

Follow this sequence (example: adding `sinc` function):

1. **Add C++ declaration** (`include/asnumpy/math/other_special_functions.hpp`)
   ```cpp
   NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);
   ```

2. **Implement C++ function** (`csrc/math/other_special_functions.cpp`)
   - Determine output dtype
   - Create output NPUArray
   - Call CANN operator (e.g., `aclnnSinc`)
   - Handle workspace allocation
   - Synchronize device
   - Return result

3. **Add pybind11 binding** (`bindings/python/bind_math.cpp`)
   ```cpp
   math.def("sinc", &Sinc, py::arg("x"), py::arg("dtype") = py::none());
   ```

4. **Add Python wrapper** (`src/asnumpy/math.py`)
   ```python
   from .lib.asnumpy_core.math import sinc as _ap_sinc
   
   def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
       return ndarray(_ap_sinc(x, _convert_dtype(dtype)))
   ```

5. **Export to main namespace** (`src/asnumpy/__init__.py`)
   ```python
   from .math import sinc
   __all__ = [..., "sinc"]
   ```

6. **Write tests** (`tests/asnumpy_tests/math_tests/test_miscellaneous.py`)
   ```python
   @testing.for_float_dtypes(no_float16=True)
   @testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
   def test_sinc_basic(xp, dtype):
       # Test implementation
   ```

7. **Rebuild and test**
   ```bash
   pip install -e .
   pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py::test_sinc_basic
   ```

### CANN Operator Pattern

Typical CANN operator execution flow:

```cpp
// 1. Determine dtypes
aclDataType out_dtype = /* ... */;

// 2. Create output array
NPUArray out(x.shape, out_py_dtype);

// 3. Get workspace size and executor
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
auto error = aclnnOperatorGetWorkspaceSize(
    x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
);
// Check error

// 4. Allocate workspace if needed
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 5. Execute operator
error = aclnnOperator(workspaceAddr, workspaceSize, executor, nullptr);
// Check error

// 6. Synchronize device
aclrtSynchronizeDevice();

// 7. Free workspace
if (workspaceAddr) aclrtFree(workspaceAddr);

return out;
```

**Error handling**: Always check ACL return values and use `aclGetRecentErrMsg()` for details.

### Testing Utilities

AsNumpy provides decorators for parameterized testing:

**Data type decorators**:
- `@testing.for_float_dtypes(no_float16=True)` - float32, float64
- `@testing.for_int_dtypes()` - int8, int16, int32, int64, uint8, uint16
- `@testing.for_all_dtypes()` - All supported types
- `@testing.for_dtypes([np.float32, np.int32])` - Specific types

**Comparison decorators**:
- `@testing.numpy_asnumpy_allclose(rtol=1e-7, atol=0)` - Float comparison with tolerance
- `@testing.numpy_asnumpy_array_equal()` - Exact comparison

**Test pattern**:
```python
@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_operator(xp, dtype):
    """xp is either numpy or asnumpy, depending on test run."""
    a = xp.array([1.0, 2.0, 3.0], dtype=dtype)
    return xp.operator(a)  # Automatically compares numpy vs asnumpy results
```

## Branch Management

- **Main branch**: `master` (all PRs target this)
- **Feature branches**: `feature/<name>`
- **Bug fixes**: `fix/<name>`
- **Release branches**: `release/vX.Y.Z` (for stabilization)
- **Hotfix branches**: `hotfix/<name>` (backport fixes to master)

**Always use merge commits, never squash.** This preserves authorship and contribution statistics.

**Commit hygiene**: Before submitting a PR, squash trivial commits (typo fixes, formatting) into meaningful commits:
```bash
git rebase -i master
# Mark trivial commits with 's' (squash) or 'f' (fixup)
```

## Important Constraints

1. **Python >= 3.10** - Uses modern type hints and features
2. **CANN >= 8.2.RC1** - Required for operator support
3. **GCC >= 11.2** - C++17 features
4. **CMake >= 3.22** - scikit-build-core requirement
5. **Ascend 910B NPU** - Hardware target

## Symbol Visibility

The build uses `-fvisibility=hidden` and `-fvisibility-inlines-hidden` globally. This prevents symbol conflicts with other extensions (especially fmt/spdlog). Only `PyInit__core` is exported. Never change `CMAKE_CXX_VISIBILITY_PRESET` - see CMakeLists.txt comments for rationale.

## Dependencies

**Build-time**:
- scikit-build-core >= 1.0.0
- pybind11 >= 3.0.0
- numpy >= 2.0
- fmt (system package)
- spdlog (system package, with SPDLOG_FMT_EXTERNAL)

**Runtime**:
- numpy >= 2.0
- loguru >= 0.7.0
- CANN runtime libraries

## OpenBOAT Integration

When CANN built-in operators don't cover a NumPy API, AsNumpy uses [OpenBOAT](https://gitcode.com/HIT1920/OpenBOAT) - an open-source Ascend C operator library developed by the same research group (AISS Group, HIT). OpenBOAT operators are compiled and linked separately but follow the same ACLNN interface pattern.

## Versioning

Version is managed by `setuptools-scm` from git tags:
- Tag format: `vX.Y.Z` (e.g., `v0.2.0`)
- Version written to `src/asnumpy/_version.py` at build time
- No manual version editing needed

## Common Issues

**Build fails with "CANN SDK not found"**: Set `ASCEND_TOOLKIT_HOME` environment variable before building.

**Import error after C++ changes**: Rebuild with `pip install -e .` - Python changes are live, C++ changes need recompilation.

**Test fails with device error**: Check NPU device availability with `npu-smi info`, ensure CANN runtime is properly configured.

**Float16 precision issues**: Many operators have reduced precision or unsupported operations for float16. Use `no_float16=True` in test decorators when appropriate.
