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

#include <asnumpy/dtypes/dtype_table.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/device_memory_pool.hpp>
#include <aclnnop/aclnn_permute.h>
#include <cstddef>

namespace {

// Overflow-checked multiplication for shape / byte-size math: a wrapped
// value would allocate less memory than the tensor claims to address.
int64_t CheckedMul(int64_t a, int64_t b) {
    int64_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
        throw std::invalid_argument(fmt::format("size calculation overflow: {} * {} exceeds int64 range", a, b));
    }
    return result;
}

// aclCreateTensor returns nullptr on failure and it was previously never
// checked, so a failure surfaced later as a null tensor in an operator call,
// far from the cause.
aclTensor* CreateTensorChecked(const int64_t* view_dims, uint64_t view_ndim, aclDataType dtype,
                               const int64_t* strides, int64_t offset, aclFormat format, const int64_t* storage_dims,
                               uint64_t storage_ndim, void* data, const char* where) {
    aclTensor* tensor = aclCreateTensor(view_dims, view_ndim, dtype, strides, offset, format, storage_dims,
                                        storage_ndim, data);
    if (!tensor) {
        throw std::runtime_error(fmt::format("[npu_array.cpp]({}) aclCreateTensor returned null", where));
    }
    return tensor;
}

} // namespace

/**
 * @brief Build an array that shares `src`'s storage at a byte offset.
 *
 * Used by Reshape/Flatten/Ravel/GetItem. Callers are responsible for having
 * established that the region is contiguous and inside `src`; this constructor
 * trusts them and allocates nothing.
 */
NPUArray::NPUArray(const NPUArray& src, const std::vector<int64_t>& new_shape, int64_t offset_bytes)
    : shape(new_shape),
      dtype(src.dtype),
      aclDtype(src.aclDtype),
      tensorSize(GetShapeSize(new_shape)),
      buffer(src.buffer),
      byteOffset(offset_bytes) {
    this->strides.resize(this->shape.size());
    int64_t currentStride = 1;
    for (int64_t i = static_cast<int64_t>(this->shape.size()) - 1; i >= 0; i--) {
        this->strides[i] = currentStride;
        currentStride *= this->shape[i];
    }
    // The aclTensor points at buffer + offset, so operators never learn that
    // their input is a view.
    tensorPtr = CreateTensorChecked(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0,
                                    ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->device_address(),
                                    "view ctor");
}

/**
 * @brief Return the allocation to the pool when the last referent goes away.
 *
 * Not to aclrtFree directly: the pool matches blocks on exact byte size, and
 * this is the only place that knows the size the block was allocated with.
 */
DeviceBuffer::~DeviceBuffer() { asnumpy::DeviceMemoryPool::Release(ptr_, bytes_); }

/**
 * @brief Constructor that creates an NPUArray with specified shape and data type.
 *
 * Creates an array (aclTensor) stored on NPU by calling aclCreateTensor,
 * and initializes its shape and data type.
 *
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param dtype np.dtype defining the data type of array elements.
 * @throws std::runtime_error If memory allocation fails or data type is not supported.
 */
NPUArray::NPUArray(const std::vector<int64_t>& shape, py::dtype dtype) {
    this->shape = shape;
    this->dtype = dtype;
    this->aclDtype = GetACLDataType(dtype);
    tensorSize = GetShapeSize(shape);
    auto tensorByteSize = CheckedMul(static_cast<int64_t>(this->tensorSize), GetDataTypeSize(this->aclDtype));
    this->byteOffset = 0;
    if (tensorByteSize > 0) {
        this->buffer = std::make_shared<DeviceBuffer>(asnumpy::DeviceMemoryPool::Allocate(
                                                          static_cast<size_t>(tensorByteSize)),
                                                      static_cast<size_t>(tensorByteSize));
    }
    this->strides.resize(this->shape.size());
    int64_t currentStride = 1; // int64_t: auto would deduce int and truncate large shapes
    for (int64_t i = this->shape.size() - 1; i >= 0; i--) {
        this->strides[i] = currentStride;
        currentStride *= this->shape[i];
    }
    tensorPtr = CreateTensorChecked(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0,
                                    ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->device_address(),
                                    "ctor");
}

/**
 * @brief Constructor that creates an NPUArray with specified shape and ACL data type.
 *
 * Creates an array (aclTensor) stored on NPU by calling aclCreateTensor,
 * and initializes its shape and ACL data type directly.
 * This constructor bypasses NumPy dtype conversion for better performance.
 *
 * @param shape Vector containing array dimensions, defining the array shape.
 * @param acl_type ACL data type constant.
 * @throws std::runtime_error If memory allocation fails or data type is not supported.
 */
NPUArray::NPUArray(const std::vector<int64_t>& shape, aclDataType acl_type) {
    this->shape = shape;
    this->aclDtype = acl_type;
    this->tensorSize = GetShapeSize(shape);
    auto tensorByteSize = CheckedMul(static_cast<int64_t>(this->tensorSize), GetDataTypeSize(this->aclDtype));

    // use ACL type directly; do not create NumPy dtype
    // for compatibility, create an empty py::dtype object
    this->dtype = GetPyDtype(acl_type);

    this->byteOffset = 0;
    if (tensorByteSize > 0) {
        this->buffer = std::make_shared<DeviceBuffer>(asnumpy::DeviceMemoryPool::Allocate(
                                                          static_cast<size_t>(tensorByteSize)),
                                                      static_cast<size_t>(tensorByteSize));
    }
    this->strides.resize(this->shape.size());
    int64_t currentStride = 1; // int64_t: auto would deduce int and truncate large shapes
    for (int64_t i = this->shape.size() - 1; i >= 0; i--) {
        this->strides[i] = currentStride;
        currentStride *= this->shape[i];
    }
    tensorPtr = CreateTensorChecked(this->shape.data(), this->shape.size(), acl_type, this->strides.data(), 0,
                                    ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->device_address(),
                                    "ctor");
}

/**
 * @brief Copy constructor - deep copy.
 *
 * Creates a new NPUArray with the same content as the given NPUArray.
 * The new object owns its own memory space and is completely independent of the original.
 *
 * @param other The NPUArray to copy from.
 */
NPUArray::NPUArray(const NPUArray& other) {
    this->shape = other.shape;
    this->dtype = other.dtype;
    this->aclDtype = other.aclDtype;
    this->tensorSize = other.tensorSize;
    this->strides = other.strides;
    this->byteOffset = 0;
    auto tensorByteSize = CheckedMul(static_cast<int64_t>(this->tensorSize), GetDataTypeSize(this->aclDtype));
    if (tensorByteSize > 0) {
        this->buffer = std::make_shared<DeviceBuffer>(asnumpy::DeviceMemoryPool::Allocate(
                                                          static_cast<size_t>(tensorByteSize)),
                                                      static_cast<size_t>(tensorByteSize));
    }
    this->tensorPtr = CreateTensorChecked(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(),
                                          0, ACL_FORMAT_ND, this->shape.data(), this->shape.size(),
                                          this->device_address(), "copy ctor");
    if (tensorByteSize > 0) {
        void* srcPtr = nullptr;
        auto error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
        ACL_RT_CHECK(error, "aclGetRawTensorAddr");
        if (!srcPtr) {
            throw std::runtime_error("[npu_array.cpp](NPUArray) aclGetRawTensorAddr returned null pointer");
        }
        error =
            aclrtMemcpy(this->device_address(), tensorByteSize, srcPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
        ACL_RT_CHECK(error, "aclrtMemcpy");
        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    }
}

/**
 * @brief Move constructor.
 *
 * Transfers ownership of resources from the given NPUArray to the new object.
 * The original object is left in an invalid state.
 *
 * @param other The NPUArray to move from.
 */
NPUArray::NPUArray(NPUArray&& other) noexcept {
    this->tensorPtr = other.tensorPtr;
    this->shape = std::move(other.shape);
    this->dtype = other.dtype;
    this->aclDtype = other.aclDtype;
    this->tensorSize = other.tensorSize;
    this->strides = std::move(other.strides);
    this->buffer = std::move(other.buffer);
    this->byteOffset = other.byteOffset;
    other.tensorPtr = nullptr;
    other.byteOffset = 0;
}

/**
 * @brief Copy assignment operator.
 *
 * Implements deep copy assignment, ensuring the current object is completely independent of the right-hand object.
 *
 * @param other The NPUArray to copy from.
 * @return Reference to this NPUArray.
 */
NPUArray& NPUArray::operator=(const NPUArray& other) {
    if (this != &other) {
        // release old resources
        if (this->tensorPtr) {
            aclDestroyTensor(this->tensorPtr);
            this->tensorPtr = nullptr;
        }
        this->buffer.reset();
        this->byteOffset = 0;

        this->shape = other.shape;
        this->dtype = other.dtype;
        this->aclDtype = other.aclDtype;
        this->tensorSize = other.tensorSize;
        this->strides = other.strides;

        auto tensorByteSize = CheckedMul(static_cast<int64_t>(this->tensorSize), GetDataTypeSize(this->aclDtype));
        if (tensorByteSize > 0) {
            this->buffer = std::make_shared<DeviceBuffer>(asnumpy::DeviceMemoryPool::Allocate(
                                                              static_cast<size_t>(tensorByteSize)),
                                                          static_cast<size_t>(tensorByteSize));
        }
        this->tensorPtr =
            CreateTensorChecked(this->shape.data(), this->shape.size(), this->aclDtype, this->strides.data(), 0,
                                ACL_FORMAT_ND, this->shape.data(), this->shape.size(), this->device_address(),
                                "operator=");
        if (tensorByteSize > 0) {
            void* srcPtr = nullptr;
            auto error = aclGetRawTensorAddr(other.tensorPtr, &srcPtr);
            ACL_RT_CHECK(error, "aclGetRawTensorAddr");
            if (!srcPtr) {
                throw std::runtime_error("[npu_array.cpp](operator=) aclGetRawTensorAddr returned null pointer");
            }
            error = aclrtMemcpy(this->device_address(), tensorByteSize, srcPtr, tensorByteSize,
                                ACL_MEMCPY_DEVICE_TO_DEVICE);
            ACL_RT_CHECK(error, "aclrtMemcpy");
            error = aclrtSynchronizeDevice();
            ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        }
    }
    return *this;
}

/**
 * @brief Move assignment operator.
 *
 * Transfers ownership of resources from the given NPUArray to the current object.
 * The original object is left in an invalid state.
 *
 * @param other The NPUArray to move from.
 * @return Reference to this NPUArray.
 */
NPUArray& NPUArray::operator=(NPUArray&& other) noexcept {
    if (this != &other) {
        // release old resources
        if (this->tensorPtr) {
            aclDestroyTensor(this->tensorPtr);
            this->tensorPtr = nullptr;
        }
        this->buffer.reset();

        this->tensorPtr = other.tensorPtr;
        this->shape = std::move(other.shape);
        this->dtype = other.dtype;
        this->aclDtype = other.aclDtype;
        this->tensorSize = other.tensorSize;
        this->strides = std::move(other.strides);
        this->buffer = std::move(other.buffer);
        this->byteOffset = other.byteOffset;

        other.tensorPtr = nullptr;
        other.byteOffset = 0;
    }
    return *this;
}

/**
 * @brief Destructor that releases resources occupied by NPUArray.
 */
NPUArray::~NPUArray() {
    // Only the tensor handle is destroyed here. The allocation belongs to
    // `buffer`, whose shared_ptr returns it to the pool when the last array
    // referencing it goes away -- which is what keeps a view alive after the
    // array it was made from is gone.
    if (this->tensorPtr) {
        aclDestroyTensor(this->tensorPtr);
        this->tensorPtr = nullptr;
    }
}

/**
 * @brief Static method to create NPUArray from NumPy array.
 *
 * Creates an NPUArray from a NumPy array and copies data
 * from host memory to NPU device memory.
 *
 * @param host_data Input NumPy array.
 * @return NPUArray The created NPUArray.
 * @throws std::runtime_error If getting tensor data pointer fails or data copy fails.
 */
NPUArray NPUArray::FromNumpy(py::array hostData) {
    py::buffer_info info = hostData.request();
    // the H2D copy below assumes a C-contiguous buffer; normalize sliced /
    // transposed / negative-stride inputs first, mirroring torch.from_numpy
    auto is_c_contiguous = [&info] {
        int64_t expected = info.itemsize;
        for (size_t i = info.ndim; i-- > 0;) {
            if (info.strides[i] != expected) {
                return false;
            }
            expected *= info.shape[i];
        }
        return true;
    }();
    if (!is_c_contiguous) {
        hostData = py::array::ensure(py::module_::import("numpy").attr("ascontiguousarray")(hostData));
        info = hostData.request();
    }
    auto tensorByteSize = CheckedMul(static_cast<int64_t>(info.size), static_cast<int64_t>(info.itemsize));
    auto result = NPUArray(info.shape, hostData.dtype());
    if (tensorByteSize == 0)
        return result;
    void* rawDataPtr = nullptr;
    auto error = aclGetRawTensorAddr(result.tensorPtr, &rawDataPtr);
    ACL_RT_CHECK(error, "aclGetRawTensorAddr");
    if (!rawDataPtr) {
        throw std::runtime_error("[npu_array.cpp](FromNumpy) aclGetRawTensorAddr returned null pointer");
    }
    error = aclrtMemcpy(rawDataPtr, tensorByteSize, info.ptr, tensorByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_RT_CHECK(error, "aclrtMemcpy");
    error = aclrtSynchronizeStream(nullptr);
    ACL_RT_CHECK(error, "aclrtSynchronizeStream");
    return result;
}

/**
 * @brief Convert NPUArray to NumPy array.
 *
 * Copies data from NPU device memory to host memory and returns a NumPy array.
 *
 * @return py::array The converted NumPy array.
 * @throws std::runtime_error If getting tensor data pointer fails or data copy fails.
 * @throws std::runtime_error If tensor size doesn't match NumPy array size.
 */
py::array NPUArray::ToNumpy() const {
    auto tensorByteSize = CheckedMul(static_cast<int64_t>(this->tensorSize), GetDataTypeSize(this->aclDtype));

    // create result array
    py::array result(this->dtype, this->shape);
    py::buffer_info info = result.request();
    if (tensorByteSize == 0)
        return result;

    // The host buffer is sized from `dtype`, the device buffer from `aclDtype`. dtype_table keeps
    // those in exact correspondence, so a mismatch means that invariant is broken. Check before
    // any copy: this guard previously sat in only one branch, which is how a float16 array came to
    // be written as 4-byte floats into a 2-byte-per-element buffer.
    if (static_cast<int64_t>(info.size * info.itemsize) != static_cast<int64_t>(tensorByteSize))
        throw std::runtime_error(
            fmt::format("[npu_array.cpp](ToNumpy) size mismatch: host buffer is {} bytes for dtype '{}', but the "
                        "device tensor is {} bytes for aclDataType '{}'",
                        info.size * info.itemsize, py::str(this->dtype).cast<std::string>(), tensorByteSize,
                        asnumpy::dtypes::Name(this->aclDtype)));

    void* rawDataPtr = nullptr;
    auto error = aclGetRawTensorAddr(this->tensorPtr, &rawDataPtr);
    ACL_RT_CHECK(error, "aclGetRawTensorAddr");
    if (!rawDataPtr) {
        throw std::runtime_error("[npu_array.cpp](ToNumpy) aclGetRawTensorAddr returned null pointer");
    }

    // Every supported dtype has an identical host and device representation (NumPy float16 and
    // ACL_FLOAT16 are both IEEE-754 binary16), so a raw copy is exact for all of them.
    error = aclrtMemcpy(info.ptr, tensorByteSize, rawDataPtr, tensorByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_RT_CHECK(error, "aclrtMemcpy");

    return result;
}

/**
 * @brief Helper function to calculate total size of array.
 *
 * Calculates the total number of elements for a given shape.
 *
 * @param shape Vector containing array dimensions, defining the array shape.
 * @return int64_t Total number of elements in the array.
 * @throws std::runtime_error If any dimension in shape is less than or equal to 0.
 */
int64_t NPUArray::GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i < 0) {
            throw std::runtime_error("[npu_array.cpp](GetShapeSize) Shape Dimensions Must Be Non-Negative!");
        }
        shapeSize = CheckedMul(shapeSize, i);
    }
    return shapeSize;
}

/**
 * @brief Helper function to convert py::dtype to aclDataType.
 *
 * Thin delegation to asnumpy::dtypes::AclFromNumpy, which owns the mapping. Do not reintroduce
 * mapping logic here: the table is the single source of truth.
 *
 * @param dtype Input py::dtype.
 * @return aclDataType The converted aclDataType.
 * @throws std::invalid_argument If the dtype is big-endian, structured, or has no ACL equivalent.
 *         Surfaces to Python as ValueError.
 */
aclDataType NPUArray::GetACLDataType(py::dtype dtype) { return asnumpy::dtypes::AclFromNumpy(dtype); }

/**
 * @brief Helper function to convert aclDataType to py::dtype.
 *
 * Thin delegation to asnumpy::dtypes::NumpyFromAcl. Exact inverse of GetACLDataType over the
 * supported set.
 *
 * @param acl_type Input aclDataType.
 * @return py::dtype The converted py::dtype.
 * @throws std::invalid_argument If `acl_type` has no NumPy equivalent (bf16, fp8/6/4, int4,
 *         uint1, complex32). Surfaces to Python as ValueError.
 */
py::dtype NPUArray::GetPyDtype(aclDataType acl_type) { return asnumpy::dtypes::NumpyFromAcl(acl_type); }

/**
 * @brief Helper function to get byte size corresponding to aclDataType.
 *
 * Thin delegation to asnumpy::dtypes::ItemSize. Defined for unsupported ACL types too, so
 * byte-size math stays available for diagnostics.
 *
 * @param dataType Input aclDataType.
 * @return int64_t Byte size of the data type.
 * @throws std::invalid_argument If `dataType` is unknown. Surfaces to Python as ValueError.
 */
int64_t NPUArray::GetDataTypeSize(aclDataType dataType) { return asnumpy::dtypes::ItemSize(dataType); }

/**
 * @brief Reshape array to new shape
 */
NPUArray NPUArray::Reshape(const std::vector<int64_t>& new_shape) const {
    // Calculate total size for validation
    int64_t new_size = 1;
    int infer_dim = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_dim != -1) {
                throw std::invalid_argument("Only one dimension can be -1 in reshape");
            }
            infer_dim = static_cast<int>(i);
        } else if (new_shape[i] < 0) {
            throw std::invalid_argument("Negative dimensions (other than -1) are not allowed");
        } else {
            new_size *= new_shape[i];
        }
    }

    // Handle -1 dimension inference
    std::vector<int64_t> final_shape = new_shape;
    if (infer_dim != -1) {
        if (new_size == 0) {
            throw std::invalid_argument("Cannot infer dimension from shape with zero elements");
        }
        if (this->tensorSize % new_size != 0) {
            throw std::invalid_argument("Cannot reshape array of size " + std::to_string(this->tensorSize) +
                                        " into shape with inferred dimension");
        }
        final_shape[infer_dim] = this->tensorSize / new_size;
        new_size = this->tensorSize;
    }

    // Validate total size matches
    if (new_size != static_cast<int64_t>(this->tensorSize)) {
        throw std::invalid_argument("Cannot reshape array of size " + std::to_string(this->tensorSize) +
                                    " into shape with size " + std::to_string(new_size));
    }

    // A reshape never reorders elements, so the result is the same storage
    // seen through a different shape: no allocation, no copy. Previously this
    // allocated a fresh buffer and copied every byte, which cost 528 us on a
    // 4 MB array -- 6.7x more than running an actual multiply kernel over it.
    return NPUArray(*this, final_shape, this->byteOffset);
}

/**
 * @brief Flatten array to 1D
 */
NPUArray NPUArray::Flatten() const {
    std::vector<int64_t> flat_shape = {static_cast<int64_t>(this->tensorSize)};
    return this->Reshape(flat_shape);
}

/**
 * @brief Ravel array (alias for flatten)
 */
NPUArray NPUArray::Ravel() const {
    return this->Flatten();
}

/**
 * @brief Basic integer indexing over the leading axes.
 */
NPUArray NPUArray::GetItem(const std::vector<int64_t>& indices) const {
    const size_t ndim = this->shape.size();
    if (indices.size() > ndim) {
        // out_of_range, not invalid_argument: the module translator maps it to
        // Python IndexError, which is what NumPy raises here.
        throw std::out_of_range(fmt::format("too many indices for array: array is {}-dimensional, but {} were indexed",
                                            ndim, indices.size()));
    }

    // Normalize negative indices and bounds-check against the axis they select.
    // Done before any allocation so a bad index cannot leak device memory.
    std::vector<int64_t> resolved(indices.size());
    for (size_t axis = 0; axis < indices.size(); ++axis) {
        int64_t idx = indices[axis];
        const int64_t dim = this->shape[axis];
        if (idx < 0) {
            idx += dim;
        }
        if (idx < 0 || idx >= dim) {
            throw std::out_of_range(fmt::format("index {} is out of bounds for axis {} with size {}", indices[axis],
                                                axis, dim));
        }
        resolved[axis] = idx;
    }

    // Trailing axes that were not indexed survive into the result shape.
    std::vector<int64_t> out_shape(this->shape.begin() + indices.size(), this->shape.end());

    // strides are in elements (byte strides are only synthesized at the Python
    // boundary), so this offset is an element offset.
    int64_t element_offset = 0;
    for (size_t axis = 0; axis < resolved.size(); ++axis) {
        element_offset += CheckedMul(resolved[axis], this->strides[axis]);
    }

    // Fixing a run of leading indices leaves every trailing axis whole, so the
    // selected region is one contiguous block -- which is what makes a view
    // possible here at all. An index run with a gap (a[i, :, k]) would not be.
    const int64_t itemsize = GetDataTypeSize(this->aclDtype);
    const int64_t byte_offset = CheckedMul(element_offset, itemsize);
    return NPUArray(*this, out_shape, this->byteOffset + byte_offset);
}

/**
 * @brief Permute the array's axes via aclnnPermute.
 */
NPUArray NPUArray::Transpose(const std::vector<int64_t>& axes) const {
    const int64_t ndim = static_cast<int64_t>(this->shape.size());

    // Empty means "reverse all axes" -- NumPy's default and what .T uses.
    std::vector<int64_t> perm;
    if (axes.empty()) {
        perm.resize(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            perm[i] = ndim - 1 - i;
        }
    } else {
        if (static_cast<int64_t>(axes.size()) != ndim) {
            throw std::invalid_argument(
                fmt::format("transpose: axes don't match array: got {} axes for {}-dimensional array", axes.size(),
                            ndim));
        }
        perm = axes;
        std::vector<bool> seen(ndim, false);
        for (auto& axis : perm) {
            const int64_t original = axis;
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                throw std::invalid_argument(
                    fmt::format("transpose: axis {} is out of bounds for array of dimension {}", original, ndim));
            }
            // A repeated axis would silently drop another one, producing a
            // result whose shape is right but whose contents are not.
            if (seen[axis]) {
                throw std::invalid_argument(fmt::format("transpose: repeated axis {} in axes", axis));
            }
            seen[axis] = true;
        }
    }

    std::vector<int64_t> out_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape[i] = this->shape[perm[i]];
    }
    NPUArray out(out_shape, this->aclDtype);

    // A 0-d or 1-d array is its own transpose, and aclnnPermute need not be
    // called for an empty result.
    if (out.tensorSize == 0 || ndim < 2) {
        if (out.tensorSize > 0) {
            const int64_t bytes = CheckedMul(static_cast<int64_t>(out.tensorSize), GetDataTypeSize(this->aclDtype));
            auto error =
                aclrtMemcpy(out.device_address(), bytes, this->device_address(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
            ACL_RT_CHECK(error, "aclrtMemcpy in Transpose");
            error = aclrtSynchronizeDevice();
            ACL_RT_CHECK(error, "aclrtSynchronizeDevice in Transpose");
        }
        return out;
    }

    aclIntArray* dims = aclCreateIntArray(perm.data(), perm.size());
    if (!dims) {
        throw std::runtime_error("[npu_array.cpp](Transpose) failed to create aclIntArray");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnPermuteGetWorkspaceSize(this->tensorPtr, dims, out.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        aclDestroyIntArray(dims);
        ACLNN_CHECK(error, "aclnnPermuteGetWorkspaceSize");
    }

    {
        asnumpy::AclWorkspace workspace(workspaceSize);
        error = aclnnPermute(workspace.get(), workspaceSize, executor, nullptr);
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(dims);
            ACLNN_CHECK(error, "aclnnPermute");
        }
        error = aclrtSynchronizeDevice();
        if (error != ACL_SUCCESS) {
            aclDestroyIntArray(dims);
            ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        }
    }

    aclDestroyIntArray(dims);
    return out;
}

/**
 * @brief Interchange two axes.
 */
NPUArray NPUArray::SwapAxes(int64_t axis1, int64_t axis2) const {
    const int64_t ndim = static_cast<int64_t>(this->shape.size());
    auto normalize = [ndim](int64_t axis) {
        const int64_t original = axis;
        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            throw std::invalid_argument(
                fmt::format("swapaxes: axis {} is out of bounds for array of dimension {}", original, ndim));
        }
        return axis;
    };

    std::vector<int64_t> perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        perm[i] = i;
    }
    std::swap(perm[normalize(axis1)], perm[normalize(axis2)]);
    return this->Transpose(perm);
}

std::vector<int64_t> GetBroadcastShape(const NPUArray& a, const NPUArray& b) {
    const std::vector<int64_t>& shapeA = a.shape;
    const std::vector<int64_t>& shapeB = b.shape;

    size_t ndimA = shapeA.size();
    size_t ndimB = shapeB.size();
    size_t ndimOut = std::max(ndimA, ndimB);

    std::vector<int64_t> result(ndimOut, 1);

    for (size_t i = 0; i < ndimOut; ++i) {
        int64_t dimA = (i < ndimA) ? shapeA[ndimA - 1 - i] : 1;
        int64_t dimB = (i < ndimB) ? shapeB[ndimB - 1 - i] : 1;

        if (dimA == dimB || dimA == 1 || dimB == 1) {
            result[ndimOut - 1 - i] = std::max(dimA, dimB);
        } else {
            throw std::invalid_argument("[npu_array.cpp](GetBroadcastShape) shapes are not broadcastable. "
                                        "dimA=" +
                                        std::to_string(dimA) + " dimB=" + std::to_string(dimB) + " at axis -" +
                                        std::to_string(i + 1));
        }
    }

    return result;
}
