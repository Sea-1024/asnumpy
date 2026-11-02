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

#include <asnumpy/array/manipulation.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <fmt/format.h>
#include <numeric>
#include <stdexcept>

namespace asnumpy {

namespace {

// Normalize a possibly-negative axis against `ndim`, matching NumPy's wrapping.
int64_t NormalizeAxis(int64_t axis, int64_t ndim, const char* op) {
    const int64_t original = axis;
    if (axis < 0) {
        axis += ndim;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::invalid_argument(
            fmt::format("{}: axis {} is out of bounds for array of dimension {}", op, original, ndim));
    }
    return axis;
}

// Product of shape[from, to). Used to split a C-contiguous array into the
// "rows" that a concatenation along `axis` has to interleave.
int64_t PartialProduct(const std::vector<int64_t>& shape, size_t from, size_t to) {
    int64_t product = 1;
    for (size_t i = from; i < to; ++i) {
        product *= shape[i];
    }
    return product;
}

} // namespace

NPUArray Concatenate(const std::vector<NPUArray>& arrays, int64_t axis) {
    if (arrays.empty()) {
        throw std::invalid_argument("concatenate: need at least one array to concatenate");
    }

    const auto& first = arrays.front();
    const int64_t ndim = static_cast<int64_t>(first.shape.size());
    if (ndim == 0) {
        throw std::invalid_argument("concatenate: zero-dimensional arrays cannot be concatenated");
    }
    const int64_t norm_axis = NormalizeAxis(axis, ndim, "concatenate");

    // Validate every input before allocating, so a rejected call cannot leak
    // device memory.
    int64_t axis_total = 0;
    for (const auto& array : arrays) {
        if (static_cast<int64_t>(array.shape.size()) != ndim) {
            throw std::invalid_argument(
                fmt::format("concatenate: all input arrays must have the same number of dimensions, "
                            "but got {} and {}",
                            ndim, array.shape.size()));
        }
        if (array.aclDtype != first.aclDtype) {
            throw std::invalid_argument("concatenate: all input arrays must have the same dtype");
        }
        for (int64_t dim = 0; dim < ndim; ++dim) {
            if (dim != norm_axis && array.shape[dim] != first.shape[dim]) {
                throw std::invalid_argument(
                    fmt::format("concatenate: all input array dimensions except for the concatenation axis "
                                "must match exactly, but along dimension {} got {} and {}",
                                dim, first.shape[dim], array.shape[dim]));
            }
        }
        axis_total += array.shape[norm_axis];
    }

    std::vector<int64_t> out_shape = first.shape;
    out_shape[norm_axis] = axis_total;
    NPUArray out(out_shape, first.aclDtype);

    const int64_t itemsize = NPUArray::GetDataTypeSize(first.aclDtype);
    // Everything before `axis` iterates; everything at/after `axis` is one
    // contiguous run per input, because the arrays are dense and C-contiguous.
    const int64_t outer = PartialProduct(out_shape, 0, static_cast<size_t>(norm_axis));
    const int64_t inner = PartialProduct(out_shape, static_cast<size_t>(norm_axis) + 1, out_shape.size());

    auto* dstBase = static_cast<char*>(out.device_address());
    int64_t axis_offset = 0;
    for (const auto& array : arrays) {
        const int64_t chunk = array.shape[norm_axis] * inner; // elements copied per outer step
        if (chunk == 0) {
            continue; // empty input contributes nothing
        }
        const auto* srcBase = static_cast<const char*>(array.device_address());
        for (int64_t o = 0; o < outer; ++o) {
            const int64_t src_elem = o * chunk;
            const int64_t dst_elem = o * axis_total * inner + axis_offset * inner;
            const int64_t bytes = chunk * itemsize;
            auto error = aclrtMemcpy(dstBase + dst_elem * itemsize, bytes, srcBase + src_elem * itemsize, bytes,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
            ACL_RT_CHECK(error, "aclrtMemcpy");
        }
        axis_offset += array.shape[norm_axis];
    }

    auto error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    return out;
}

NPUArray Stack(const std::vector<NPUArray>& arrays, int64_t axis) {
    if (arrays.empty()) {
        throw std::invalid_argument("stack: need at least one array to stack");
    }

    const auto& first = arrays.front();
    // The new axis may land one past the last existing axis, so the bound here
    // is ndim + 1 rather than ndim.
    const int64_t out_ndim = static_cast<int64_t>(first.shape.size()) + 1;
    const int64_t norm_axis = NormalizeAxis(axis, out_ndim, "stack");

    for (const auto& array : arrays) {
        if (array.shape != first.shape) {
            throw std::invalid_argument("stack: all input arrays must have the same shape");
        }
        if (array.aclDtype != first.aclDtype) {
            throw std::invalid_argument("stack: all input arrays must have the same dtype");
        }
    }

    std::vector<int64_t> out_shape = first.shape;
    out_shape.insert(out_shape.begin() + norm_axis, static_cast<int64_t>(arrays.size()));
    NPUArray out(out_shape, first.aclDtype);

    const int64_t itemsize = NPUArray::GetDataTypeSize(first.aclDtype);
    // Each input contributes a size-1 slab at `norm_axis`, so the layout is the
    // same as concatenating arrays whose axis length is 1.
    const int64_t outer = PartialProduct(first.shape, 0, static_cast<size_t>(norm_axis));
    const int64_t inner = PartialProduct(first.shape, static_cast<size_t>(norm_axis), first.shape.size());

    auto* dstBase = static_cast<char*>(out.device_address());
    const int64_t count = static_cast<int64_t>(arrays.size());
    for (int64_t i = 0; i < count; ++i) {
        if (inner == 0) {
            continue;
        }
        const auto* srcBase = static_cast<const char*>(arrays[i].device_address());
        for (int64_t o = 0; o < outer; ++o) {
            const int64_t src_elem = o * inner;
            const int64_t dst_elem = o * count * inner + i * inner;
            const int64_t bytes = inner * itemsize;
            auto error = aclrtMemcpy(dstBase + dst_elem * itemsize, bytes, srcBase + src_elem * itemsize, bytes,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
            ACL_RT_CHECK(error, "aclrtMemcpy");
        }
    }

    auto error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    return out;
}

NPUArray Squeeze(const NPUArray& array, const std::vector<int64_t>& axes) {
    const int64_t ndim = static_cast<int64_t>(array.shape.size());

    std::vector<int64_t> out_shape;
    if (axes.empty()) {
        // Drop every length-1 axis. A fully length-1 array squeezes to 0-d,
        // matching numpy.squeeze rather than collapsing to shape (1,).
        for (int64_t dim : array.shape) {
            if (dim != 1) {
                out_shape.push_back(dim);
            }
        }
    } else {
        std::vector<bool> drop(ndim, false);
        for (int64_t axis : axes) {
            const int64_t original = axis;
            if (axis < 0) {
                axis += ndim;
            }
            if (axis < 0 || axis >= ndim) {
                throw std::invalid_argument(
                    fmt::format("squeeze: axis {} is out of bounds for array of dimension {}", original, ndim));
            }
            if (drop[axis]) {
                throw std::invalid_argument(fmt::format("squeeze: repeated axis {}", axis));
            }
            // NumPy refuses to squeeze a non-unit axis rather than silently
            // leaving it in place, because that would change the result rank.
            if (array.shape[axis] != 1) {
                throw std::invalid_argument(
                    fmt::format("squeeze: cannot select an axis to squeeze out which has size {} not equal to one",
                                array.shape[axis]));
            }
            drop[axis] = true;
        }
        for (int64_t i = 0; i < ndim; ++i) {
            if (!drop[i]) {
                out_shape.push_back(array.shape[i]);
            }
        }
    }

    // Squeeze only relabels axes; the element order is unchanged, so this is a
    // plain reshape over the same contiguous buffer.
    return array.Reshape(out_shape);
}

NPUArray ExpandDims(const NPUArray& array, int64_t axis) {
    // The new axis may land one past the last existing axis, hence ndim + 1.
    const int64_t out_ndim = static_cast<int64_t>(array.shape.size()) + 1;
    const int64_t norm_axis = NormalizeAxis(axis, out_ndim, "expand_dims");

    std::vector<int64_t> out_shape = array.shape;
    out_shape.insert(out_shape.begin() + norm_axis, 1);
    return array.Reshape(out_shape);
}

} // namespace asnumpy
