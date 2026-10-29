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

#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_cumprod.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_linalg_cross.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_nansum.h>
#include <aclnnop/aclnn_reduce_sum.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace asnumpy {
namespace {

// NumPy reduction accumulate dtype by default; explicit `dtype` wins.
aclDataType ReductionAccDtype(const NPUArray& a, const std::optional<py::dtype>& dtype) {
    if (dtype.has_value()) {
        return NPUArray::GetACLDataType(dtype.value());
    }
    return PromoteReduction(a.aclDtype);
}

// ACL reduction ops reject UINT64: compute in INT64, cast the result back.
aclDataType ReductionComputeDtype(aclDataType acc) { return acc == ACL_UINT64 ? ACL_INT64 : acc; }

// Cast the result back to the accumulate dtype (uint64 round-trip only).
NPUArray EnsureResultDtype(NPUArray&& result, aclDataType acc) {
    if (result.aclDtype != acc) {
        return CastToDtype(result, acc);
    }
    return std::move(result);
}

// Resolve the reduction input for an accumulate dtype: cast only when the ACL op cannot
// compute in it (uint64). Keeps the cast buffer alive for the caller's scope.
class ReductionInput {
  public:
    ReductionInput(const NPUArray& a, aclDataType acc) : ref_(&a) {
        aclDataType compute = ReductionComputeDtype(acc);
        if (a.aclDtype != compute) {
            casted_ = CastToDtype(a, compute);
            ref_ = &*casted_;
        }
    }
    const NPUArray& get() const { return *ref_; }

  private:
    const NPUArray* ref_;
    std::optional<NPUArray> casted_;
};

// All-axes reduction; avoids aclnnFlatten, which rejects ACL_DOUBLE.
NPUArray ReduceSumAllAxes(const NPUArray& input) {
    std::vector<int64_t> axes(input.shape.size());
    std::iota(axes.begin(), axes.end(), 0);
    AclIntArrayGuard axis_array(aclCreateIntArray(axes.data(), axes.size()));
    std::vector<int64_t> ones(input.shape.size(), 1);
    auto result = NPUArray(ones, input.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceSumGetWorkspaceSize(input.tensorPtr, axis_array.get(), true, result.aclDtype,
                                                result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnReduceSumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnReduceSum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnReduceSum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnReduceSum completed");
    return result;
}

NPUArray ProdWholeTensor(const NPUArray& input) {
    auto result = NPUArray({1}, input.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnProdGetWorkspaceSize(input.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnProdGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnProd(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnProd");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProd completed");
    return result;
}

// All-axes prod with keepdims semantics: (1, ..., 1) result. No multi-axis prod op
// exists, so reduce one axis at a time; keepdims=true keeps the rank (and later axis
// indices) valid. 0-d input has no axis to reduce and falls back to aclnnProd.
NPUArray ProdAllAxes(const NPUArray& input) {
    if (input.shape.empty()) {
        return ProdWholeTensor(input);
    }
    auto shape = input.shape;
    std::optional<NPUArray> acc;
    for (int64_t ax = 0; ax < static_cast<int64_t>(shape.size()); ++ax) {
        const NPUArray& src = acc.has_value() ? *acc : input;
        shape[ax] = 1;
        NPUArray result(shape, input.aclDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnProdDimGetWorkspaceSize(src.tensorPtr, ax, true, result.aclDtype, result.tensorPtr,
                                                  &workspaceSize, &executor);
        ACLNN_CHECK(error, "aclnnProdDimGetWorkspaceSize");
        AclWorkspace workspace(workspaceSize);
        error = aclnnProdDim(workspace.get(), workspaceSize, executor, nullptr);
        ACLNN_CHECK(error, "aclnnProdDim");
        error = aclrtSynchronizeDevice();
        ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
        acc = std::move(result);
    }
    LOG_INFO("aclnnProdDim all-axes prod completed");
    return std::move(*acc);
}

// NaN -> nanValue via aclnnNanToNum; floating inputs only (integers carry no NaN).
NPUArray ReplaceNan(const NPUArray& a, float nanValue) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    auto temp = NPUArray(a.shape, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, nanValue, std::numeric_limits<float>::infinity(),
                                               -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize,
                                               &executor);
    ACLNN_CHECK(error, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnNanToNum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnNanToNum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");
    return temp;
}

} // namespace

NPUArray Prod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnProdDim start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    aclDataType acc = ReductionAccDtype(a, dtype);
    ReductionInput input(a, acc);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    auto result = NPUArray(shape, input.get().aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnProdDimGetWorkspaceSize(input.get().tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr,
                                              &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnProdDimGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnProdDim(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnProdDim");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProdDim completed");
    return EnsureResultDtype(std::move(result), acc);
}

// axis=None + keepdims: NumPy returns a (1, ..., 1) array instead of a scalar.
NPUArray Prod(const NPUArray& a, bool /*keepdims*/, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnProdDim all-axes start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    ReductionInput input(a, acc);
    return EnsureResultDtype(ProdAllAxes(input.get()), acc);
}

// The scalar overload deliberately keeps aclnnProd: one whole-tensor launch instead of
// the per-axis loop of ProdAllAxes.
py::object Prod(const NPUArray& a, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnProd start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape), a.tensorSize,
              AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    ReductionInput input(a, acc);
    auto result = ProdWholeTensor(input.get());
    return EnsureResultDtype(std::move(result), acc).ToNumpy().attr("reshape")(-1)[py::int_(0)];
}

NPUArray Sum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceSum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    aclDataType acc = ReductionAccDtype(a, dtype);
    ReductionInput input(a, acc);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> tmp{axis};
    AclIntArrayGuard axis_array(aclCreateIntArray(tmp.data(), tmp.size()));
    auto result = NPUArray(shape, input.get().aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceSumGetWorkspaceSize(input.get().tensorPtr, axis_array.get(), keepdims, result.aclDtype,
                                                result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnReduceSumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnReduceSum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnReduceSum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnReduceSum completed");
    return EnsureResultDtype(std::move(result), acc);
}

// axis=None + keepdims: NumPy returns a (1, ..., 1) array instead of a scalar.
NPUArray Sum(const NPUArray& a, bool /*keepdims*/, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceSum all-axes start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    ReductionInput input(a, acc);
    return EnsureResultDtype(ReduceSumAllAxes(input.get()), acc);
}

py::object Sum(const NPUArray& a, std::optional<py::dtype> dtype) {
    // Extract the scalar from the all-axes (1, ..., 1) result; same op sequence.
    return Sum(a, false, dtype).ToNumpy().attr("reshape")(-1)[py::int_(0)];
}

NPUArray Nanprod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    aclDataType acc = ReductionAccDtype(a, dtype);
    std::optional<NPUArray> cleaned;
    const NPUArray* base = &a;
    if (IsFloatingAclDtype(a.aclDtype)) {
        cleaned = ReplaceNan(a, 1.0F);
        base = &*cleaned;
    }
    ReductionInput input(*base, acc);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    auto result = NPUArray(shape, input.get().aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_DEBUG("aclnnProdDim start: input_shape={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(input.get().shape), AclDtypeName(input.get().aclDtype), axis, keepdims);
    auto error = aclnnProdDimGetWorkspaceSize(input.get().tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr,
                                              &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnProdDimGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnProdDim(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnProdDim");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnProdDim completed");
    return EnsureResultDtype(std::move(result), acc);
}

// axis=None + keepdims: NumPy returns a (1, ..., 1) array instead of a scalar.
NPUArray Nanprod(const NPUArray& a, bool /*keepdims*/, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum all-axes start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    // NaN cleanup runs on the original floating input BEFORE the acc cast: an explicit
    // integer dtype would otherwise turn NaN into garbage instead of the 1 neutral element.
    std::optional<NPUArray> cleaned;
    const NPUArray* base = &a;
    if (IsFloatingAclDtype(a.aclDtype)) {
        cleaned = ReplaceNan(a, 1.0F);
        base = &*cleaned;
    }
    ReductionInput input(*base, acc);
    return EnsureResultDtype(ProdAllAxes(input.get()), acc);
}

// The scalar overload deliberately keeps aclnnProd: one whole-tensor launch instead of
// the per-axis loop of ProdAllAxes.
py::object Nanprod(const NPUArray& a, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    // NaN cleanup runs on the original floating input BEFORE the acc cast: an explicit
    // integer dtype would otherwise turn NaN into garbage instead of the 1 neutral element.
    std::optional<NPUArray> cleaned;
    const NPUArray* base = &a;
    if (IsFloatingAclDtype(a.aclDtype)) {
        cleaned = ReplaceNan(a, 1.0F);
        base = &*cleaned;
    }
    ReductionInput input(*base, acc);
    auto result = ProdWholeTensor(input.get());
    return EnsureResultDtype(std::move(result), acc).ToNumpy().attr("reshape")(-1)[py::int_(0)];
}

NPUArray Nansum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceNansum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}, keepdims={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), axis, keepdims);
    aclDataType acc = ReductionAccDtype(a, dtype);
    // Pre-clean NaN only when the acc cast would drop the floating dtype (NaN -> garbage
    // integer); otherwise aclnnReduceNansum handles NaN natively.
    std::optional<NPUArray> cleaned;
    const NPUArray* base = &a;
    if (IsFloatingAclDtype(a.aclDtype) && !IsFloatingAclDtype(ReductionComputeDtype(acc))) {
        cleaned = ReplaceNan(a, 0.0F);
        base = &*cleaned;
    }
    ReductionInput input(*base, acc);
    auto shape = a.shape;
    int64_t ax = axis;
    if (axis < 0) {
        ax = shape.size() + axis;
    }
    if (keepdims) {
        shape[ax] = 1;
    } else {
        shape.erase(shape.begin() + ax);
    }
    std::vector<int64_t> tmp{axis};
    AclIntArrayGuard axis_array(aclCreateIntArray(tmp.data(), tmp.size()));
    auto result = NPUArray(shape, input.get().aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnReduceNansumGetWorkspaceSize(input.get().tensorPtr, axis_array.get(), keepdims, result.aclDtype,
                                                   result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnReduceNansumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnReduceNansum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnReduceNansum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnReduceNansum completed");
    return EnsureResultDtype(std::move(result), acc);
}

// axis=None + keepdims: NumPy returns a (1, ..., 1) array instead of a scalar.
NPUArray Nansum(const NPUArray& a, bool /*keepdims*/, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnReduceSum all-axes start: input_shape={}, tensorSize={}, aclDtype={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype));
    aclDataType acc = ReductionAccDtype(a, dtype);
    // NaN cleanup runs on the original floating input BEFORE the acc cast: an explicit
    // integer dtype would otherwise turn NaN into garbage instead of 0. aclnnReduceNansum
    // rejects ACL_DOUBLE, hence NaN -> 0 + ReduceSum.
    std::optional<NPUArray> cleaned;
    const NPUArray* base = &a;
    if (IsFloatingAclDtype(a.aclDtype)) {
        cleaned = ReplaceNan(a, 0.0F);
        base = &*cleaned;
    }
    ReductionInput input(*base, acc);
    return EnsureResultDtype(ReduceSumAllAxes(input.get()), acc);
}

py::object Nansum(const NPUArray& a, std::optional<py::dtype> dtype) {
    // Extract the scalar from the all-axes (1, ..., 1) result; same op sequence.
    return Nansum(a, false, dtype).ToNumpy().attr("reshape")(-1)[py::int_(0)];
}

NPUArray Cumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnCumprod start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    AclScalarGuard axis_scalar(aclCreateScalar(&axis, ACL_INT64));
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error = aclnnCumprodGetWorkspaceSize(a.tensorPtr, axis_scalar.get(), result.aclDtype, result.tensorPtr,
                                              &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCumprodGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnCumprod(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCumprod");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumprod completed");
    return result;
}

NPUArray Cumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnCumsum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnCumsumGetWorkspaceSize(a.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnCumsumGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnCumsum(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnCumsum");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumsum completed");
    return result;
}

NPUArray Nancumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    auto axis_scalar = AclScalarGuard(aclCreateScalar(&axis, ACL_INT64));
    float scalar = 1.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnCumprod start: input_shape={}, aclDtype={}, axis={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis);
    auto error2 = aclnnCumprodGetWorkspaceSize(temp.tensorPtr, axis_scalar.get(), result.aclDtype, result.tensorPtr,
                                               &workspaceSize2, &executor2);
    ACLNN_CHECK(error2, "aclnnCumprodGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnCumprod(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnCumprod");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumprod completed");
    return result;
}

NPUArray Nancumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
    LOG_DEBUG("aclnnNanToNum start: input_shape={}, tensorSize={}, aclDtype={}, axis={}", detail::FormatShape(a.shape),
              a.tensorSize, AclDtypeName(a.aclDtype), axis);
    py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
    auto shape = a.shape;
    float scalar = 0.0;
    auto temp = NPUArray(shape, a.aclDtype);
    auto result = NPUArray(shape, outDtype);
    uint64_t workspaceSize1 = 0;
    aclOpExecutor* executor1;
    auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, std::numeric_limits<float>::infinity(),
                                                -std::numeric_limits<float>::infinity(), temp.tensorPtr,
                                                &workspaceSize1, &executor1);
    ACLNN_CHECK(error1, "aclnnNanToNumGetWorkspaceSize");
    AclWorkspace workspace1(workspaceSize1);
    error1 = aclnnNanToNum(workspace1.get(), workspaceSize1, executor1, nullptr);
    ACLNN_CHECK(error1, "aclnnNanToNum");
    error1 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error1, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnNanToNum completed");

    uint64_t workspaceSize2 = 0;
    aclOpExecutor* executor2;
    LOG_DEBUG("aclnnCumsum start: input_shape={}, aclDtype={}, axis={}", detail::FormatShape(temp.shape),
              AclDtypeName(temp.aclDtype), axis);
    auto error2 = aclnnCumsumGetWorkspaceSize(temp.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize2,
                                              &executor2);
    ACLNN_CHECK(error2, "aclnnCumsumGetWorkspaceSize");
    AclWorkspace workspace2(workspaceSize2);
    error2 = aclnnCumsum(workspace2.get(), workspaceSize2, executor2, nullptr);
    ACLNN_CHECK(error2, "aclnnCumsum");
    error2 = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error2, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCumsum completed");
    return result;
}

NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axis) {
    LOG_DEBUG("aclnnLinalgCross start: a_shape={}, b_shape={}, axis={}, aclDtype={}", detail::FormatShape(a.shape),
              detail::FormatShape(b.shape), axis, AclDtypeName(a.aclDtype));
    auto broadcast = GetBroadcastShape(a, b);
    auto result = NPUArray(broadcast, a.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    auto error =
        aclnnLinalgCrossGetWorkspaceSize(a.tensorPtr, b.tensorPtr, axis, result.tensorPtr, &workspaceSize, &executor);
    ACLNN_CHECK(error, "aclnnLinalgCrossGetWorkspaceSize");
    AclWorkspace workspace(workspaceSize);
    error = aclnnLinalgCross(workspace.get(), workspaceSize, executor, nullptr);
    ACLNN_CHECK(error, "aclnnLinalgCross");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnLinalgCross completed");
    return result;
}
} // namespace asnumpy
