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

#include <asnumpy/math/exponents_and_logarithms.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_exp2.h>
#include <aclnnop/aclnn_expm1.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_log10.h>
#include <aclnnop/aclnn_log1p.h>
#include <aclnnop/aclnn_log2.h>
#include <aclnnop/aclnn_logaddexp.h>
#include <aclnnop/aclnn_logaddexp2.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

namespace {

template <typename GetWs, typename Exec>
NPUArray BinaryFloatingOp(const NPUArray& x1, const NPUArray& x2, bool supports_float64, GetWs&& get_ws, Exec&& exec,
                          const char* op_name, const char* api_name) {
    aclDataType desired = PromoteBinaryFloating(x1.aclDtype, x2.aclDtype);
    ACL_DTYPE_WARN(x1.aclDtype, desired, op_name);
    ACL_DTYPE_WARN(x2.aclDtype, desired, op_name);
    aclDataType compute = AclComputeFloatingDtype(desired, supports_float64);
    NPUArray in1 = EnsureAclDtype(x1, compute);
    NPUArray in2 = EnsureAclDtype(x2, compute);
    py::dtype dtype = NPUArray::GetPyDtype(compute);
    NPUArray out =
        EXECUTE_BINARY_OP(in1, in2, dtype, std::forward<GetWs>(get_ws), std::forward<Exec>(exec), op_name, api_name);
    if (desired != compute) {
        return CastToDtype(out, desired);
    }
    return out;
}

} // namespace

NPUArray Exp(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnExpGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnExp(workspace, workspaceSize, executor, nullptr);
        },
        "Exp", "aclnnExp");
}

NPUArray Expm1(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnExpm1GetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnExpm1(workspace, workspaceSize, executor, nullptr);
        },
        "Expm1", "aclnnExpm1");
}

NPUArray Exp2(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnExp2GetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnExp2(workspace, workspaceSize, executor, nullptr);
        },
        "Exp2", "aclnnExp2");
}

NPUArray Log(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLog(workspace, workspaceSize, executor, nullptr);
        },
        "Log", "aclnnLog");
}

NPUArray Log10(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLog10GetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLog10(workspace, workspaceSize, executor, nullptr);
        },
        "Log10", "aclnnLog10");
}

NPUArray Log2(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLog2GetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLog2(workspace, workspaceSize, executor, nullptr);
        },
        "Log2", "aclnnLog2");
}

NPUArray Log1p(const NPUArray& x) {
    return UnaryFloatingPromoteOp(
        x, /*supports_float64=*/true,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLog1pGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLog1p(workspace, workspaceSize, executor, nullptr);
        },
        "Log1p", "aclnnLog1p");
}

NPUArray Logaddexp(const NPUArray& x1, const NPUArray& x2) {
    // aclnnLogAddExp does not support float64 outputs.
    return BinaryFloatingOp(
        x1, x2, /*supports_float64=*/false,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogAddExpGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogAddExp(workspace, workspaceSize, executor, nullptr);
        },
        "Logaddexp", "aclnnLogAddExp");
}

NPUArray Logaddexp2(const NPUArray& x1, const NPUArray& x2) {
    // aclnnLogAddExp2 does not support float64 outputs.
    return BinaryFloatingOp(
        x1, x2, /*supports_float64=*/false,
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnLogAddExp2GetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnLogAddExp2(workspace, workspaceSize, executor, nullptr);
        },
        "Logaddexp2", "aclnnLogAddExp2");
}

} // namespace asnumpy
