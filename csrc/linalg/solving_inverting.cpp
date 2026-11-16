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

#include <asnumpy/linalg/norms.hpp>
#include <asnumpy/linalg/solving_inverting.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_inverse.h>

#include <cmath>
#include <fmt/core.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>

using namespace asnumpy;

namespace {

bool IsNumericallySingular(const NPUArray& sign, const NPUArray& logdet) {
    NPUArray sign_f64 = EnsureAclDtype(sign, ACL_DOUBLE);
    NPUArray logdet_f64 = EnsureAclDtype(logdet, ACL_DOUBLE);
    py::array sign_host = sign_f64.ToNumpy();
    py::array logdet_host = logdet_f64.ToNumpy();
    py::buffer_info sign_info = sign_host.request();
    py::buffer_info logdet_info = logdet_host.request();
    const auto* sign_data = static_cast<const double*>(sign_info.ptr);
    const auto* logdet_data = static_cast<const double*>(logdet_info.ptr);
    const auto n = static_cast<ssize_t>(sign_info.size);
    // Threshold: exp(logabsdet) below ~1e-10 treats the matrix as singular for float32/64 cases.
    constexpr double kLogAbsDetSingular = -23.0; // log(1e-10) ≈ -23
    for (ssize_t i = 0; i < n; ++i) {
        if (sign_data[i] == 0.0) {
            return true;
        }
        if (!std::isfinite(logdet_data[i]) && logdet_data[i] < 0) {
            return true;
        }
        if (logdet_data[i] < kLogAbsDetSingular) {
            return true;
        }
    }
    return false;
}

} // namespace

NPUArray Linalg_Inv(const NPUArray& a) {
    // NumPy raises LinAlgError on singular matrices. slogdet sign may stay non-zero
    // under float noise, so also treat tiny |det| (via logabsdet) as singular.
    auto [sign, logdet] = Linalg_Slogdet(a);
    if (IsNumericallySingular(sign, logdet)) {
        throw std::runtime_error("Singular matrix: inv() failed because the matrix is singular (LinAlgError)");
    }

    return EXECUTE_UNARY_OP(
        a, a.dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnInverseGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnInverse(workspace, workspaceSize, executor, nullptr);
        },
        "Linalg_Inv", "aclnnInverse");
}
