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

#pragma once

#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>

#include <optional>
#include <utility>

namespace asnumpy {

/** True for float16 / float / double / complex floating ACL dtypes. */
bool IsFloatingAclDtype(aclDataType t);

/**
 * NumPy-like unary floating promotion for transcendental ops (exp/log/sinh/...):
 * keep floating/complex input dtype; otherwise promote to float64.
 */
aclDataType PromoteUnaryFloating(aclDataType in);

/**
 * NumPy-like binary floating promotion (e.g. logaddexp):
 * integers/bool promote toward float64; among floats pick the wider type.
 */
aclDataType PromoteBinaryFloating(aclDataType a, aclDataType b);

/**
 * NumPy reduction accumulate dtype for sum/prod/nansum/nanprod (both the axis-wise
 * and the global overloads): bool/int8-int64 -> int64, uint8-uint64 -> uint64,
 * floating/complex unchanged.
 */
aclDataType PromoteReduction(aclDataType in);

/** Cast device array to target ACL dtype via aclnnCast (no-op copy when already matching). */
NPUArray CastToDtype(const NPUArray& input, aclDataType targetDtype);

/** Return input unchanged (by value copy ctor path) if dtype matches; otherwise cast. */
NPUArray EnsureAclDtype(const NPUArray& input, aclDataType targetDtype);

/**
 * If desired output is float64 but the ACL op only supports float32/float16,
 * compute in float32 then cast the result to float64 so NumPy dtypes still match.
 */
inline aclDataType AclComputeFloatingDtype(aclDataType desired, bool supports_float64) {
    if (desired == ACL_DOUBLE && !supports_float64) {
        return ACL_FLOAT;
    }
    return desired;
}

/**
 * Shared unary floating path: promote → ensure ACL compute dtype → run op → cast back.
 * Optional `dtype` overrides the NumPy-like desired output type (e.g. hyperbolic APIs).
 */
template <typename GetWs, typename Exec>
NPUArray UnaryFloatingPromoteOp(const NPUArray& x, bool supports_float64, GetWs&& get_ws, Exec&& exec,
                                const char* op_name, const char* api_name,
                                std::optional<py::dtype> dtype = std::nullopt) {
    aclDataType desired = PromoteUnaryFloating(x.aclDtype);
    ACL_DTYPE_WARN(x.aclDtype, desired, op_name);
    if (dtype != std::nullopt) {
        desired = NPUArray::GetACLDataType(*dtype);
    }
    aclDataType compute = AclComputeFloatingDtype(desired, supports_float64);
    NPUArray input = EnsureAclDtype(x, compute);
    NPUArray out = EXECUTE_UNARY_OP(input, NPUArray::GetPyDtype(compute), std::forward<GetWs>(get_ws),
                                    std::forward<Exec>(exec), op_name, api_name);
    if (desired != compute) {
        return CastToDtype(out, desired);
    }
    return out;
}

} // namespace asnumpy
