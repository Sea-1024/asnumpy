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

#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/dtype_promotion.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <aclnnop/aclnn_cast.h>

namespace asnumpy {

bool IsFloatingAclDtype(aclDataType t) {
    return t == ACL_FLOAT16 || t == ACL_FLOAT || t == ACL_DOUBLE || t == ACL_COMPLEX64 || t == ACL_COMPLEX128 ||
           t == ACL_BF16;
}

static int FloatingRank(aclDataType t) {
    switch (t) {
    case ACL_FLOAT16:
    case ACL_BF16:
        return 1;
    case ACL_FLOAT:
    case ACL_COMPLEX64:
        return 2;
    case ACL_DOUBLE:
    case ACL_COMPLEX128:
        return 3;
    default:
        return 0;
    }
}

aclDataType PromoteUnaryFloating(aclDataType in) {
    if (IsFloatingAclDtype(in)) {
        return in;
    }
    // Match NumPy ufunc integer→float defaults:
    // bool/int8/uint8/int16/uint16 → float32; wider integers → float64.
    switch (in) {
    case ACL_BOOL:
    case ACL_INT8:
    case ACL_UINT8:
    case ACL_INT16:
    case ACL_UINT16:
        return ACL_FLOAT;
    default:
        return ACL_DOUBLE;
    }
}

aclDataType PromoteBinaryFloating(aclDataType a, aclDataType b) {
    aclDataType left = IsFloatingAclDtype(a) ? a : PromoteUnaryFloating(a);
    aclDataType right = IsFloatingAclDtype(b) ? b : PromoteUnaryFloating(b);
    return FloatingRank(left) >= FloatingRank(right) ? left : right;
}

NPUArray CastToDtype(const NPUArray& input, aclDataType targetDtype) {
    LOG_DEBUG("aclnnCast start: input_shape={}, aclDtype={}, targetDtype={}", detail::FormatShape(input.shape),
              AclDtypeName(input.aclDtype), AclDtypeName(targetDtype));
    auto result = NPUArray(input.shape, targetDtype);
    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto err = aclnnCastGetWorkspaceSize(input.tensorPtr, targetDtype, result.tensorPtr, &wsSize, &exec);
    ACLNN_CHECK(err, "aclnnCastGetWorkspaceSize");
    AclWorkspace ws(wsSize);
    err = aclnnCast(ws.get(), wsSize, exec, nullptr);
    ACLNN_CHECK(err, "aclnnCast");
    err = aclrtSynchronizeDevice();
    ACL_RT_CHECK(err, "aclrtSynchronizeDevice");
    LOG_INFO("aclnnCast completed");
    return result;
}

NPUArray EnsureAclDtype(const NPUArray& input, aclDataType targetDtype) {
    if (input.aclDtype == targetDtype) {
        return input;
    }
    return CastToDtype(input, targetDtype);
}

} // namespace asnumpy
