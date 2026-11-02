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

#include "../utils/npu_array.hpp"
#include <acl/acl.h>
#include <vector>

namespace asnumpy {

/**
 * @brief Join a sequence of arrays along an existing axis.
 *
 * All inputs must share a dtype and must agree on every dimension except
 * `axis`, matching numpy.concatenate.
 *
 * @param arrays Input arrays; must be non-empty.
 * @param axis Axis to join along. Negative values wrap.
 * @return NPUArray Newly allocated joined array.
 * @throws std::invalid_argument On empty input, dtype mismatch, rank mismatch,
 *         out-of-range axis, or mismatched non-axis dimensions.
 */
NPUArray Concatenate(const std::vector<NPUArray>& arrays, int64_t axis);

/**
 * @brief Join a sequence of arrays along a new axis.
 *
 * All inputs must share a dtype and an identical shape, matching numpy.stack.
 * The result has one more dimension than the inputs.
 *
 * @param arrays Input arrays; must be non-empty.
 * @param axis Position of the new axis in the result. Negative values wrap.
 * @return NPUArray Newly allocated stacked array.
 * @throws std::invalid_argument On empty input, dtype mismatch, shape mismatch,
 *         or out-of-range axis.
 */
NPUArray Stack(const std::vector<NPUArray>& arrays, int64_t axis);

/**
 * @brief Remove axes of length one.
 *
 * Mirrors numpy.squeeze. When `axes` is empty every length-1 axis is removed;
 * otherwise only the named axes are, and each must actually have length 1.
 *
 * @param array Input array.
 * @param axes Axes to remove, or empty for all length-1 axes. Negative wrap.
 * @return NPUArray Newly allocated array with the axes removed.
 * @throws std::invalid_argument If an axis is out of range, repeated, or does
 *         not have length 1.
 */
NPUArray Squeeze(const NPUArray& array, const std::vector<int64_t>& axes);

/**
 * @brief Insert a new axis of length one.
 *
 * Mirrors numpy.expand_dims. The new axis may be placed one past the last
 * existing axis, so valid positions are [-(ndim+1), ndim].
 *
 * @param array Input array.
 * @param axis Position of the new axis. Negative values wrap.
 * @return NPUArray Newly allocated array with one more dimension.
 * @throws std::invalid_argument If `axis` is out of range.
 */
NPUArray ExpandDims(const NPUArray& array, int64_t axis);

} // namespace asnumpy
