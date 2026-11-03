# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Contract tests for NPUArray's shared-storage model.

These lock down behavior that is invisible when everything works: views share
storage, so they must outlive their source, must not copy, and must stay
readable after the array they came from is gone. A regression in any of them
would be a use-after-free or a silent wrong answer, not a clean failure.

The tests are deliberately not parameterized over dtypes -- the property under
test is about storage, and dtype is already covered by the per-operation suites.
"""

import gc

import numpy
import pytest

import asnumpy as ap


def _np(data, dtype=numpy.float32):
    return numpy.array(data, dtype=dtype)


def _dev(data, dtype=numpy.float32):
    return ap.ndarray.from_numpy(_np(data, dtype))


def test_view_outlives_its_source():
    """A view must keep its source's allocation alive.

    This is the whole point of the shared DeviceBuffer. Without refcounting the
    allocation would be returned to the pool while the view still references it,
    and the pool would hand the same block to the next allocation -- corruption
    that shows up as wrong numbers, not as a crash.
    """

    def make_view():
        source = _dev([[1, 2, 3], [4, 5, 6]])
        return source.reshape((3, 2))

    view = make_view()
    gc.collect()  # force the source's destructor to have run
    numpy.testing.assert_array_equal(
        view.to_numpy(), _np([[1, 2, 3], [4, 5, 6]]).reshape(3, 2)
    )


def test_view_outlives_source_across_pool_churn():
    """The pool must not recycle a block that a live view still references."""
    source = _dev(numpy.arange(16).reshape(4, 4))
    view = source.reshape((16,))
    del source
    gc.collect()

    # Force the same-sized blocks to be allocated and freed repeatedly; if the
    # view's block were reusable, one of these would overwrite it.
    for i in range(20):
        scratch = _dev(numpy.full((16,), float(i)))
        del scratch

    numpy.testing.assert_array_equal(view.to_numpy(), numpy.arange(16, dtype=numpy.float32))


def test_chained_views():
    source = _dev(numpy.arange(24).reshape(2, 3, 4))
    chained = source.reshape((6, 4)).reshape((4, 6))
    numpy.testing.assert_array_equal(chained.to_numpy(), numpy.arange(24, dtype=numpy.float32).reshape(4, 6))


@pytest.mark.parametrize("index", [(0,), (1,), (0, 1), (1, 2), (-1,), (-1, -1)])
def test_getitem_view_matches_numpy(index):
    np_a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    a = ap.ndarray.from_numpy(np_a)
    numpy.testing.assert_array_equal(a[index].to_numpy(), np_a[index])


def test_getitem_view_survives_source():
    source = _dev(numpy.arange(12).reshape(3, 4))
    row = source[1]
    del source
    gc.collect()
    numpy.testing.assert_array_equal(row.to_numpy(), numpy.arange(12, dtype=numpy.float32).reshape(3, 4)[1])


def test_flatten_after_transpose_reads_permuted_order():
    """Flatten must read the materialized permutation, not the original buffer.

    Transpose is the one non-view path; this pins that the combination still
    produces contiguous, correctly-ordered output.
    """
    np_a = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    got = ap.ndarray.from_numpy(np_a).transpose().flatten().to_numpy()
    numpy.testing.assert_array_equal(got, np_a.T.flatten())


def test_many_views_do_not_multiply_memory():
    """200 views of one array must all remain correct.

    Cheap proxy for "views do not allocate": if each view copied, the source
    would still be fine, so correctness alone does not prove it -- but a view
    that copied and then aliased would break here.
    """
    source = _dev(numpy.arange(1000 * 1000).reshape(1000, 1000))
    views = [source.reshape((1000, 1000)) for _ in range(200)]
    expected = numpy.arange(1000 * 1000, dtype=numpy.float32).reshape(1000, 1000)
    numpy.testing.assert_array_equal(views[-1].to_numpy(), expected)
    numpy.testing.assert_array_equal(source.to_numpy(), expected)


def test_zero_size_array_roundtrip():
    """A zero-element array has no allocation; the buffer is null."""
    z = ap.ndarray.from_numpy(numpy.zeros((0,), dtype=numpy.float32))
    assert z.to_numpy().shape == (0,)
    numpy.testing.assert_array_equal(z.reshape((0, 3)).to_numpy(), numpy.zeros((0, 3), dtype=numpy.float32))


def test_copy_constructor_is_independent():
    """ndarray(a) must deep-copy, unlike a view.

    The public constructor keeps copy semantics so callers can rely on
    `ndarray(existing)` not aliasing. Views use a separate private path.
    """
    source = _dev([1, 2, 3])
    copied = ap.ndarray(source)
    del source
    gc.collect()
    numpy.testing.assert_array_equal(copied.to_numpy(), _np([1, 2, 3]))


def test_reshape_rejects_element_count_mismatch():
    a = _dev([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        a.reshape((4, 2))


def test_reshape_infers_single_minus_one():
    a = _dev(numpy.arange(12).reshape(3, 4))
    assert a.reshape((-1, 2)).shape == (6, 2)
    numpy.testing.assert_array_equal(
        a.reshape((-1, 2)).to_numpy(), numpy.arange(12, dtype=numpy.float32).reshape(6, 2)
    )


def test_reshape_rejects_two_minus_ones():
    a = _dev(numpy.arange(12).reshape(3, 4))
    with pytest.raises(ValueError):
        a.reshape((-1, -1))


def test_memory_pool_serves_repeat_allocations_from_cache():
    """The pool must actually hit, not just exist.

    Timing alone cannot distinguish a working pool from one that never matches
    a size; the counters can. A run of same-sized operations must be served
    almost entirely from cache.
    """
    from asnumpy._core import memory_pool

    a = _dev(numpy.ones((1000, 1000)))
    before = memory_pool.stats()
    for _ in range(50):
        ap.multiply(a, a)
    after = memory_pool.stats()

    allocs = (after["hits"] + after["misses"]) - (before["hits"] + before["misses"])
    hits = after["hits"] - before["hits"]
    assert allocs >= 50, f"expected at least 50 allocations, saw {allocs}"
    assert hits / allocs > 0.8, f"pool hit rate too low: {hits}/{allocs}"


def test_memory_pool_clear_empties_the_cache():
    from asnumpy._core import memory_pool

    _dev(numpy.ones((500, 500)))  # populate
    assert memory_pool.stats()["cached_bytes"] > 0

    memory_pool.clear()
    assert memory_pool.stats()["cached_bytes"] == 0
