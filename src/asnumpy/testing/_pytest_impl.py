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

"""pytest integration implementation

Handles integration between pytest and the Asnumpy test framework:
- parameterize() - implementation of parameterized tests
- _TestingParameterizeMixin - mixin class for parameterized tests
- is_available() - check whether pytest is available
"""

__all__ = [
    "is_available",
    "parameterize",
    "fixture",
    "skip",
    "skipif",
    "xfail",
    "_TestingParameterizeMixin",
]

import functools
import importlib.util


def is_available():
    """Check whether pytest is available.

    Returns:
        bool: True if pytest is installed and importable.
    """
    return importlib.util.find_spec("pytest") is not None


class _TestingParameterizeMixin:
    """Mixin class for parameterized tests.

    Inherit from this class to support pytest-style parameterized tests.
    """

    @classmethod
    def setup_class(cls):
        """Setup executed before the test class starts."""
        pass

    @classmethod
    def teardown_class(cls):
        """Cleanup executed after the test class finishes."""
        pass


def parameterize(*args, **kwargs):
    """Parameterized test decorator.

    Provides functionality similar to pytest.mark.parametrize,
    integrated with the Asnumpy test framework.

    Args:
        *args: Parameter names and values.
        **kwargs: Additional options.

    Returns:
        Decorator function.

    Examples:
        @parameterize('dtype', [numpy.float32, numpy.float64])
        def test_func(self, dtype):
            ...
    """
    if not is_available():
        # Fall back to a simple loop implementation when pytest is unavailable
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *func_args, **func_kwargs):
                if len(args) >= 2:
                    param_name = args[0]
                    param_values = args[1]
                    for value in param_values:
                        func_kwargs[param_name] = value
                        func(self, *func_args, **func_kwargs)
                else:
                    func(self, *func_args, **func_kwargs)

            return wrapper

        return decorator

    # Use pytest.mark.parametrize when pytest is available
    import pytest

    return pytest.mark.parametrize(*args, **kwargs)


def fixture(*args, **kwargs):
    """Fixture decorator.

    Provides functionality similar to pytest.fixture.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Decorator function or fixture object.
    """
    if not is_available():
        # Return a simple passthrough decorator when pytest is unavailable
        if len(args) == 1 and callable(args[0]):
            # Used directly as @fixture
            return args[0]
        else:
            # Used as @fixture(...)
            def decorator(func):
                return func

            return decorator

    # Use pytest.fixture when pytest is available
    import pytest

    return pytest.fixture(*args, **kwargs)


def skip(reason):
    """Skip test decorator.

    Args:
        reason: Reason for skipping the test.

    Returns:
        Decorator function.
    """
    if not is_available():
        import unittest

        return unittest.skip(reason)

    import pytest

    return pytest.mark.skip(reason=reason)


def skipif(condition, reason):
    """Conditional skip test decorator.

    Args:
        condition: Condition under which the test is skipped.
        reason: Reason for skipping the test.

    Returns:
        Decorator function.
    """
    if not is_available():
        import unittest

        return unittest.skipIf(condition, reason)

    import pytest

    return pytest.mark.skipif(condition, reason=reason)


def xfail(reason="", strict=False):
    """Expected failure test decorator.

    Args:
        reason: Reason for the expected failure.
        strict: Whether to use strict mode.

    Returns:
        Decorator function.
    """
    if not is_available():
        import unittest

        return unittest.expectedFailure

    import pytest

    return pytest.mark.xfail(reason=reason, strict=strict)
