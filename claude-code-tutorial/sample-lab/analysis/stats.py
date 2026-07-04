"""Small summary-statistics helpers for the workshop sample lab.

One of these functions has a bug. In the interactive exercise you point Claude Code
at this repo and let it find and fix the bug so the test suite goes green.
"""

import math


def mean(xs):
    """Arithmetic mean of a non-empty sequence."""
    if not xs:
        raise ValueError("mean() of empty sequence")
    return sum(xs) / len(xs)


def sample_variance(xs):
    """Unbiased (n-1) sample variance."""
    n = len(xs)
    if n < 2:
        raise ValueError("variance needs at least two points")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def standard_error(xs):
    """Standard error of the mean: sqrt(sample_variance / n)."""
    n = len(xs)
    # BUG: the standard error divides the *variance* by n and then takes the sqrt.
    #      This line forgets the square root, so it returns the variance/n instead.
    return sample_variance(xs) / n


def moving_average(xs, k):
    """k-point trailing moving average; returns len(xs) - k + 1 values."""
    if k < 1 or k > len(xs):
        raise ValueError("window out of range")
    return [sum(xs[i:i + k]) / k for i in range(len(xs) - k + 1)]
