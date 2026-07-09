"""Water-quality analysis helpers for the LakeWatch monitoring project.

These are small, pure functions used to summarise sensor readings.

One function in this module has a planted bug. In Task 3 you point Claude Code at
the repository and let it find and fix the bug so the test suite goes green. Every
other function here is already correct.
"""
from __future__ import annotations

import math


def c_to_f(celsius):
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32


def f_to_c(fahrenheit):
    """Convert a Fahrenheit temperature to Celsius."""
    return (fahrenheit - 32) * 5 / 9


def mean(values):
    """Arithmetic mean of a non-empty sequence of numbers."""
    if not values:
        raise ValueError("mean() of an empty sequence")
    return sum(values) / len(values)


def sample_std(values):
    """Unbiased sample standard deviation (uses the n - 1 denominator)."""
    n = len(values)
    if n < 2:
        raise ValueError("standard deviation needs at least two values")
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def rolling_mean(values, window):
    """Trailing moving average over `window` consecutive points.

    For a series of length n and a window of size k, the window can sit in
    n - k + 1 positions, so this returns n - k + 1 averages (e.g. 5 points with
    a window of 3 gives 3 averages).
    """
    k = window
    if k < 1 or k > len(values):
        raise ValueError("window out of range")
    # BUG: this range stops one position short, so the last window is dropped and
    #      the result has n - k values instead of the n - k + 1 promised above.
    return [sum(values[i:i + k]) / k for i in range(len(values) - k)]


def flag_outliers(values, z=3.0):
    """Return the indices of points more than `z` sample standard deviations
    from the mean. Useful for spotting a mis-recorded reading."""
    if len(values) < 2:
        return []
    m = mean(values)
    s = sample_std(values)
    if s == 0:
        return []
    return [i for i, x in enumerate(values) if abs(x - m) / s > z]
