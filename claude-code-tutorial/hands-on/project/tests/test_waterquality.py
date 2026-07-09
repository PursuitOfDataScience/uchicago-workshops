"""Ground-truth tests for waterquality.py.

Do NOT edit these tests — they define correct behaviour. One function in
``src/waterquality.py`` has a planted bug, so the rolling-mean tests start red.

Run from the project root:  python -m pytest -q
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from waterquality import (  # noqa: E402
    c_to_f,
    f_to_c,
    mean,
    sample_std,
    rolling_mean,
    flag_outliers,
)


def test_c_to_f():
    assert math.isclose(c_to_f(100), 212.0)
    assert math.isclose(c_to_f(0), 32.0)


def test_f_to_c():
    assert math.isclose(f_to_c(212), 100.0)
    assert math.isclose(f_to_c(32), 0.0)


def test_temp_roundtrip():
    # converting C -> F -> C returns the original value
    assert math.isclose(f_to_c(c_to_f(21.0)), 21.0)


def test_mean():
    assert mean([2, 4, 6]) == 4.0


def test_sample_std():
    # std of [2, 4, 6] with the (n - 1) denominator is exactly 2.0
    assert math.isclose(sample_std([2, 4, 6]), 2.0)


def test_rolling_mean_length():
    # 5 points, window 3 -> 3 averages (positions 0-2, 1-3, 2-4)
    assert len(rolling_mean([1, 2, 3, 4, 5], 3)) == 3


def test_rolling_mean_values():
    assert rolling_mean([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]


def test_flag_outliers():
    # nine readings near 10, one clear outlier at 20
    data = [10, 10, 10, 10, 10, 10, 10, 10, 10, 20]
    assert flag_outliers(data, z=2.0) == [9]
