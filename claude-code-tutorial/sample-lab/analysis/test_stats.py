"""Ground-truth tests for stats.py. Do NOT edit these — they define correct behaviour.

Run:  python -m pytest -q
One test starts red because of a planted bug in stats.py.
"""

import math

from stats import mean, sample_variance, standard_error, moving_average


def test_mean():
    assert mean([2, 4, 6]) == 4.0


def test_sample_variance():
    # variance of [2,4,6] with (n-1) denominator = 4.0
    assert sample_variance([2, 4, 6]) == 4.0


def test_standard_error():
    # se = sqrt(variance / n) = sqrt(4.0 / 3) ≈ 1.1547
    assert math.isclose(standard_error([2, 4, 6]), math.sqrt(4.0 / 3), rel_tol=1e-9)


def test_moving_average():
    assert moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]
