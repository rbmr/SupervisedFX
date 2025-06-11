import math
import numpy as np
from hypothesis import given, strategies as st, assume
from common.envs.dp import (  # adjust import as needed
    get_exposure_levels,
    get_exposure_idx,
    get_low_high_exposure_idx,
    get_exposure_val,
)

n_values = st.integers(min_value = 1, max_value = 100)
x_values = st.floats(min_value = -1.0, max_value = 1.0, allow_nan = False, allow_infinity = False)

@given(n=n_values)
def test_get_exposure_levels(n):
    levels = get_exposure_levels(n)
    assert len(levels) == 2 * n + 1
    assert levels[0] == -1.0
    assert levels[-1] == 1.0
    assert np.all(np.isclose(np.diff(levels), (1 / n)))

@given(x=x_values, n=n_values)
def test_get_exposure_idx_in_bounds(x, n):
    idx = get_exposure_idx(x, n)
    assert 0 <= idx <= 2 * n

@given(x=x_values, n=n_values)
def test_get_exposure_idx_closest_match(x, n):
    levels = get_exposure_levels(n)
    idx = get_exposure_idx(x, n)
    closest_val = levels[idx]
    assert all(abs(x - closest_val) <= abs(x - l) for l in levels)

@given(x=x_values, n=n_values)
def test_get_low_high_exposure_idx_bounds(x, n):
    low, high = get_low_high_exposure_idx(x, n)
    assert 0 <= low <= 2 * n
    assert 0 <= high <= 2 * n
    assert high >= low

@given(x=x_values, n=n_values)
def test_low_high_enclose_idx(x, n):
    idx = (x + 1) * n
    low, high = get_low_high_exposure_idx(x, n)
    assert math.floor(idx) == low
    assert math.ceil(idx) == high

@given(n=n_values)
def test_get_exposure_val_matches_levels(n):
    levels = get_exposure_levels(n)
    for i in range(2 * n + 1):
        val = get_exposure_val(i, n)
        assert math.isclose(val, levels[i], rel_tol=1e-12, abs_tol=1e-12)

@given(i=st.integers(min_value=0, max_value=200), n=n_values)
def test_get_exposure_val_bounds(i, n):
    assume(i <= 2 * n)
    val = get_exposure_val(i, n)
    assert -1.0 <= val <= 1.0