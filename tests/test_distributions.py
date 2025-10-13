"""
Tests for statistical distributions used in parameter sampling.
"""

import pytest
import random
import numpy as np
from src.distributions import (
    sample_parameter,
    sample_normal,
    sample_exponential,
    DistributionType
)


def test_sample_uniform_basic():
    """Tests that uniform distribution samples within bounds."""
    random.seed(42)
    for _ in range(100):
        value = sample_parameter(0.0, 10.0, "uniform")
        assert 0.0 <= value <= 10.0, f"Value {value} out of bounds"


def test_sample_uniform_distribution_shape():
    """Tests that uniform distribution is actually uniform using statistical test."""
    random.seed(42)
    samples = [sample_parameter(0.0, 100.0, "uniform") for _ in range(10000)]

    # Divide range into 10 bins
    hist, _ = np.histogram(samples, bins=10, range=(0, 100))

    # Each bin should have ~1000 samples (10000 / 10)
    # Allow 20% deviation for randomness
    for count in hist:
        assert 800 <= count <= 1200, f"Bin count {count} suggests non-uniform distribution"


def test_sample_normal_basic():
    """Tests that normal distribution samples within bounds."""
    random.seed(42)
    for _ in range(100):
        value = sample_normal(-10.0, 10.0)
        assert -10.0 <= value <= 10.0, f"Value {value} out of bounds"


def test_sample_normal_centered():
    """Tests that normal distribution is centered at midpoint."""
    random.seed(42)
    samples = [sample_normal(0.0, 100.0) for _ in range(10000)]

    mean = np.mean(samples)
    # Mean should be close to 50 (midpoint)
    assert 48.0 <= mean <= 52.0, f"Mean {mean} not centered at midpoint 50"


def test_sample_normal_68_95_99_rule():
    """Tests that normal distribution follows 68-95-99.7 rule (approximately)."""
    random.seed(42)
    min_val, max_val = 0.0, 60.0
    mean = 30.0
    sigma = 10.0  # (max - min) / 6 = 60 / 6 = 10

    samples = [sample_normal(min_val, max_val) for _ in range(10000)]

    # Count samples within 1, 2, 3 sigma
    within_1_sigma = sum(1 for s in samples if abs(s - mean) <= sigma)
    within_2_sigma = sum(1 for s in samples if abs(s - mean) <= 2 * sigma)
    within_3_sigma = sum(1 for s in samples if abs(s - mean) <= 3 * sigma)

    # Expected: 68%, 95%, 99.7% (with some tolerance for clipping at bounds)
    assert 6500 <= within_1_sigma <= 7000, f"1-sigma: {within_1_sigma}/10000"
    assert 9300 <= within_2_sigma <= 9700, f"2-sigma: {within_2_sigma}/10000"
    assert 9900 <= within_3_sigma <= 10000, f"3-sigma: {within_3_sigma}/10000"


def test_sample_exponential_basic():
    """Tests that exponential distribution samples within bounds."""
    random.seed(42)
    for _ in range(100):
        value = sample_exponential(0.0, 10.0)
        assert 0.0 <= value <= 10.0, f"Value {value} out of bounds"


def test_sample_exponential_biased_toward_min():
    """Tests that exponential distribution is strongly biased toward minimum value."""
    random.seed(42)
    samples = [sample_exponential(0.0, 100.0) for _ in range(10000)]

    # Mode should be at min (0.0)
    # Mean should be around (max-min)/30 ≈ 3.3 for λ = 3/(max-min)
    mean = np.mean(samples)
    assert 2.0 <= mean <= 5.0, f"Mean {mean} not in expected range for exponential"

    # Most samples should be in lower 10% of range (~63%)
    in_lower_tenth = sum(1 for s in samples if s < 10.0)
    assert in_lower_tenth >= 6000, f"Only {in_lower_tenth}/10000 in lower tenth, expected >= 60%"


def test_sample_exponential_exponential_decay():
    """Tests that exponential distribution has strong bias toward minimum."""
    random.seed(42)
    samples = [sample_exponential(0.0, 100.0) for _ in range(10000)]

    # With strong exponential bias, most samples should be in first bin
    # Check that first bin >> last bin
    hist, _ = np.histogram(samples, bins=5, range=(0, 100))

    # First bin should have vastly more samples than last bin
    assert hist[0] > hist[-1] * 10, f"First bin ({hist[0]}) not >> last bin ({hist[-1]})"

    # At least 85% should be in first two bins (0-40)
    in_first_two = hist[0] + hist[1]
    assert in_first_two >= 8500, f"Only {in_first_two}/10000 in first 40% of range"


def test_sample_parameter_with_uniform():
    """Tests sample_parameter function with uniform distribution."""
    random.seed(42)
    value = sample_parameter(5.0, 15.0, "uniform")
    assert 5.0 <= value <= 15.0


def test_sample_parameter_with_normal():
    """Tests sample_parameter function with normal distribution."""
    random.seed(42)
    value = sample_parameter(-5.0, 5.0, "normal")
    assert -5.0 <= value <= 5.0


def test_sample_parameter_with_exponential():
    """Tests sample_parameter function with exponential distribution."""
    random.seed(42)
    value = sample_parameter(0.0, 20.0, "exponential")
    assert 0.0 <= value <= 20.0


def test_sample_parameter_invalid_distribution():
    """Tests that sample_parameter raises error for invalid distribution type."""
    with pytest.raises(ValueError, match="Unknown distribution"):
        sample_parameter(0.0, 10.0, "invalid_dist")


def test_sample_parameter_with_negative_range():
    """Tests sampling works correctly with negative ranges."""
    random.seed(42)

    # Normal distribution with negative range
    samples = [sample_parameter(-50.0, -10.0, "normal") for _ in range(1000)]
    mean = np.mean(samples)
    assert -35.0 <= mean <= -25.0, f"Mean {mean} not centered at midpoint -30"

    # All samples should be in bounds
    assert all(-50.0 <= s <= -10.0 for s in samples)


def test_sample_parameter_with_zero_crossing_range():
    """Tests sampling works correctly with ranges that cross zero."""
    random.seed(42)

    # Normal distribution crossing zero
    samples = [sample_parameter(-10.0, 10.0, "normal") for _ in range(1000)]
    mean = np.mean(samples)
    assert -2.0 <= mean <= 2.0, f"Mean {mean} not centered at 0"


def test_sample_parameter_edge_case_min_equals_max():
    """Tests sampling when min equals max (should always return that value)."""
    value = sample_parameter(5.0, 5.0, "uniform")
    assert value == 5.0

    value = sample_parameter(5.0, 5.0, "normal")
    assert value == 5.0

    value = sample_parameter(5.0, 5.0, "exponential")
    assert value == 5.0


def test_distributions_are_deterministic_with_seed():
    """Tests that distributions produce same values with same seed."""
    # Uniform
    random.seed(123)
    u1 = [sample_parameter(0.0, 100.0, "uniform") for _ in range(10)]
    random.seed(123)
    u2 = [sample_parameter(0.0, 100.0, "uniform") for _ in range(10)]
    assert u1 == u2, "Uniform not deterministic with seed"

    # Normal
    random.seed(456)
    n1 = [sample_parameter(0.0, 100.0, "normal") for _ in range(10)]
    random.seed(456)
    n2 = [sample_parameter(0.0, 100.0, "normal") for _ in range(10)]
    assert n1 == n2, "Normal not deterministic with seed"

    # Exponential
    random.seed(789)
    e1 = [sample_parameter(0.0, 100.0, "exponential") for _ in range(10)]
    random.seed(789)
    e2 = [sample_parameter(0.0, 100.0, "exponential") for _ in range(10)]
    assert e1 == e2, "Exponential not deterministic with seed"


def test_exponential_with_very_small_range():
    """Tests exponential works with very small ranges."""
    random.seed(42)
    samples = [sample_exponential(0.0, 1.0) for _ in range(1000)]

    # All should be in bounds
    assert all(0.0 <= s <= 1.0 for s in samples)

    # Mean should be much less than 0.5 (strongly biased toward 0)
    mean = np.mean(samples)
    assert mean < 0.15, f"Mean {mean} not strongly biased toward min"


def test_normal_with_very_large_range():
    """Tests normal distribution works with very large ranges."""
    random.seed(42)
    samples = [sample_normal(0.0, 10000.0) for _ in range(1000)]

    # All should be in bounds
    assert all(0.0 <= s <= 10000.0 for s in samples)

    # Mean should be around midpoint
    mean = np.mean(samples)
    assert 4500.0 <= mean <= 5500.0, f"Mean {mean} not centered"


# =============================================================================
# Tests for NumPy-optimized batch sampling
# =============================================================================

def test_sample_parameter_batch_uniform():
    """Tests batch sampling with uniform distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    # Sample batch
    samples = sample_parameter_batch(0.0, 10.0, "uniform", size=1000)

    # Check shape
    assert samples.shape == (1000,), f"Expected shape (1000,), got {samples.shape}"

    # Check bounds
    assert np.all((samples >= 0.0) & (samples <= 10.0)), "Some values out of bounds"

    # Check uniform distribution
    hist, _ = np.histogram(samples, bins=10, range=(0, 10))
    # Each bin should have ~100 samples
    for count in hist:
        assert 70 <= count <= 130, f"Bin count {count} suggests non-uniform distribution"


def test_sample_parameter_batch_normal():
    """Tests batch sampling with normal distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    # Sample batch
    samples = sample_parameter_batch(0.0, 100.0, "normal", size=10000)

    # Check shape
    assert samples.shape == (10000,), f"Expected shape (10000,), got {samples.shape}"

    # Check bounds
    assert np.all((samples >= 0.0) & (samples <= 100.0)), "Some values out of bounds"

    # Check centering
    mean = np.mean(samples)
    assert 48.0 <= mean <= 52.0, f"Mean {mean} not centered at midpoint 50"

    # Check 68-95-99.7 rule
    sigma = (100.0 - 0.0) / 6.0
    midpoint = 50.0
    within_1_sigma = np.sum(np.abs(samples - midpoint) <= sigma)
    within_2_sigma = np.sum(np.abs(samples - midpoint) <= 2 * sigma)

    assert 6500 <= within_1_sigma <= 7000, f"1-sigma: {within_1_sigma}/10000"
    assert 9300 <= within_2_sigma <= 9700, f"2-sigma: {within_2_sigma}/10000"


def test_sample_parameter_batch_exponential():
    """Tests batch sampling with exponential distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    # Sample batch
    samples = sample_parameter_batch(0.0, 100.0, "exponential", size=10000)

    # Check shape
    assert samples.shape == (10000,), f"Expected shape (10000,), got {samples.shape}"

    # Check bounds
    assert np.all((samples >= 0.0) & (samples <= 100.0)), "Some values out of bounds"

    # Check strong bias toward minimum
    mean = np.mean(samples)
    assert 2.0 <= mean <= 5.0, f"Mean {mean} not in expected range for exponential"

    # Check that ~63% are in lower 10%
    in_lower_tenth = np.sum(samples < 10.0)
    assert in_lower_tenth >= 6000, f"Only {in_lower_tenth}/10000 in lower tenth"

    # Check exponential decay
    hist, _ = np.histogram(samples, bins=5, range=(0, 100))
    assert hist[0] > hist[-1] * 10, f"First bin ({hist[0]}) not >> last bin ({hist[-1]})"


def test_sample_parameter_batch_deterministic():
    """Tests that batch sampling is deterministic with seed."""
    from src.distributions import sample_parameter_batch

    # Uniform
    np.random.seed(123)
    u1 = sample_parameter_batch(0.0, 100.0, "uniform", size=100)
    np.random.seed(123)
    u2 = sample_parameter_batch(0.0, 100.0, "uniform", size=100)
    assert np.allclose(u1, u2), "Uniform batch not deterministic"

    # Normal
    np.random.seed(456)
    n1 = sample_parameter_batch(0.0, 100.0, "normal", size=100)
    np.random.seed(456)
    n2 = sample_parameter_batch(0.0, 100.0, "normal", size=100)
    assert np.allclose(n1, n2), "Normal batch not deterministic"

    # Exponential
    np.random.seed(789)
    e1 = sample_parameter_batch(0.0, 100.0, "exponential", size=100)
    np.random.seed(789)
    e2 = sample_parameter_batch(0.0, 100.0, "exponential", size=100)
    assert np.allclose(e1, e2), "Exponential batch not deterministic"


def test_sample_parameter_batch_edge_cases():
    """Tests batch sampling edge cases."""
    from src.distributions import sample_parameter_batch

    # Min equals max
    samples = sample_parameter_batch(5.0, 5.0, "uniform", size=10)
    assert np.allclose(samples, 5.0), "Min==max should return constant array"

    # Single sample
    sample = sample_parameter_batch(0.0, 10.0, "uniform", size=1)
    assert sample.shape == (1,), "Single sample should have shape (1,)"
    assert 0.0 <= sample[0] <= 10.0, "Single sample out of bounds"

    # Negative range
    samples = sample_parameter_batch(-50.0, -10.0, "normal", size=100)
    assert np.all((samples >= -50.0) & (samples <= -10.0)), "Negative range out of bounds"

    # Zero-crossing range
    samples = sample_parameter_batch(-10.0, 10.0, "normal", size=100)
    assert np.all((samples >= -10.0) & (samples <= 10.0)), "Zero-crossing range out of bounds"


def test_sample_parameter_batch_invalid_distribution():
    """Tests that batch sampling raises error for invalid distribution."""
    from src.distributions import sample_parameter_batch

    with pytest.raises(ValueError, match="Unknown distribution"):
        sample_parameter_batch(0.0, 10.0, "invalid_dist", size=10)


# =============================================================================
# Tests for Beta distribution
# =============================================================================

def test_sample_beta_basic():
    """Tests that beta distribution samples within [0, 1] bounds."""
    random.seed(42)
    from src.distributions import sample_beta

    for _ in range(100):
        value = sample_beta()
        assert 0.0 <= value <= 1.0, f"Value {value} out of [0, 1] bounds"


def test_sample_beta_biased_toward_one():
    """Tests that beta distribution with alpha=5, beta=2 is biased toward 1.0."""
    random.seed(42)
    from src.distributions import sample_beta

    samples = [sample_beta(alpha=5.0, beta=2.0) for _ in range(1000)]

    mean = np.mean(samples)
    # For Beta(5, 2), mean = 5/(5+2) ≈ 0.714
    assert 0.65 <= mean <= 0.76, f"Mean {mean} not in expected range ~0.714"

    # Most samples should be > 0.5
    above_half = sum(1 for s in samples if s > 0.5)
    assert above_half >= 800, f"Only {above_half}/1000 above 0.5"


def test_sample_beta_biased_toward_zero():
    """Tests that beta distribution with alpha=2, beta=5 is biased toward 0.0."""
    random.seed(42)
    from src.distributions import sample_beta

    samples = [sample_beta(alpha=2.0, beta=5.0) for _ in range(1000)]

    mean = np.mean(samples)
    # For Beta(2, 5), mean = 2/(2+5) ≈ 0.286
    assert 0.24 <= mean <= 0.35, f"Mean {mean} not in expected range ~0.286"

    # Most samples should be < 0.5
    below_half = sum(1 for s in samples if s < 0.5)
    assert below_half >= 800, f"Only {below_half}/1000 below 0.5"


def test_sample_beta_uniform_like():
    """Tests that beta distribution with alpha=1, beta=1 is uniform."""
    random.seed(42)
    from src.distributions import sample_beta

    samples = [sample_beta(alpha=1.0, beta=1.0) for _ in range(10000)]

    # Divide range into 10 bins
    hist, _ = np.histogram(samples, bins=10, range=(0, 1))

    # Each bin should have ~1000 samples (10000 / 10)
    # Allow 20% deviation
    for count in hist:
        assert 800 <= count <= 1200, f"Bin count {count} suggests non-uniform"


def test_sample_parameter_with_beta():
    """Tests sample_parameter function with beta distribution."""
    random.seed(42)
    value = sample_parameter(0.0, 1.0, "beta")
    assert 0.0 <= value <= 1.0


# =============================================================================
# Tests for Lognormal distribution
# =============================================================================

def test_sample_lognormal_basic():
    """Tests that lognormal distribution samples within bounds."""
    random.seed(42)
    from src.distributions import sample_lognormal

    for _ in range(100):
        value = sample_lognormal(0.0, 10.0)
        assert 0.0 <= value <= 10.0, f"Value {value} out of bounds"


def test_sample_lognormal_biased_toward_min():
    """Tests that lognormal distribution is biased toward minimum."""
    random.seed(42)
    from src.distributions import sample_lognormal

    samples = [sample_lognormal(0.0, 100.0) for _ in range(10000)]

    # Lognormal should be biased toward minimum
    mean = np.mean(samples)
    assert mean < 40.0, f"Mean {mean} not biased toward minimum"

    # Most samples should be in lower half
    in_lower_half = sum(1 for s in samples if s < 50.0)
    assert in_lower_half >= 6000, f"Only {in_lower_half}/10000 in lower half"


def test_sample_lognormal_deterministic():
    """Tests that lognormal is deterministic with seed."""
    from src.distributions import sample_lognormal

    np.random.seed(123)
    s1 = [sample_lognormal(0.0, 100.0) for _ in range(10)]
    np.random.seed(123)
    s2 = [sample_lognormal(0.0, 100.0) for _ in range(10)]
    assert all(abs(a - b) < 1e-10 for a, b in zip(s1, s2)), "Lognormal not deterministic"


def test_sample_parameter_with_lognormal():
    """Tests sample_parameter function with lognormal distribution."""
    random.seed(42)
    value = sample_parameter(0.0, 20.0, "lognormal")
    assert 0.0 <= value <= 20.0


# =============================================================================
# Tests for Truncated Normal distribution
# =============================================================================

def test_sample_truncated_normal_basic():
    """Tests that truncated normal samples within bounds."""
    random.seed(42)
    from src.distributions import sample_truncated_normal

    for _ in range(100):
        value = sample_truncated_normal(-10.0, 10.0)
        assert -10.0 <= value <= 10.0, f"Value {value} out of bounds"


def test_sample_truncated_normal_centered():
    """Tests that truncated normal is centered at midpoint."""
    random.seed(42)
    from src.distributions import sample_truncated_normal

    samples = [sample_truncated_normal(0.0, 100.0) for _ in range(10000)]

    mean = np.mean(samples)
    # Mean should be close to 50 (midpoint)
    assert 48.0 <= mean <= 52.0, f"Mean {mean} not centered at 50"


def test_sample_truncated_normal_respects_bounds_strictly():
    """Tests that truncated normal never exceeds bounds (unlike clipped normal)."""
    random.seed(42)
    from src.distributions import sample_truncated_normal

    # Generate many samples with tight bounds
    samples = [sample_truncated_normal(0.0, 1.0) for _ in range(10000)]

    # ALL samples must be strictly within bounds
    assert all(0.0 <= s <= 1.0 for s in samples), "Some samples violated bounds"

    # Distribution should still be approximately normal-shaped
    hist, _ = np.histogram(samples, bins=5, range=(0, 1))
    # Middle bins should have more samples than edge bins
    assert hist[2] > hist[0] and hist[2] > hist[4], "Distribution not normal-shaped"


def test_sample_parameter_with_truncated_normal():
    """Tests sample_parameter function with truncated_normal distribution."""
    random.seed(42)
    value = sample_parameter(-5.0, 5.0, "truncated_normal")
    assert -5.0 <= value <= 5.0


def test_sample_parameter_batch_beta():
    """Tests batch sampling with beta distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    samples = sample_parameter_batch(0.0, 1.0, "beta", size=1000)

    assert samples.shape == (1000,)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_sample_parameter_batch_lognormal():
    """Tests batch sampling with lognormal distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    samples = sample_parameter_batch(0.0, 100.0, "lognormal", size=1000)

    assert samples.shape == (1000,)
    assert np.all((samples >= 0.0) & (samples <= 100.0))

    # Should be biased toward minimum
    mean = np.mean(samples)
    assert mean < 40.0


def test_sample_parameter_batch_truncated_normal():
    """Tests batch sampling with truncated_normal distribution."""
    np.random.seed(42)
    from src.distributions import sample_parameter_batch

    samples = sample_parameter_batch(0.0, 100.0, "truncated_normal", size=10000)

    assert samples.shape == (10000,)
    assert np.all((samples >= 0.0) & (samples <= 100.0))

    # Should be centered
    mean = np.mean(samples)
    assert 48.0 <= mean <= 52.0
