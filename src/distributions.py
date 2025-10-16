"""Statistical distributions for parameter sampling.

This module provides functions for sampling parameters from different probability
distributions to create more realistic synthetic data. Instead of uniform sampling
across all parameter ranges, users can choose distributions that better match
real-world frequency patterns.

For example, most real-world text is straight (no curve), so arc_radius should use
an exponential distribution biased toward 0, rather than uniform distribution.
"""

import random
import math
from typing import Literal, Union
import numpy as np

# Type alias for supported distribution types
DistributionType = Literal["uniform", "normal", "exponential", "beta", "lognormal", "truncated_normal"]


def sample_parameter(
    min_val: float,
    max_val: float,
    distribution: DistributionType = "uniform"
) -> float:
    """Sample a parameter value from a specified distribution within bounds.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        distribution: Type of distribution ("uniform", "normal", "exponential").

    Returns:
        A sampled value within [min_val, max_val].

    Raises:
        ValueError: If distribution type is not recognized.

    Examples:
        >>> random.seed(42)
        >>> sample_parameter(0.0, 10.0, "uniform")
        6.394267984578837
        >>> sample_parameter(0.0, 10.0, "normal")
        5.123...
        >>> sample_parameter(0.0, 10.0, "exponential")
        1.234...
    """
    # Handle edge case where min equals max
    if min_val == max_val:
        return min_val

    if distribution == "uniform":
        return random.uniform(min_val, max_val)
    elif distribution == "normal":
        return sample_normal(min_val, max_val)
    elif distribution == "exponential":
        return sample_exponential(min_val, max_val)
    elif distribution == "beta":
        # Beta is naturally [0, 1], so scale to [min_val, max_val]
        return min_val + sample_beta() * (max_val - min_val)
    elif distribution == "lognormal":
        return sample_lognormal(min_val, max_val)
    elif distribution == "truncated_normal":
        return sample_truncated_normal(min_val, max_val)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def sample_normal(min_val: float, max_val: float) -> float:
    """Sample from a normal (Gaussian) distribution centered at the midpoint.

    The distribution is parameterized so that:
    - Mean = (min_val + max_val) / 2 (centered)
    - Standard deviation = (max_val - min_val) / 6 (3-sigma rule)
    - Values are clipped to [min_val, max_val]

    This ensures that approximately 99.7% of unclipped samples fall within the range,
    following the empirical 68-95-99.7 rule.

    Use this for parameters with a natural "center" value, such as:
    - rotation_angle (centered at 0°)
    - brightness_factor (centered at 1.0)
    - contrast_factor (centered at 1.0)

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Returns:
        A sample from normal distribution, clipped to [min_val, max_val].

    Examples:
        >>> random.seed(42)
        >>> samples = [sample_normal(0.0, 100.0) for _ in range(5)]
        >>> # Mean will be close to 50, with most values between 25 and 75
    """
    mean = (min_val + max_val) / 2.0
    sigma = (max_val - min_val) / 6.0  # 3-sigma = range

    # Sample from normal distribution
    value = random.gauss(mean, sigma)

    # Clip to bounds
    return max(min_val, min(max_val, value))


def sample_exponential(min_val: float, max_val: float) -> float:
    """Sample from an exponential distribution strongly biased toward minimum.

    The exponential distribution creates a strong bias toward the minimum value
    with rapid exponential decay. This models degradation effects where most samples
    should have minimal degradation (near min_val = best quality), with occasional
    severe degradation (near max_val = worst quality).

    The rate parameter λ is chosen to create strong bias toward minimum:
    λ = 30 / (max - min), giving mean ≈ (max - min) / 30

    Use this for degradation parameters where 0 = best quality, higher = worse:
    - arc_radius (most text is straight, radius ≈ 0)
    - sine_amplitude (most text is straight, amplitude ≈ 0)
    - blur_radius (most images are sharp, blur ≈ 0)
    - noise_amount (most images are clean, noise ≈ 0)
    - ink_bleed_radius (most text is crisp, no bleeding)
    - distortion parameters (most images undistorted)

    With this parameterization:
    - ~63% of samples will be in the first 10% of the range
    - ~86% of samples will be in the first 20% of the range
    - ~95% of samples will be within the full range
    - Mean ≈ (max - min) / 30 (strongly biased toward minimum)

    Args:
        min_val: Minimum value (inclusive) - mode of distribution (best quality).
        max_val: Maximum value (inclusive) - worst quality.

    Returns:
        A sample from exponential distribution, clipped to [min_val, max_val].

    Examples:
        >>> random.seed(42)
        >>> samples = [sample_exponential(0.0, 100.0) for _ in range(1000)]
        >>> # ~63% of values will be < 10, with exponential decay toward 100
        >>> # Mean will be around 3.3, median around 2.3
    """
    # Rate parameter: chosen to give mean ≈ (max - min) / 30
    # For exponential distribution: mean = 1 / λ
    # Therefore: λ = 30 / (max - min)
    # This gives ~63% of samples in first 10% of range
    range_size = max_val - min_val
    lambda_rate = 30.0 / range_size

    # Sample from exponential distribution
    # Note: random.expovariate(λ) returns value from Exp(λ)
    value = min_val + random.expovariate(lambda_rate)

    # Clip to maximum (min is already guaranteed by adding to min_val)
    return min(max_val, value)


def sample_beta(alpha: float = 2.0, beta: float = 5.0) -> float:
    """Sample from a beta distribution bounded in [0, 1].

    The beta distribution is naturally bounded in [0, 1], making it ideal for
    parameters that represent probabilities or proportions without needing rescaling.

    Args:
        alpha: Shape parameter (default 2.0).
        beta: Shape parameter (default 5.0).

    Returns:
        A sample from Beta(alpha, beta) in [0, 1].

    Examples:
        >>> random.seed(42)
        >>> sample_beta(alpha=5.0, beta=2.0)  # Biased toward 1.0
        >>> sample_beta(alpha=2.0, beta=5.0)  # Biased toward 0.0
        >>> sample_beta(alpha=1.0, beta=1.0)  # Uniform in [0, 1]
    """
    # Python's random module doesn't have beta, so use NumPy
    return np.random.beta(alpha, beta)


def sample_lognormal(min_val: float, max_val: float) -> float:
    """Sample from a lognormal distribution biased toward minimum.

    Lognormal is an alternative to exponential for modeling degradation effects.
    It has a right-skewed distribution similar to exponential but with a heavier tail.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Returns:
        A sample from lognormal distribution, clipped to [min_val, max_val].

    Examples:
        >>> random.seed(42)
        >>> sample_lognormal(0.0, 100.0)
        >>> # Values biased toward minimum with long tail
    """
    range_size = max_val - min_val
    # Parameters chosen to give similar bias to exponential
    mu = 0.0
    sigma = 0.8
    value = min_val + np.random.lognormal(mu, sigma) * (range_size / 10.0)
    return min(max_val, value)


def sample_truncated_normal(min_val: float, max_val: float) -> float:
    """Sample from a truncated normal distribution.

    Unlike clipped normal (sample_normal), this uses proper truncation which
    maintains the normal shape within bounds without accumulating probability
    mass at the boundaries.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Returns:
        A sample from truncated normal distribution in [min_val, max_val].

    Examples:
        >>> random.seed(42)
        >>> sample_truncated_normal(0.0, 100.0)
        >>> # Normally distributed around 50, but properly truncated at bounds
    """
    from scipy import stats
    mean = (min_val + max_val) / 2.0
    sigma = (max_val - min_val) / 6.0

    # Standardize bounds
    a = (min_val - mean) / sigma
    b = (max_val - mean) / sigma

    # Sample from truncated normal
    return stats.truncnorm.rvs(a, b, loc=mean, scale=sigma)


def sample_parameter_batch(
    min_val: float,
    max_val: float,
    distribution: DistributionType = "uniform",
    size: int = 1
) -> np.ndarray:
    """Sample multiple parameter values from a specified distribution (NumPy-optimized).

    This is a vectorized version of sample_parameter() that uses NumPy for efficient
    batch sampling. It is significantly faster than calling sample_parameter() in a loop
    for large batch sizes.

    Args:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        distribution: Type of distribution ("uniform", "normal", "exponential").
        size: Number of samples to generate.

    Returns:
        A NumPy array of shape (size,) containing sampled values within [min_val, max_val].

    Raises:
        ValueError: If distribution type is not recognized.

    Examples:
        >>> np.random.seed(42)
        >>> samples = sample_parameter_batch(0.0, 10.0, "uniform", size=1000)
        >>> samples.shape
        (1000,)
        >>> np.all((samples >= 0.0) & (samples <= 10.0))
        True

        >>> samples = sample_parameter_batch(0.0, 10.0, "normal", size=100)
        >>> # Most values will be near 5.0 (midpoint)

        >>> samples = sample_parameter_batch(0.0, 10.0, "exponential", size=100)
        >>> # Most values will be near 0.0 (minimum)
    """
    # Handle edge case where min equals max
    if min_val == max_val:
        return np.full(size, min_val, dtype=np.float64)

    if distribution == "uniform":
        return np.random.uniform(min_val, max_val, size=size)

    elif distribution == "normal":
        mean = (min_val + max_val) / 2.0
        sigma = (max_val - min_val) / 6.0  # 3-sigma rule
        samples = np.random.normal(mean, sigma, size=size)
        # Clip to bounds
        return np.clip(samples, min_val, max_val)

    elif distribution == "exponential":
        range_size = max_val - min_val
        lambda_rate = 30.0 / range_size
        # NumPy uses scale parameter (1/λ) not rate parameter (λ)
        scale = 1.0 / lambda_rate
        samples = min_val + np.random.exponential(scale, size=size)
        # Clip to maximum
        return np.minimum(samples, max_val)

    elif distribution == "beta":
        # Beta is naturally [0, 1], so scale to [min_val, max_val]
        samples = np.random.beta(2.0, 5.0, size=size)
        return min_val + samples * (max_val - min_val)

    elif distribution == "lognormal":
        range_size = max_val - min_val
        mu, sigma = 0.0, 0.8
        samples = min_val + np.random.lognormal(mu, sigma, size=size) * (range_size / 10.0)
        return np.minimum(samples, max_val)

    elif distribution == "truncated_normal":
        from scipy import stats
        mean = (min_val + max_val) / 2.0
        sigma = (max_val - min_val) / 6.0
        a = (min_val - mean) / sigma
        b = (max_val - mean) / sigma
        return stats.truncnorm.rvs(a, b, loc=mean, scale=sigma, size=size)

    else:
        raise ValueError(f"Unknown distribution: {distribution}")
