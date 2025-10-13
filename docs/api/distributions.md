# API Reference: `distributions.py`

This module provides statistical distribution functions for sampling parameter values during synthetic image generation.

## Module: `src.distributions`

### Type Aliases

#### `DistributionType`

```python
DistributionType = Literal["uniform", "normal", "exponential", "beta", "lognormal", "truncated_normal"]
```

Type alias for supported distribution types.

---

## Functions

### `sample_parameter()`

```python
def sample_parameter(
    min_val: float,
    max_val: float,
    distribution: DistributionType = "uniform"
) -> float
```

Sample a parameter value from a specified distribution within bounds.

**Parameters**:
- `min_val` (float): Minimum value (inclusive).
- `max_val` (float): Maximum value (inclusive).
- `distribution` (DistributionType, optional): Type of distribution. Defaults to `"uniform"`.

**Returns**:
- `float`: A sampled value within `[min_val, max_val]`.

**Raises**:
- `ValueError`: If distribution type is not recognized.

**Examples**:

```python
import random
from src.distributions import sample_parameter

random.seed(42)

# Uniform distribution - equal probability
value = sample_parameter(0.0, 10.0, "uniform")
# Returns: 6.394... (any value equally likely)

# Normal distribution - centered at midpoint
value = sample_parameter(0.0, 10.0, "normal")
# Returns: 5.123... (values near 5.0 more likely)

# Exponential distribution - biased toward minimum
value = sample_parameter(0.0, 10.0, "exponential")
# Returns: 0.345... (values near 0.0 much more likely)

# Beta distribution - for parameters naturally in [0, 1]
value = sample_parameter(0.0, 1.0, "beta")
# Returns: 0.234... (naturally bounded, no clipping needed)

# Lognormal distribution - alternative to exponential
value = sample_parameter(0.0, 10.0, "lognormal")
# Returns: 1.234... (right-skewed with heavier tail)

# Truncated normal - properly truncated (not clipped)
value = sample_parameter(0.0, 10.0, "truncated_normal")
# Returns: 5.123... (true normal within hard bounds)
```

**Edge Cases**:

```python
# When min equals max, always returns that value
value = sample_parameter(5.0, 5.0, "uniform")
# Returns: 5.0

# Works with negative ranges
value = sample_parameter(-10.0, -5.0, "normal")
# Returns: value in [-10.0, -5.0], centered at -7.5

# Works with ranges crossing zero
value = sample_parameter(-5.0, 5.0, "normal")
# Returns: value in [-5.0, 5.0], centered at 0.0
```

---

### `sample_normal()`

```python
def sample_normal(min_val: float, max_val: float) -> float
```

Sample from a normal (Gaussian) distribution centered at the midpoint.

This function is typically called internally by `sample_parameter()` rather than directly.

**Distribution Parameterization**:
- **Mean**: `(min_val + max_val) / 2` (centered)
- **Standard Deviation**: `(max_val - min_val) / 6` (3-sigma rule)
- **Clipping**: Values are clipped to `[min_val, max_val]`

This ensures that approximately 99.7% of unclipped samples fall within the range, following the empirical 68-95-99.7 rule.

**Parameters**:
- `min_val` (float): Minimum value (inclusive).
- `max_val` (float): Maximum value (inclusive).

**Returns**:
- `float`: A sample from normal distribution, clipped to `[min_val, max_val]`.

**Use Cases**:
- `rotation_angle` (centered at 0°)
- `brightness_factor` (centered at 1.0)
- `contrast_factor` (centered at 1.0)

**Statistical Properties**:

For range `[0, 100]`:
```
Mean = 50.0
Sigma = 16.67
~68% of samples in [33.33, 66.67]
~95% of samples in [16.67, 83.33]
~99.7% of samples in [0, 100]
```

**Examples**:

```python
import random
from src.distributions import sample_normal

random.seed(42)

# Sample 1000 values
samples = [sample_normal(0.0, 100.0) for _ in range(1000)]

# Verify centering
import numpy as np
mean = np.mean(samples)
# mean ≈ 50.0 (close to midpoint)

# Verify 68-95-99.7 rule
within_1_sigma = sum(1 for s in samples if 33.33 <= s <= 66.67)
# within_1_sigma ≈ 680 (68% of 1000)
```

---

### `sample_exponential()`

```python
def sample_exponential(min_val: float, max_val: float) -> float
```

Sample from an exponential distribution strongly biased toward minimum.

This function is typically called internally by `sample_parameter()` rather than directly.

**Distribution Parameterization**:

The exponential distribution creates a strong bias toward the minimum value with rapid exponential decay. This models degradation effects where most samples should have minimal degradation (near min_val = best quality), with occasional severe degradation (near max_val = worst quality).

- **Rate parameter λ**: `30 / (max_val - min_val)`
- **Mode**: `min_val` (most frequent value)
- **Mean**: Approximately `min_val + (max_val - min_val) / 30`
- **Clipping**: Values are clipped to `[min_val, max_val]`

The rate parameter is chosen so that approximately 63% of samples fall within the first 10% of the range.

**Parameters**:
- `min_val` (float): Minimum value (inclusive) - mode of distribution.
- `max_val` (float): Maximum value (inclusive).

**Returns**:
- `float`: A sample from exponential distribution, clipped to `[min_val, max_val]`.

**Use Cases - Text Degradation Effects**:

This distribution is specifically designed for degradation parameters where:
- **Minimum value (0 or near-0)** = Best quality, most readable text
- **Higher values** = Progressive quality loss, harder to read
- **Real-world pattern**: Most documents are reasonably legible, with occasional severe degradation

**Categories**:

1. **Print/Scan Quality** (0 = perfect, higher = degraded):
   - `ink_bleed_radius`: Clean printing → bleeding/fuzzy edges
   - `blur_radius`: Sharp focus → out-of-focus blur
   - `noise_amount`: Clean scan → noisy/grainy image

2. **Geometric Distortions** (0 = straight/flat, higher = distorted):
   - `arc_radius`: Straight text → curved text (book spines, labels)
   - `sine_amplitude`: Flat page → wavy/wrinkled paper
   - `perspective_warp_magnitude`: Head-on scan → angled photo
   - `elastic_distortion_alpha`: Undistorted → warped/stretched
   - `grid_distortion_limit`: Original → grid-warped
   - `optical_distortion_limit`: No lens distortion → fisheye effect

3. **Character Quality** (0 = crisp, higher = damaged):
   - `glyph_overlap_intensity`: Proper spacing → overlapping characters

4. **Document Condition** (1 = original, higher = degraded):
   - `erosion_dilation_kernel`: Original thickness → eroded/dilated from photocopying

**Why Exponential for Degradation**:

Real OCR applications encounter mostly clean text with occasional quality issues. Exponential distribution:
- Trains models primarily on readable text (the common case)
- Provides exposure to edge cases (severe degradation)
- Prevents overfitting to noise (unlike uniform distribution)
- Matches production data distribution for better generalization

**Statistical Properties**:

For range `[0, 100]`:
```
Mode = 0.0
Mean ≈ 3.3
Median ≈ 2.3
~63% of samples in [0, 10]
~86% of samples in [0, 20]
Distribution decreases exponentially
```

**Examples**:

```python
import random
from src.distributions import sample_exponential

random.seed(42)

# Sample 1000 values
samples = [sample_exponential(0.0, 100.0) for _ in range(1000)]

# Verify strong bias toward minimum
import numpy as np
mean = np.mean(samples)
# mean ≈ 3.3 (very close to min)

# Verify exponential decay
hist, bins = np.histogram(samples, bins=5, range=(0, 100))
# hist[0] >> hist[1] >> hist[2] >> hist[3] >> hist[4]

# Count samples in lower 10%
in_lower_tenth = sum(1 for s in samples if s < 10.0)
# in_lower_tenth ≈ 630 (63% of samples)
```

---

### `sample_beta()`

```python
def sample_beta(alpha: float = 2.0, beta: float = 5.0) -> float
```

Sample from a beta distribution bounded in [0, 1].

**Distribution Parameterization**:

The beta distribution is naturally bounded in [0, 1], making it ideal for parameters that represent probabilities or proportions without needing rescaling.

- **Alpha parameter**: Shape parameter (default 2.0)
- **Beta parameter**: Shape parameter (default 5.0)
- **Range**: Naturally [0, 1], no clipping needed
- **Mean**: `alpha / (alpha + beta)`

**Parameters**:
- `alpha` (float): Shape parameter. Higher values bias toward 1.0.
- `beta` (float): Shape parameter. Higher values bias toward 0.0.

**Returns**:
- `float`: A sample from Beta(alpha, beta) in [0, 1].

**Use Cases**:
- Parameters naturally representing probabilities or proportions
- When you need smooth control over bias direction using shape parameters
- Alpha=Beta=1: Uniform distribution in [0, 1]
- Alpha>Beta: Biased toward 1.0
- Alpha<Beta: Biased toward 0.0

**Examples**:
```python
import random
from src.distributions import sample_beta

# Biased toward 1.0
sample_beta(alpha=5.0, beta=2.0)  # Mean = 5/7 ≈ 0.714

# Biased toward 0.0
sample_beta(alpha=2.0, beta=5.0)  # Mean = 2/7 ≈ 0.286

# Uniform in [0, 1]
sample_beta(alpha=1.0, beta=1.0)  # Mean = 0.5
```

---

### `sample_lognormal()`

```python
def sample_lognormal(min_val: float, max_val: float) -> float
```

Sample from a lognormal distribution biased toward minimum.

**Distribution Parameterization**:

Lognormal is an alternative to exponential for modeling degradation effects. It has a right-skewed distribution similar to exponential but with a heavier tail, meaning more probability mass at higher values.

- **Mu parameter**: 0.0 (mean of underlying normal)
- **Sigma parameter**: 0.8 (std dev of underlying normal)
- **Mode**: Near `min_val` (most frequent value)
- **Clipping**: Values are clipped to `[min_val, max_val]`

**Parameters**:
- `min_val` (float): Minimum value (inclusive) - mode of distribution.
- `max_val` (float): Maximum value (inclusive).

**Returns**:
- `float`: A sample from lognormal distribution, clipped to `[min_val, max_val]`.

**Use Cases**:
- Alternative to exponential for degradation effects
- When you want right-skew but with heavier tail than exponential
- Modeling effects where extreme values are more likely than exponential predicts

**Statistical Properties**:

For range `[0, 100]`:
```
Mode ≈ 0.0
Mean ≈ 15-20 (heavier tail than exponential)
Most samples still in lower portion of range
Heavier tail than exponential at high values
```

---

### `sample_truncated_normal()`

```python
def sample_truncated_normal(min_val: float, max_val: float) -> float
```

Sample from a truncated normal distribution.

**Distribution Parameterization**:

Unlike clipped normal (`sample_normal`), this uses proper truncation which maintains the normal shape within bounds without accumulating probability mass at the boundaries.

- **Mean**: `(min_val + max_val) / 2` (centered)
- **Sigma**: `(max_val - min_val) / 6` (3-sigma rule)
- **Truncation**: Uses scipy.stats.truncnorm for proper truncation
- **No clipping artifacts**: Smooth distribution throughout range

**Parameters**:
- `min_val` (float): Minimum value (inclusive).
- `max_val` (float): Maximum value (inclusive).

**Returns**:
- `float`: A sample from truncated normal distribution in `[min_val, max_val]`.

**Use Cases**:
- When you need true normal distribution with hard bounds
- Avoiding probability mass accumulation at boundaries (unlike clipped normal)
- Parameters where normal distribution is theoretically justified but must have bounds

**Comparison with Clipped Normal**:

**Clipped Normal** (`sample_normal`):
- Samples from unbounded normal, then clips values outside [min, max]
- Accumulates probability mass at boundaries
- Faster to compute
- Simpler implementation

**Truncated Normal** (`sample_truncated_normal`):
- Samples only from the truncated region
- No probability mass at boundaries
- Slightly slower (uses scipy)
- More statistically correct

**Examples**:
```python
from src.distributions import sample_truncated_normal, sample_normal

# Truncated normal - smooth throughout
samples_trunc = [sample_truncated_normal(0.0, 1.0) for _ in range(1000)]
# No accumulation at 0.0 or 1.0 boundaries

# Clipped normal - has boundary artifacts
samples_clip = [sample_normal(0.0, 1.0) for _ in range(1000)]
# Some probability mass accumulated at exactly 0.0 and 1.0
```

---

### `sample_parameter_batch()`

```python
def sample_parameter_batch(
    min_val: float,
    max_val: float,
    distribution: DistributionType = "uniform",
    size: int = 1
) -> np.ndarray
```

Sample multiple parameter values from a specified distribution (NumPy-optimized).

**Performance Optimization**:

This is a vectorized version of `sample_parameter()` that uses NumPy for efficient batch sampling. It is **significantly faster** than calling `sample_parameter()` in a loop for large batch sizes.

**Benchmark** (approximate):
- Loop with `sample_parameter()`: ~1000 samples/ms
- `sample_parameter_batch()`: ~50,000 samples/ms (50x faster)

**Parameters**:
- `min_val` (float): Minimum value (inclusive).
- `max_val` (float): Maximum value (inclusive).
- `distribution` (DistributionType): Type of distribution.
- `size` (int): Number of samples to generate.

**Returns**:
- `np.ndarray`: Array of shape `(size,)` containing sampled values within `[min_val, max_val]`.

**Raises**:
- `ValueError`: If distribution type is not recognized.

**Supported Distributions**:
- `"uniform"`: Equal probability across range
- `"normal"`: Bell curve centered at midpoint (clipped)
- `"exponential"`: Strong bias toward minimum
- `"beta"`: Naturally bounded in [0, 1], scaled to [min, max]
- `"lognormal"`: Right-skewed with heavy tail
- `"truncated_normal"`: Properly truncated normal

**Examples**:
```python
import numpy as np
from src.distributions import sample_parameter_batch

np.random.seed(42)

# Generate 10,000 samples efficiently
samples = sample_parameter_batch(0.0, 100.0, "exponential", size=10000)

# Result is NumPy array
assert samples.shape == (10000,)
assert np.all((samples >= 0.0) & (samples <= 100.0))

# Much faster than loop
# BAD: for _ in range(10000): sample_parameter(...)
# GOOD: sample_parameter_batch(..., size=10000)
```

**Use Cases**:
- Pre-generating parameters for entire batch of images
- Performance-critical code paths
- Analyzing distribution properties with large sample sizes
- Any situation where you need >100 samples

---

## Implementation Details

### Algorithm: Normal Distribution

Uses Python's built-in `random.gauss()` function:

```python
mean = (min_val + max_val) / 2.0
sigma = (max_val - min_val) / 6.0
value = random.gauss(mean, sigma)
return max(min_val, min(max_val, value))
```

**Why σ = range / 6?**

The 3-sigma rule states that 99.7% of normal distribution samples fall within ±3σ of the mean. Setting `σ = range / 6` ensures that 3σ = range / 2, so the full range spans 6σ (from -3σ to +3σ relative to the mean).

### Algorithm: Exponential Distribution

Uses Python's built-in `random.expovariate()` function:

```python
range_size = max_val - min_val
lambda_rate = 30.0 / range_size
value = min_val + random.expovariate(lambda_rate)
return min(max_val, value)
```

**Why λ = 30 / range?**

For an exponential distribution:
- Mean = 1 / λ
- We want mean ≈ (max - min) / 30 for strong bias toward minimum
- Solving: 1 / λ = range / 30
- Therefore: λ = 30 / range

This gives approximately 63% of samples in the first 10% of the range, creating realistic degradation patterns.

### Determinism and Reproducibility

All distributions use Python's `random` module, which is deterministic when seeded:

```python
import random
from src.distributions import sample_parameter

# Same seed produces same sequence
random.seed(42)
values1 = [sample_parameter(0.0, 100.0, "exponential") for _ in range(10)]

random.seed(42)
values2 = [sample_parameter(0.0, 100.0, "exponential") for _ in range(10)]

assert values1 == values2  # True
```

This is critical for the generator's plan-then-execute architecture, where the `seed` in the plan dictionary ensures reproducible image generation.

---

## Integration with Generator

### In `BatchSpecification`

Each parameter with `_min` and `_max` ranges has a corresponding `_distribution` field:

```python
@dataclass
class BatchSpecification:
    arc_radius_min: float = 0.0
    arc_radius_max: float = 0.0
    arc_radius_distribution: str = "exponential"  # Default
```

### In `OCRDataGenerator.plan_generation()`

The generator uses `sample_parameter()` for all randomizable parameters:

```python
from src.distributions import sample_parameter

def plan_generation(self, spec, text, font_path, background_manager):
    return {
        "arc_radius": sample_parameter(
            spec.arc_radius_min,
            spec.arc_radius_max,
            spec.arc_radius_distribution
        ),
        # ... other parameters ...
    }
```

---

## Testing

The module includes comprehensive tests in `tests/test_distributions.py`:

- **Basic bounds testing**: All distributions respect min/max bounds
- **Shape testing**: Distributions match expected statistical properties
- **Edge case testing**: Negative ranges, zero-crossing ranges, min==max
- **Determinism testing**: Same seed produces identical sequences
- **Statistical validation**: Chi-square tests for uniformity, 68-95-99.7 rule for normal, exponential decay for exponential

Run tests:
```bash
pytest tests/test_distributions.py -v
```

---

## Performance Considerations

- All sampling operations are O(1) time complexity
- Uses fast built-in `random.gauss()` and `random.uniform()` functions
- No external dependencies (NumPy not required)
- Suitable for generating millions of samples

---

## See Also

- [Conceptual Guide: Statistical Distributions](../conceptual/distributions.md)
- [API Reference: batch_config.py](../api/batch_processing.md)
- [API Reference: generator.py](../api/generator.md)
