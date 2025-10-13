# Statistical Distributions for Parameter Sampling

## Overview

The Synthetic OCR Data Generator uses probability distributions to control how parameter values are sampled during image generation. Instead of uniform randomness across all parameter ranges, users can choose distributions that better match real-world frequency patterns.

## Why Distributions Matter

In real-world documents, certain characteristics appear more frequently than others:

- **Most text is straight**, not curved
- **Most images are sharp**, with minimal blur
- **Most text has no distortion**, with occasional perspective or elastic warping
- **Rotation is often centered around 0°**, with equal probability of slight left/right rotation
- **Brightness/contrast usually close to normal** (factor ≈ 1.0), with occasional dim or bright images

Using appropriate distributions creates more realistic training data, improving OCR model performance on real-world documents.

## Available Distributions

### Uniform Distribution

**When to use**: Parameters with no natural bias, or discrete values.

**Behavior**: Equal probability across the entire range.

**Examples**:
- `sine_phase` (0 to 2π, no natural bias)
- `sine_frequency` (no natural center point)
- `cutout_width/height` (discrete pixel values)
- `grid_distortion_steps` (discrete integer values)

**Statistical properties**:
```
Mean = (min + max) / 2
All values equally likely
```

### Normal Distribution (Gaussian)

**When to use**: Parameters with a natural "center" value.

**Behavior**: Bell curve centered at the midpoint, with most values near the center and fewer at the extremes.

**Examples**:
- `rotation_angle` (centered at 0°)
- `brightness_factor` (centered at 1.0 = normal brightness)
- `contrast_factor` (centered at 1.0 = normal contrast)

**Statistical properties**:
```
Mean = (min + max) / 2
Standard deviation = (max - min) / 6  (3-sigma rule)
~68% of values within ±σ of mean
~95% of values within ±2σ of mean
~99.7% of values within ±3σ of mean
```

**Example**: For `rotation_angle_min: -15.0` and `rotation_angle_max: 15.0`:
- Mean = 0°
- σ = 5°
- ~68% of images rotated between -5° and +5°
- ~95% between -10° and +10°
- Remaining 5% between -15° and -10° or +10° and +15°

### Exponential Distribution

**When to use**: Degradation effects where lower values represent better readability and higher values represent progressive quality loss. This distribution creates realistic datasets where most text is easy to read, with occasional difficult examples.

**Behavior**: Strong bias toward the minimum value (best quality), with exponential-like decay toward the maximum (worst quality). Most samples cluster near the minimum, simulating real-world conditions where most documents are reasonably legible.

**Primary Use Case - Text Degradation**:

In real-world OCR scenarios, most documents are readable with only occasional severe degradation. Exponential distribution models this by:
- Making 0 or near-0 (pristine quality) the most common case
- Creating a long tail toward higher values (severe degradation)
- Ensuring the OCR model sees mostly clean text during training, with exposure to edge cases

**Examples by Category**:

**1. Print/Scan Quality Degradation** (0 = perfect, higher = worse):
- `ink_bleed_radius`: Clean printed text (0) is common, blurred/bleeding ink (>3) is rare
- `blur_radius`: Sharp scans (0) are typical, out-of-focus images (>2) are occasional
- `noise_amount`: Clean images (0) dominate, noisy scans (>0.5) from poor lighting/sensors are uncommon

**2. Geometric Degradations** (0 = straight/undistorted, higher = worse):
- `arc_radius`: Straight text (0) is standard, curved text (>100) from book spines/packaging is rare
- `sine_amplitude`: Flat text (0) is normal, wavy text (>10) from wrinkled/warped paper is occasional
- `perspective_warp_magnitude`: Head-on scans (0) are common, angled photos (>0.3) are less frequent
- `elastic_distortion_alpha`: Undistorted (0) is typical, warped documents (>100) are rare
- `grid_distortion_limit`: No grid distortion (0) is standard, severe warping (>30) is uncommon
- `optical_distortion_limit`: No lens distortion (0) is common, fisheye effects (>0.5) are rare

**3. Character-Level Degradations** (0 = crisp, higher = worse):
- `glyph_overlap_intensity`: Properly spaced characters (0) are normal, overlapping glyphs (>0.3) from poor kerning/compression are rare

**4. Other Real-World Quality Loss** (0 = ideal, higher = worse):
- `erosion_dilation_kernel`: Original text thickness (1) is most common, morphological changes (>3) from photocopying/faxing are occasional

**Why This Matters for OCR Training**:

Using exponential distribution for degradation effects trains models that:
1. **Perform well on typical inputs**: Most training data is clean, matching real-world document quality
2. **Handle edge cases gracefully**: Exposure to occasional severe degradation prevents catastrophic failures
3. **Avoid overfitting to noise**: Unlike uniform distribution, models don't waste capacity learning equally from heavily degraded examples
4. **Match production data distribution**: The training set's quality distribution resembles actual deployment scenarios

**Statistical properties**:
```
Mode = min (most frequent value)
Mean ≈ min + (max - min) / 3
Distribution decreases monotonically from min to max
```

**Example**: For `arc_radius_min: 0.0` and `arc_radius_max: 300.0`:
- Mode = 0.0 (straight lines most common)
- Mean ≈ 100.0
- Most samples < 120
- Few samples between 200-300
- Creates realistic ratio: ~70% straight, ~30% curved

## Configuration

### YAML Configuration

Each parameter with a `_min` and `_max` range has a corresponding `_distribution` field:

```yaml
specifications:
  - name: "realistic_documents"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "corpus.txt"

    # Most text is straight (exponential = biased toward 0)
    arc_radius_min: 0.0
    arc_radius_max: 300.0
    arc_radius_distribution: "exponential"  # Default

    # Rotation centered at 0° (normal = bell curve)
    rotation_angle_min: -15.0
    rotation_angle_max: 15.0
    rotation_angle_distribution: "normal"  # Default

    # Override default: test extreme rotation with uniform distribution
    rotation_angle_distribution: "uniform"  # Custom
```

### Default Distribution Choices

The system provides sensible defaults based on real-world frequency:

#### Exponential (Strong bias toward minimum)
- `glyph_overlap_intensity`
- `ink_bleed_radius`
- `perspective_warp_magnitude`
- `elastic_distortion_alpha`
- `elastic_distortion_sigma`
- `grid_distortion_limit`
- `optical_distortion_limit`
- `noise_amount`
- `blur_radius`
- `arc_radius`
- `sine_amplitude`

#### Normal (Centered at midpoint)
- `rotation_angle`
- `brightness_factor`
- `contrast_factor`

#### Uniform (Equal probability)
- `grid_distortion_steps`
- `erosion_dilation_kernel`
- `cutout_width`
- `cutout_height`
- `sine_frequency`
- `sine_phase`

## Visual Comparison

### Arc Radius Distribution Comparison

Generating 1000 images with `arc_radius_min: 0.0`, `arc_radius_max: 300.0`:

**Uniform distribution**:
```
[0-60):   ████████████████████  (200 images, 20%)
[60-120): ████████████████████  (200 images, 20%)
[120-180):████████████████████  (200 images, 20%)
[180-240):████████████████████  (200 images, 20%)
[240-300]:████████████████████  (200 images, 20%)
```

**Exponential distribution** (recommended):
```
[0-60):   ████████████████████████████████████████  (400 images, 40%)
[60-120): ████████████████████████████  (280 images, 28%)
[120-180):███████████████  (150 images, 15%)
[180-240):████████  (80 images, 8%)
[240-300]:████  (90 images, 9%)
```

Exponential creates a more realistic dataset where straight text is common and highly curved text is rare.

## Technical Implementation

### Module: `src/distributions.py`

```python
from src.distributions import sample_parameter

# Sample with specified distribution
value = sample_parameter(
    min_val=0.0,
    max_val=10.0,
    distribution="exponential"  # or "uniform", "normal"
)
```

### Integration

The `OCRDataGenerator.plan_generation()` method uses `sample_parameter()` for all randomizable parameters, automatically using the distribution type specified in the `BatchSpecification`.

### Reproducibility

Distribution types are **NOT** saved in the plan/label file. Only the sampled values are saved. This ensures:

1. Perfect reproducibility: `generate_from_plan()` recreates the exact image from saved values
2. Smaller label files: No need to store distribution metadata
3. Flexibility: Can change distribution strategy without invalidating existing datasets

## Additional Use Cases and Considerations

### When to Override Default Distributions

While the defaults are chosen for typical OCR training scenarios, specific use cases may require different distributions:

#### 1. **Historical Document Restoration**

When training models for damaged historical documents:
```yaml
# Historical documents often have significant aging artifacts
blur_radius_min: 0.0
blur_radius_max: 5.0
blur_radius_distribution: "normal"  # Override: expect moderate blur more often

noise_amount_min: 0.0
noise_amount_max: 0.5
noise_amount_distribution: "normal"  # Override: aging noise is common, not rare
```

#### 2. **Mobile Phone Camera OCR**

For text captured via smartphone cameras (photos of receipts, signs, documents):
```yaml
# Phone photos often have perspective distortion
perspective_warp_magnitude_min: 0.0
perspective_warp_magnitude_max: 0.5
perspective_warp_magnitude_distribution: "normal"  # Override: angled shots are common

# Phone photos may have motion blur
blur_radius_min: 0.0
blur_radius_max: 3.0
blur_radius_distribution: "normal"  # Override: motion blur is frequent

# Varying lighting conditions
brightness_factor_min: 0.5
brightness_factor_max: 1.5
brightness_factor_distribution: "normal"  # Keep: centered at 1.0 is still appropriate
```

#### 3. **Low-Quality Photocopy/Fax Training**

For degraded photocopies and fax documents:
```yaml
# Photocopies often have erosion/dilation
erosion_dilation_kernel_min: 1
erosion_dilation_kernel_max: 5
erosion_dilation_kernel_distribution: "normal"  # Override: expect moderate changes

# Faxes have characteristic noise patterns
noise_amount_min: 0.1
noise_amount_max: 0.6
noise_amount_distribution: "uniform"  # Override: all noise levels common in faxes
```

#### 4. **Curved Text Specialists** (Product Labels, Book Spines)

When training specifically for curved text recognition:
```yaml
# Override to ensure significant curved text exposure
arc_radius_min: 50.0  # No straight text at all
arc_radius_max: 300.0
arc_radius_distribution: "uniform"  # Override: train on full spectrum of curvature

sine_amplitude_min: 5.0  # No straight text
sine_amplitude_max: 20.0
sine_amplitude_distribution: "uniform"  # Override: full range of wave patterns
```

#### 5. **Stress Testing / Adversarial Robustness**

For creating challenging test sets to evaluate model robustness:
```yaml
# Force uniform distributions for maximum variety
blur_radius_distribution: "uniform"  # Override: test full range equally
noise_amount_distribution: "uniform"  # Override: test full range equally
perspective_warp_magnitude_distribution: "uniform"  # Override: test full range equally
```

### Normal Distribution Use Cases

Normal distribution is specifically appropriate for:

#### 1. **Rotation from Scanning/Capture**
- Documents are rarely perfectly aligned
- Slight rotations (±5°) are very common
- Extreme rotations (±15°) are rare
- Centered at 0° makes sense

#### 2. **Lighting Variations**
- Most photos/scans have reasonable lighting (brightness ≈ 1.0)
- Too dark (brightness < 0.7) or too bright (brightness > 1.3) are less common
- Natural bell curve around "normal" brightness

#### 3. **Contrast Adjustments**
- Most documents have normal contrast (contrast ≈ 1.0)
- Washed-out (low contrast) or over-contrasted documents are occasional
- Centered distribution reflects typical document quality

### Uniform Distribution Use Cases

Uniform distribution is appropriate when:

#### 1. **No Natural Bias Exists**
- `sine_phase`: Phase offset has no "natural" starting point
- `sine_frequency`: Frequency has no preferred value for wavy text
- All values in range are equally plausible

#### 2. **Discrete Choices**
- `cutout_width/height`: Occlusion size has no preferred value
- `grid_distortion_steps`: Number of grid cells is arbitrary
- Equal probability across discrete options makes sense

#### 3. **Intentional Stress Testing**
- Override defaults with uniform to test models on full parameter range
- Useful for adversarial evaluation datasets

### Multi-Modal Real-World Scenarios

Some deployments encounter multiple distinct document types. Consider using multiple batch specifications:

```yaml
specifications:
  # 70% clean scanned documents
  - name: "clean_scans"
    proportion: 0.7
    blur_radius_distribution: "exponential"  # Mostly sharp
    noise_amount_distribution: "exponential"  # Mostly clean

  # 20% mobile phone captures
  - name: "mobile_photos"
    proportion: 0.2
    perspective_warp_magnitude_distribution: "normal"  # Often angled
    blur_radius_distribution: "normal"  # Frequent motion blur

  # 10% degraded photocopies
  - name: "photocopies"
    proportion: 0.1
    erosion_dilation_kernel_distribution: "normal"  # Expect changes
    noise_amount_distribution: "normal"  # Noisy copies common
```

This approach better matches production environments where different document sources have different quality profiles.

## Best Practices

### 1. Match Real-World Frequency

Choose distributions based on how often the effect appears in real documents:
- Common effects: `uniform` (50% of data should have this effect)
- Occasionally present: `normal` (effect present but varies in intensity)
- Rarely present: `exponential` (most data lacks this effect, strong degradation bias)

### 2. Start with Defaults

The default distribution choices are based on real-world document analysis. Only override when testing specific scenarios.

### 3. Validate Statistically

Generate a sample batch and verify the distribution matches expectations:

```python
import json
import matplotlib.pyplot as plt

# Load all label files
arc_radii = []
for label_file in label_files:
    with open(label_file) as f:
        plan = json.load(f)
        arc_radii.append(plan['arc_radius'])

# Plot histogram
plt.hist(arc_radii, bins=20)
plt.xlabel('Arc Radius')
plt.ylabel('Frequency')
plt.title('Distribution of Arc Radius Values')
plt.show()
```

### 4. Consider OCR Training Impact

Different distributions create different training scenarios:
- **Exponential**: Model learns to handle clean text primarily, with robustness to occasional effects
- **Normal**: Model expects moderate effects, balances clean and distorted text
- **Uniform**: Model trains on full spectrum equally, may be less realistic but more robust

## Examples

### Realistic Historical Documents

```yaml
specifications:
  - name: "historical_docs"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "historical.txt"

    # Very little curvature
    arc_radius_min: 0.0
    arc_radius_max: 500.0
    arc_radius_distribution: "exponential"

    # Significant aging artifacts
    noise_amount_min: 0.0
    noise_amount_max: 0.3
    noise_amount_distribution: "exponential"

    blur_radius_min: 0.0
    blur_radius_max: 3.0
    blur_radius_distribution: "exponential"

    # Varied document quality
    brightness_factor_min: 0.7
    brightness_factor_max: 1.3
    brightness_factor_distribution: "normal"
```

### Extreme Stress Testing

```yaml
specifications:
  - name: "stress_test"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"

    # Force uniform distributions for maximum variety
    arc_radius_min: 0.0
    arc_radius_max: 300.0
    arc_radius_distribution: "uniform"  # Override default

    rotation_angle_min: -45.0
    rotation_angle_max: 45.0
    rotation_angle_distribution: "uniform"  # Override default

    blur_radius_min: 0.0
    blur_radius_max: 5.0
    blur_radius_distribution: "uniform"  # Override default
```

## See Also

- [API Reference: distributions.py](../api/distributions.md)
- [API Reference: batch_config.py](../api/batch_processing.md)
- [How-To: Configure Batch Parameters](../how_to/configure_batches.md)
