# How-To: Comprehensive Feature Testing

This guide explains how to use the comprehensive test configuration to validate all features of the synthetic OCR data generator.

## Overview

The comprehensive test configuration (`test_configs/comprehensive_test.yaml`) is designed to exercise every feature of the generator in a single batch run. This is useful for:

- **Feature validation**: Verify all features work together correctly
- **Regression testing**: Ensure new changes don't break existing functionality
- **Performance benchmarking**: Profile the full feature set
- **Distribution testing**: Validate all 6 statistical distributions
- **Integration testing**: Test all text directions and curve types

## Configuration Structure

The comprehensive config generates 100 images across 4 batch specifications:

### Specification 1: straight_ltr (40% - 40 images)

**Purpose**: Test straight left-to-right text with exponential distributions

**Features exercised**:
- Text direction: `left_to_right`
- Curve type: `none` (straight text)
- Distributions: Primarily `exponential` (degradation bias toward minimum)
- Effects: Rotation, ink bleed, noise, blur, perspective warp, elastic distortion
- Lighting: Normal distributions for brightness/contrast

**Typical use case**: Standard printed documents with occasional degradation

### Specification 2: curved_arc_rtl (30% - 30 images)

**Purpose**: Test curved arc right-to-left text with lognormal distributions

**Features exercised**:
- Text direction: `right_to_left`
- Curve type: `arc` with radius 50-300
- Distributions: Primarily `lognormal` (heavier tail than exponential)
- Effects: Grid/optical distortion, rotation, noise, blur
- Lighting: Truncated normal distributions

**Typical use case**: Arabic/Hebrew text on curved surfaces (product labels, book spines)

### Specification 3: wavy_sine_ttb (20% - 20 images)

**Purpose**: Test sine wave top-to-bottom text with beta/truncated_normal

**Features exercised**:
- Text direction: `top_to_bottom`
- Curve type: `sine` with configurable amplitude/frequency/phase
- Distributions: `beta` and `truncated_normal` for statistical variety
- Effects: Sine wave parameters, erosion/dilation, cutout, noise, blur
- Lighting: Truncated normal distributions

**Typical use case**: Vertical Asian scripts with wavy distortion

### Specification 4: stress_uniform_btt (10% - 10 images)

**Purpose**: Stress test with bottom-to-top text and uniform distributions

**Features exercised**:
- Text direction: `bottom_to_top`
- Curve type: `none` (focus on degradation)
- Distributions: All `uniform` (test full parameter ranges equally)
- Effects: ALL effects enabled with maximum ranges
- Lighting: Extreme brightness/contrast variations

**Typical use case**: Adversarial/stress testing for model robustness

## Running the Comprehensive Test

### Basic Execution

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config test_configs/comprehensive_test.yaml \
    --output-dir ./test_output \
    --font-dir ./data.nosync/fonts \
    --background-dir ./data.nosync/backgrounds \
    --corpus-dir ./data.nosync/corpus_text/ltr
```

### With Profiling (see [Profiling Guide](./profiling.md))

```bash
PYTHONPATH=. python3 scripts/profile_generation.py \
    --config test_configs/comprehensive_test.yaml \
    --output profile_results.txt
```

### Running Validation Tests

```bash
# Validate config structure
python3 -m pytest tests/test_comprehensive_config.py -v

# All tests should pass:
# ✓ test_comprehensive_config_loads_and_validates
# ✓ test_comprehensive_config_spec_names
# ✓ test_comprehensive_config_proportions
# ✓ test_comprehensive_config_text_directions
# ✓ test_comprehensive_config_curve_types
# ✓ test_comprehensive_config_distributions
# ✓ test_comprehensive_config_non_zero_parameters
```

## Expected Output

Generating 100 images with this config should produce:

- **40 images**: Straight LTR text with exponential degradation patterns
- **30 images**: Curved RTL text with lognormal degradation patterns
- **20 images**: Wavy TTB text with beta/truncated_normal patterns
- **10 images**: Stress test BTT with uniform (maximum variety)

Each image will have a corresponding JSON label file containing:
- All generation parameters
- Character-level bounding boxes
- Text direction and curve type
- Distribution types used (for analysis)

## Analyzing Results

### Verify Feature Coverage

```python
import json
from pathlib import Path
from collections import Counter

# Load all label files
labels_dir = Path("./test_output")
label_files = sorted(labels_dir.glob("*.json"))

# Collect statistics
text_directions = Counter()
curve_types = Counter()
distributions = Counter()

for label_file in label_files:
    with open(label_file) as f:
        plan = json.load(f)
        text_directions[plan["direction"]] += 1
        curve_types[plan["curve_type"]] += 1
        # Collect distribution types from various parameters
        # (distributions are not saved in plans, but could be added for analysis)

print("Text Directions:", dict(text_directions))
print("Curve Types:", dict(curve_types))
```

Expected output:
```
Text Directions: {'left_to_right': 40, 'right_to_left': 30, 'top_to_bottom': 20, 'bottom_to_top': 10}
Curve Types: {'none': 50, 'arc': 30, 'sine': 20}
```

### Verify Parameter Distributions

The comprehensive config uses all 6 distribution types. To verify distribution characteristics:

1. **Exponential** (spec 1): Check that most samples cluster near minimum values
2. **Lognormal** (spec 2): Check for right-skewed distributions with heavier tails
3. **Beta** (spec 3): Check bounded [0,1] samples scaled to parameter ranges
4. **Truncated Normal** (specs 2,3): Check centered distributions without edge accumulation
5. **Uniform** (spec 4): Check equal probability across full ranges
6. **Normal** (specs 1,2): Check bell curves centered at midpoints

## Troubleshooting

### Config Fails to Load

```
ValueError: Validation errors in specification 'X'
```

**Solution**: The config includes extensive validation. Check error message for specific issue:
- Invalid distribution types (must be one of: uniform, normal, exponential, beta, lognormal, truncated_normal)
- Proportions don't sum to 1.0
- Invalid text_direction or curve_type
- Curve parameters inconsistent with curve_type

### Generation is Slow

The comprehensive config exercises all features, so generation will be slower than minimal configs. For faster iteration:

1. Reduce `total_images` from 100 to 10
2. Disable heavy effects (elastic_distortion, grid_distortion)
3. Use profiling to identify bottlenecks

### Out of Memory

With 100 images and all effects enabled, memory usage can be high. Solutions:

1. Reduce `total_images`
2. Process in batches using batch planning mode
3. Reduce image sizes in corpus files

## Next Steps

- **Profile generation**: Use `scripts/profile_generation.py` to identify bottlenecks
- **Create custom configs**: Use this as a template for domain-specific configurations
- **Analyze distributions**: Export parameters for statistical analysis
- **Train OCR model**: Use generated data to evaluate synthetic data quality

## See Also

- [Running Generation Jobs](./run_generation.md)
- [Profiling Performance](./profiling.md) (if created)
- [Statistical Distributions](../conceptual/distributions.md)
- [Batch Configuration API](../api/batch_processing.md)
