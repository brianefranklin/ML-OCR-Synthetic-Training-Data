# How-To: Profile Generation Performance

This guide explains how to profile the OCR data generation process to identify performance bottlenecks and optimization opportunities.

## Overview

The profiling script (`scripts/profile_generation.py`) uses Python's `cProfile` module to analyze where time is spent during image generation. This helps identify:

- **Slow functions**: Which operations take the most time
- **Call counts**: How many times each function is called
- **Optimization targets**: Where to focus performance improvements

## Basic Usage

```bash
python3 scripts/profile_generation.py \
    --config CONFIG_PATH \
    --output PROFILE_OUTPUT
```

### Required Arguments

- `--config`: Path to batch configuration YAML file

### Optional Arguments

- `--output`: Output file for results (default: `profile_results.txt`)
- `--limit`: Number of images to generate (default: 10)
- `--sort`: Sort results by `cumulative`, `time`, or `calls` (default: `cumulative`)
- `--top`: Number of top functions to display (default: 20)

## Examples

### Profile Comprehensive Test Config

```bash
python3 scripts/profile_generation.py \
    --config test_configs/comprehensive_test.yaml \
    --output comprehensive_profile.txt \
    --limit 10 \
    --sort cumulative \
    --top 30
```

This generates 10 images using the comprehensive config and saves detailed profiling results.

### Quick Profile with Minimal Config

```bash
python3 scripts/profile_generation.py \
    --config batch_config.yaml \
    --limit 5
```

Generates just 5 images for quick profiling iteration.

### Profile Sorted by Time Per Call

```bash
python3 scripts/profile_generation.py \
    --config test_configs/comprehensive_test.yaml \
    --sort time \
    --top 50
```

Identifies functions with the highest time-per-call (useful for finding individual slow operations).

## Interpreting Results

### Output Format

The profiling output includes several columns:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.050    0.001    2.500    0.025 module.py:42(function_name)
```

- **ncalls**: Number of times the function was called
- **tottime**: Total time spent in the function (excluding sub-calls)
- **percall**: Time per call (tottime / ncalls)
- **cumtime**: Cumulative time (including sub-calls)
- **percall**: Cumulative time per call (cumtime / ncalls)

### What to Look For

1. **High cumtime**: Functions that consume the most total time (including their callees)
2. **High tottime**: Functions that do the most work themselves (excluding callees)
3. **High ncalls**: Functions called many times (optimization candidates for batch operations)
4. **High tottime + high ncalls**: Prime optimization targets

### Example Analysis

```
Top functions by cumulative time:

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       10    0.001    0.000    5.234    0.523 generator.py:207(generate_from_plan)
       10    2.103    0.210    2.103    0.210 effects.py:62(add_noise)
       10    0.876    0.088    0.876    0.088 augmentations.py:135(apply_elastic_distortion)
       10    0.654    0.065    0.654    0.065 generator.py:305(_render_text)
     5000    0.523    0.000    0.523    0.000 effects.py:76(random.randint)  # OLD IMPLEMENTATION
```

**Insights**:
- `generate_from_plan` is the main entry point (52% of total time)
- `add_noise` takes 2.1 seconds (40% of generate_from_plan time) - **optimization target**
- `apply_elastic_distortion` takes 0.9 seconds (17%) - moderate cost
- `random.randint` called 5000 times in a loop - **vectorization opportunity**

After optimizing `add_noise` with NumPy:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       10    0.001    0.000    3.131    0.313 generator.py:207(generate_from_plan)
       10    0.042    0.004    0.042    0.004 effects.py:62(add_noise)  # 50x faster!
       10    0.876    0.088    0.876    0.088 augmentations.py:135(apply_elastic_distortion)
```

## Common Bottlenecks

Based on profiling the comprehensive test config, common bottlenecks include:

### 1. Loop-Based Image Processing

**Symptom**: High call counts for pixel-level operations in Python loops

**Solution**: Vectorize with NumPy

**Example**: `add_noise()` was optimized from loop-based to vectorized (50x speedup)

### 2. Image Transformations

**Symptom**: High time in augmentation functions (elastic, grid, optical distortion)

**Solutions**:
- Use OpenCV's optimized implementations
- Batch process multiple images
- Reduce distortion parameters for faster iteration

### 3. Text Rendering

**Symptom**: High time in `_render_text()` and related functions

**Solutions**:
- Cache font metrics
- Optimize glyph positioning calculations
- Use simpler fonts during development

### 4. File I/O

**Symptom**: High time in image save operations

**Solutions**:
- Use faster image formats (PNG with low compression)
- Batch file operations
- Use tmpfs/ramdisk for output during profiling

## Advanced Profiling

### Line-by-Line Profiling

For more detailed analysis, use `line_profiler`:

```bash
pip install line_profiler
```

Add `@profile` decorator to target functions and run:

```bash
kernprof -l -v scripts/profile_generation.py --config CONFIG
```

### Memory Profiling

To identify memory bottlenecks:

```bash
pip install memory_profiler
python -m memory_profiler scripts/profile_generation.py --config CONFIG
```

### Visual Profiling

For interactive exploration:

```bash
pip install snakeviz
python -m cProfile -o profile.prof scripts/profile_generation.py --config CONFIG
snakeviz profile.prof
```

## Optimization Workflow

1. **Establish baseline**: Profile current implementation
2. **Identify bottleneck**: Find the slowest function with high impact
3. **Optimize**: Implement improvement (vectorization, caching, etc.)
4. **Measure improvement**: Re-profile to verify speedup
5. **Repeat**: Move to next bottleneck

### Example Workflow

```bash
# 1. Baseline
python3 scripts/profile_generation.py --config test_configs/comprehensive_test.yaml --output baseline.txt

# Note: add_noise takes 2.1 seconds

# 2. Optimize add_noise (implement NumPy vectorization)

# 3. Measure improvement
python3 scripts/profile_generation.py --config test_configs/comprehensive_test.yaml --output optimized.txt

# Result: add_noise now takes 0.042 seconds (50x speedup!)
```

## Performance Targets

Based on the comprehensive test config with all features enabled:

- **Current**: ~0.5-1.0 seconds per image (after add_noise optimization)
- **Target**: <0.3 seconds per image with further optimizations
- **Theoretical**: ~0.1 seconds per image with full vectorization and caching

Actual performance depends on:
- Image size
- Number of effects enabled
- Effect intensities
- Hardware (CPU, memory)

## Troubleshooting

### "No module named 'src'"

**Solution**: Run with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python3 scripts/profile_generation.py --config CONFIG
```

### "FileNotFoundError: corpus/fonts/backgrounds"

**Solution**: Update paths in `profile_generation.py` or ensure data directories exist

### Profiling Takes Too Long

**Solution**: Reduce `--limit` to generate fewer images:

```bash
python3 scripts/profile_generation.py --config CONFIG --limit 3
```

## See Also

- [Comprehensive Testing Guide](./comprehensive_testing.md)
- [Running Generation Jobs](./run_generation.md)
- [add_noise() Optimization Example](../api/effects.md#add_noise)
