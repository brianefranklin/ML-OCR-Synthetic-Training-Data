# How-To: Run a Generation Job

This guide explains how to run the main data generation script, `main.py`.

## Basic Usage

The script is executed from the command line and requires several arguments to specify the configuration and data locations.

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config [PATH_TO_YAML] \
    --output-dir [OUTPUT_DIRECTORY] \
    --font-dir [FONT_DIRECTORY] \
    --background-dir [BACKGROUND_DIRECTORY] \
    --corpus-dir [CORPUS_DIRECTORY]
```

### Argument Breakdown

- **`PYTHONPATH=.`**: This is a crucial environment variable setting. It tells the Python interpreter to include the current directory in its search path, which allows it to find the `src` module correctly.
- **`python3 src/main.py`**: This executes the main script.
- **`--batch-config`**: The path to your YAML configuration file. This file defines the total number of images and the specifications for each batch.
- **`--output-dir`**: The directory where the generated images and their corresponding JSON label files will be saved.
- **`--font-dir`**: The path to the directory containing all your `.ttf` font files.
- **`--background-dir`**: The path to the directory containing all your background images.
- **`--corpus-dir`**: The path to the directory containing your text corpus files.

### Example

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config batch_config.yaml \
    --output-dir ./output \
    --font-dir ./data.nosync/fonts \
    --background-dir ./data.nosync/backgrounds \
    --corpus-dir ./data.nosync/corpus_text/ltr
```

This command will start the generation process, displaying a progress bar in the console. Upon completion, the `output` directory will contain the generated images and their JSON labels.

## Additional Command-Line Options

### Resume Mode (`--resume`)

Resume interrupted generation jobs by skipping already-generated images:

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config batch_config.yaml \
    --output-dir ./output \
    --font-dir ./data.nosync/fonts \
    --background-dir ./data.nosync/backgrounds \
    --corpus-dir ./data.nosync/corpus_text \
    --resume
```

**How it works**:
- Scans output directory for existing `image_*.png` files
- Loads checkpoint file (`.generation_checkpoint.json`) if it exists
- Skips generation for existing images
- Warns if configuration has changed since last run
- Updates checkpoint after each chunk/batch

**When to use**:
- Generation was interrupted (crash, manual stop, timeout)
- Want to continue after partial completion
- Need to recover from errors mid-generation

**Note**: Always use the same configuration file when resuming. Changing the configuration will trigger a warning but allow resume to continue (useful for debugging, but may result in inconsistent output).

### Logging Level (`--log-level`)

Control the verbosity of log output:

```bash
python3 -m src.main \
    --batch-config batch_config.yaml \
    ... \
    --log-level DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

- **DEBUG**: Detailed information for debugging
- **INFO**: General progress and status messages (default)
- **WARNING**: Only warnings and errors
- **ERROR**: Only errors and critical issues

### Parallel Processing

See [Parallel Generation Guide](parallel_generation.md) for detailed information on:
- `--generation-workers`: Parallel image generation
- `--workers`: Parallel I/O operations
- `--chunk-size`: Streaming chunk size for memory efficiency
- `--io-batch-size`: I/O batching for performance

## Automatic Configuration Validation

Before generation starts, the system automatically validates:

✓ **Resource Existence**:
- Font directory exists and contains `.ttf` files
- Background directory exists
- Corpus directory exists
- Corpus files specified in config exist

✓ **Configuration Values**:
- `total_images` is positive
- Specifications list is not empty
- Padding ranges are valid (min ≤ max, non-negative)
- Font size ranges are valid (positive, min ≤ max)
- Direction weights are non-negative

✓ **Required Fields**:
- All required top-level fields present
- Specifications have required fields

**If validation fails**, generation stops with a clear error message:

```
============================================================
ERROR: Configuration validation failed
============================================================

Specification 'my_spec': No corpus files found matching pattern 'missing.txt'
in directory /path/to/corpus

Please fix the configuration and try again.
============================================================
```

## Color Configuration Examples

The system supports three text color modes: uniform (single color), per-glyph (different color per character), and gradient (smooth transition). All modes support RGB color ranges for variation.

### Uniform Color Mode (Default: Black Text)

Generate images with black text (backward compatible with existing configs):

```yaml
specifications:
  - name: "black_text"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"  # Can be omitted, defaults to uniform
    text_color_min: [0, 0, 0]  # Black
    text_color_max: [0, 0, 0]  # Black
```

Generate images with random dark blue text:

```yaml
specifications:
  - name: "dark_blue_text"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"
    text_color_min: [0, 0, 100]    # Dark blue
    text_color_max: [50, 50, 200]   # Light blue range
```

Generate images with full RGB color variation:

```yaml
specifications:
  - name: "color_text"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"
    text_color_min: [0, 0, 0]       # Black
    text_color_max: [255, 255, 255] # White (full RGB range)
```

### Per-Glyph Color Mode

Generate rainbow text with different colors per character:

```yaml
specifications:
  - name: "rainbow_text"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "per_glyph"
    text_color_min: [0, 0, 0]       # Sample from full RGB cube
    text_color_max: [255, 255, 255]
    per_glyph_palette_size_min: 2   # Not used (palette size = text length)
    per_glyph_palette_size_max: 5   # Not used (palette size = text length)
```

**Note**: In per-glyph mode, one random color is sampled per character from the `text_color_min` to `text_color_max` range. The `per_glyph_palette_size_*` parameters are reserved for future use.

### Gradient Color Mode

Generate text with smooth red-to-blue gradient:

```yaml
specifications:
  - name: "red_blue_gradient"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "gradient"
    gradient_start_color_min: [255, 0, 0]   # Pure red
    gradient_start_color_max: [255, 0, 0]   # Pure red
    gradient_end_color_min: [0, 0, 255]     # Pure blue
    gradient_end_color_max: [0, 0, 255]     # Pure blue
```

Generate text with variable gradient (random start/end colors):

```yaml
specifications:
  - name: "variable_gradient"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "gradient"
    gradient_start_color_min: [200, 0, 0]   # Dark red range
    gradient_start_color_max: [255, 50, 50] # Bright red range
    gradient_end_color_min: [0, 0, 200]     # Dark blue range
    gradient_end_color_max: [50, 50, 255]   # Bright blue range
```

### Multi-Specification Color Strategy

Progressive training approach (easy → medium → hard):

```yaml
total_images: 30000

specifications:
  # Stage 1: Black text only (10k images, 33%)
  - name: "stage1_black"
    proportion: 0.33
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"
    text_color_min: [0, 0, 0]
    text_color_max: [0, 0, 0]

  # Stage 2: Grayscale variation (10k images, 33%)
  - name: "stage2_grayscale"
    proportion: 0.33
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"
    text_color_min: [0, 0, 0]       # Black
    text_color_max: [100, 100, 100] # Dark gray

  # Stage 3: Full color variation (10k images, 34%)
  - name: "stage3_color"
    proportion: 0.34
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    color_mode: "uniform"
    text_color_min: [0, 0, 0]
    text_color_max: [255, 255, 255]
```

### ML Training Recommendations

**For OCR model training**, consider this progression:

1. **Initial Training**: Start with uniform black text (`(0,0,0)`) to learn character shapes
2. **Color Robustness**: Add uniform color variation (`(0,0,0)` to `(255,255,255)`)
3. **Gradient Handling**: Include gradient mode for smooth color transitions
4. **Advanced**: Add per-glyph mode only for models that need extreme color invariance

**Color mode complexity**:
- Uniform < Gradient < Per-glyph
- Start simple, add complexity as model improves

## Font Size Configuration Examples

The system supports variable font sizes to help train scale-invariant OCR models. Font size is sampled uniformly from the specified range.

### Single Font Size

Generate images with consistent 32pt font (default):

```yaml
specifications:
  - name: "standard_size"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 32  # Can be omitted, defaults to 32
    font_size_max: 32  # Can be omitted, defaults to 32
```

### Small to Medium Range

Generate images with font sizes between 18-36pt:

```yaml
specifications:
  - name: "small_medium"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 18
    font_size_max: 36
```

### Full Range (Scale-Invariant Training)

Generate images with wide font size variation (18-120pt):

```yaml
specifications:
  - name: "scale_invariant"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 18
    font_size_max: 120
```

### Multi-Specification Font Size Strategy

Progressive scale training approach:

```yaml
total_images: 30000

specifications:
  # Stage 1: Small fonts only (10k images, 33%)
  - name: "stage1_small"
    proportion: 0.33
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 18
    font_size_max: 28

  # Stage 2: Medium fonts (10k images, 33%)
  - name: "stage2_medium"
    proportion: 0.33
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 40
    font_size_max: 60

  # Stage 3: Large fonts (10k images, 34%)
  - name: "stage3_large"
    proportion: 0.34
    text_direction: "left_to_right"
    corpus_file: "sample.txt"
    font_size_min: 70
    font_size_max: 120
```

### ML Training Recommendations

**For OCR model training**, consider this progression:

1. **Initial Training**: Start with single size (`font_size_min == font_size_max`) to learn shapes
2. **Narrow Range**: Add slight variation (e.g., 28-36) to improve generalization
3. **Medium Range**: Expand to moderate sizes (e.g., 24-72) for common use cases
4. **Full Range**: Use wide range (e.g., 18-120) only for fully scale-invariant models

**Font size considerations**:
- Smaller fonts (< 24pt) are harder to recognize and may require higher resolution
- Larger fonts (> 72pt) generate bigger images but provide more detail
- Most real-world OCR tasks use 24-48pt range

## Batch Planning Mode

For advanced use cases, you can separate the planning phase from the execution phase using the `plan_generation_batch()` method. This allows you to:

- Pre-generate all parameters before starting generation
- Analyze parameter distributions to verify your configuration
- Parallelize generation across multiple processes or machines
- Debug parameter selection without running full generation

### Example: Using Batch Planning Mode

```python
from src.batch_config import BatchConfig
from src.generation_orchestrator import GenerationOrchestrator
from src.generator import OCRDataGenerator
from src.background_manager import BackgroundImageManager
from pathlib import Path
import json

# Load configuration
batch_config = BatchConfig.from_yaml("batch_config.yaml")

# Initialize managers
background_manager = BackgroundImageManager(dir_weights={"./backgrounds": 1.0})
corpus_map = {f.name: str(f) for f in Path("./corpus").rglob('*.txt')}
all_fonts = [str(p) for p in Path("./fonts").rglob('*.ttf')] 

# Create orchestrator to get tasks
orchestrator = GenerationOrchestrator(
    batch_config=batch_config,
    corpus_map=corpus_map,
    all_fonts=all_fonts,
    background_manager=background_manager
)

# Get task list
tasks = orchestrator.create_task_list(min_text_len=10, max_text_len=50)

# Initialize generator
generator = OCRDataGenerator()

# PHASE 1: Generate all plans at once
print("Planning phase...")
task_tuples = [(task.source_spec, task.text, task.font_path) for task in tasks]
plans = generator.plan_generation_batch(task_tuples, background_manager)

# Save plans for later use or analysis
with open("plans.json", "w") as f:
    json.dump(plans, f, indent=2)

print(f"Generated {len(plans)} plans")

# PHASE 2: Execute generation (can be done separately or in parallel)
print("Execution phase...")
for i, plan in enumerate(plans):
    image, bboxes = generator.generate_from_plan(plan)
    image.save(f"output/image_{i:05d}.png")

    # Add bboxes to plan and save
    plan["bboxes"] = bboxes
    with open(f"output/image_{i:05d}.json", "w") as f:
        json.dump(plan, f, indent=4)
```

### Benefits of Batch Planning Mode

1. **Faster Iteration**: Test parameter distributions without full generation
2. **Parallelization**: Generate images in parallel across multiple processes
3. **Debugging**: Inspect exact parameters before committing compute resources
4. **Analysis**: Export plans for statistical analysis of parameter distributions
5. **Reproducibility**: Save plans to regenerate exact images later

## Troubleshooting

### `DecompressionBombWarning` with Large Backgrounds

If you use very large background images (e.g., > 90 megapixels), you may encounter a `DecompressionBombWarning` from the Pillow library. This is a security feature to prevent memory exhaustion. The current limit in the script is set to handle images just over 100 megapixels. If you need to use even larger images, you will need to edit `src/main.py` and increase the `Image.MAX_IMAGE_PIXELS` value defined at the top of the file.