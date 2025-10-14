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
