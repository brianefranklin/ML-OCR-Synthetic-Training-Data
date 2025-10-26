# API Reference: Resource Management

This section describes the classes responsible for managing resources like fonts and backgrounds, including the automatic health tracking system that learns from generation failures and successes.

## Overview

The resource management system provides **automatic health tracking** for fonts and backgrounds during image generation. When a font or background causes a generation failure (e.g., font loading error, rendering failure), its health score decreases. Resources below a health threshold are automatically filtered out, preventing problematic resources from being selected in future generations.

## `resource_manager.py`

### `ResourceManager`

A generic, reusable class that provides a score-based health tracking system for any type of resource.

#### Health Scoring Mechanics

- **Starting Score:** 100 points (all resources start healthy)
- **Success:** +1 point per successful generation (capped at 100)
- **Failure:** -10 points per failed generation
- **Health Threshold:** 50 points (resources below this are denylisted)

#### Denylisting Behavior

A resource becomes denylisted after **6 consecutive failures** (100 - 60 = 40 < 50 threshold).

Example:
```python
font_manager = FontHealthManager()

# After 6 failures, a problematic font is denylisted
for _ in range(6):
    font_manager.record_failure("/fonts/problematic.ttf")

available = font_manager.get_available_fonts(all_fonts)
# problematic.ttf will NOT be in available fonts
```

#### Key Methods

- `record_failure(resource_id)`: Records a generation failure for a resource
- `record_success(resource_id)`: Records a successful generation for a resource
- `get_available_resources(all_resources)`: Returns only healthy resources (score >= 50)
- `select_resource(resource_list)`: Weighted random selection favoring healthier resources

#### Weighted Selection

Healthier resources are more likely to be selected. Selection probability is proportional to health score:

```python
# Font A: score 100, Font B: score 50
# Font A is selected 2x more often than Font B (100:50 ratio)
```

## `font_health_manager.py`

### `FontHealthManager`

A specialized subclass of `ResourceManager` for managing font files.

#### Integration with Generation Pipeline

The `FontHealthManager` is automatically integrated into the generation pipeline. When a font fails to load or causes a rendering error:

1. **Failure is detected** in the worker process during `generate_image_from_task()`
2. **Error is logged** with the font path
3. **Health score decreases** via `record_failure(font_path)` in the main process
4. **After 6 failures**, the font is automatically denylisted

#### Example: Real Production Behavior

From a production run generating 30,000 images:

```
WARNING - Failed to generate image 954 with font 'data.nosync/fonts/NotoColorEmojiCompatTest-Regular.ttf': ValueError: invalid pixel size
```

After this font failed 14 times:
- **Health score:** 100 - (14 × 10) = -40 points
- **Status:** Denylisted (well below threshold of 50)
- **Future selections:** This font will not be selected again in this session

#### Key Methods

- `get_available_fonts(all_fonts)`: Returns list of healthy fonts
- `select_font(font_list)`: Selects a single font with weighted probability

## `background_manager.py`

### `BackgroundImageManager`

A specialized subclass of `ResourceManager` for managing background images.

#### Key Features

1. **Directory Discovery:** Automatically discovers image files from specified directories
2. **Health Tracking:** Inherits all health tracking behavior from `ResourceManager`
3. **Integration:** Tracks both successes and failures during generation

#### Directory Weighting

Background directories can be weighted to control selection probability:

```python
background_manager = BackgroundImageManager(
    dir_weights={
        "backgrounds/high_quality": 0.7,   # 70% of selections
        "backgrounds/experimental": 0.3,   # 30% of selections
    }
)
```

#### Key Methods

- `get_available_backgrounds()`: Returns set of healthy background paths
- `select_background()`: Selects a single background with weighted probability

## Health Tracking in Production

### Automatic Behavior

The health tracking system operates automatically during all image generation:

- **No configuration required** - enabled by default
- **Transparent operation** - failures are logged but generation continues
- **Session-based** - health scores reset each generation run
- **Thread-safe** - works correctly in parallel generation

### Example: Mixed Success/Failure Pattern

```python
# Font starts at 100 points
font_manager.record_failure(font)   # 90 points
font_manager.record_failure(font)   # 80 points
font_manager.record_success(font)   # 81 points
font_manager.record_failure(font)   # 71 points
# Font still available (above 50 threshold)
```

### Monitoring Health Status

To check a resource's health status during development:

```python
font_manager = FontHealthManager()

# After some generations...
record = font_manager._get_or_create_record("/fonts/test.ttf")
print(f"Health score: {record.health_score}")

# Check if font is available
available = font_manager.get_available_fonts(["/fonts/test.ttf"])
if available:
    print("Font is healthy")
else:
    print("Font is denylisted")
```

## Design Rationale

### Why Score-Based vs. Binary?

A score-based system (vs. immediate denylisting after one failure) provides several benefits:

1. **Tolerance for transient failures:** Occasional failures due to temporary issues (e.g., file system delays) don't immediately denylist resources
2. **Recovery mechanism:** Resources can recover with successful generations (+1 per success)
3. **Weighted selection:** Partially degraded resources are selected less often but not completely excluded
4. **Statistical filtering:** Only consistently problematic resources are denylisted

### Why -10 for failure, +1 for success?

This asymmetric scoring reflects that:
- **Failures are expensive:** A failed generation wastes compute and delays the batch
- **Successes are expected:** Most resources should work most of the time
- **Quick denylisting:** Consistently bad resources are removed after 6 failures
- **Slow recovery:** Resources must prove reliability with many successes to recover

### Session-Based vs. Persistent

Health scores currently **reset each session** (not persisted to disk). This design choice:

- **Pros:** Allows retrying fonts in new sessions (failures may be context-specific)
- **Cons:** Doesn't learn across sessions

Future enhancement: Add optional persistence for long-running production environments.
