# Background Image System

## Overview

The background image system provides sophisticated management of background images for OCR data generation. Instead of simple solid-color backgrounds, you can use real images as backgrounds with automatic validation, performance scoring, and intelligent selection.

**Key Features:**
- ✅ Per-batch background configuration
- ✅ Multiple source directories with weighted selection
- ✅ Automatic validation (size, corruption)
- ✅ Performance scoring (persisted to disk, session-only by default)
- ✅ Random region cropping for large backgrounds
- ✅ Fallback to solid colors when needed

## Quick Start

### Basic Usage

```yaml
# batch_config.yaml
total_images: 1000

batches:
  - name: with_backgrounds
    proportion: 1.0
    background_dirs:
      - data/backgrounds/textures
      - data/backgrounds/paper
    background_pattern: "*.{png,jpg,jpeg}"
```

### Advanced Configuration

```yaml
batches:
  - name: high_quality
    proportion: 0.7
    background_dirs:
      - data/backgrounds/high_res
      - data/backgrounds/scanned_documents
    background_pattern: "*.png"
    background_weights:
      data/backgrounds/high_res: 2.0
      data/backgrounds/scanned_documents: 1.0
    use_solid_background_fallback: true  # Fall back if no valid backgrounds
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────┐
│     BackgroundImageManager              │
│                                         │
│  1. Discover images from directories    │
│  2. Load performance scores from disk   │
│  3. Validate against canvas/text size   │
│  4. Crop random region if too large     │
│  5. Update scores based on success      │
│  6. Save scores at end (if enabled)     │
└─────────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────────┐
│     Canvas Placement                    │
│                                         │
│  - Text rendered with transparent bg    │
│  - Background composited underneath     │
│  - Text remains fully visible           │
└─────────────────────────────────────────┘
```

### Validation Pipeline

For each background candidate:

1. **Corruption Check**: Can the image be opened?
   - ❌ Fail → Penalty: 1.0 (severe), skip

2. **Text Bbox Check**: Is background larger than text bounding box?
   - ❌ Fail → Penalty: 1.0 (severe), skip
   - ✅ Pass → Continue

3. **Canvas Size Check**: Is background larger than canvas?
   - ❌ Fail → Penalty: 0.5 (moderate), skip
   - ✅ Pass → Use (crop random region if needed)

4. **Success**: Image used successfully
   - Penalty: 0.0 (none), score increases

### Score Persistence

**Session-Only Scoring (Default)**
- Scores start at 1.0 for each image
- Updated during generation based on validation results
- **NOT saved to disk** between runs
- Useful for batch jobs with varying canvas sizes

**Persistent Scoring (Optional)**
- Enable with `enable_persistence=True` parameter
- Scores saved to `.background_scores_<batch_name>.json`
- Loaded automatically on next run
- Useful for long-running training data generation

## Configuration Reference

### Batch Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `background_dirs` | list[string] | No | None | List of directories containing background images |
| `background_pattern` | string | No | "*.{png,jpg,jpeg}" | Glob pattern for image matching (supports brace expansion) |
| `background_weights` | dict | No | {} | Directory → weight mapping for weighted selection |
| `use_solid_background_fallback` | bool | No | True | Fall back to solid color if no valid backgrounds |

### Glob Patterns

**Simple Pattern:**
```yaml
background_pattern: "*.png"
```

**Multiple Extensions (Brace Expansion):**
```yaml
background_pattern: "*.{png,jpg,jpeg}"
```

**Specific Naming:**
```yaml
background_pattern: "scan_*.jpg"
```

### Directory Weights

Assign relative weights to prioritize certain directories:

```yaml
background_weights:
  data/backgrounds/high_quality: 3.0  # 3x more likely
  data/backgrounds/medium: 2.0        # 2x more likely
  data/backgrounds/low: 1.0           # Normal probability
```

## Performance Scoring

### How Scores Work

Each background starts with a score of **1.0**. Penalties are applied based on validation failures:

- **Corrupt image**: -1.0 (reduces score to 0.0)
- **Smaller than text bbox**: -1.0 (severe)
- **Smaller than canvas**: -0.5 (moderate)
- **Success**: -0.0 (no penalty)

Scores are clamped to minimum of **0.01** to ensure all images have some chance of selection.

### Selection Algorithm

Selection weight = `directory_weight × performance_score`

Example:
```
High-quality directory (weight: 2.0)
  - background1.png (score: 1.0) → selection weight: 2.0
  - background2.png (score: 0.5) → selection weight: 1.0

Low-quality directory (weight: 1.0)
  - background3.png (score: 1.0) → selection weight: 1.0
```

Random selection uses these weights, so `background1.png` is twice as likely as `background3.png`.

### Score Files

When persistence is enabled, scores are saved to:
```
.background_scores_<batch_name>.json
```

**Format:**
```json
{
  "/path/to/background1.png": 0.85,
  "/path/to/background2.png": 0.92,
  "/path/to/background3.png": 0.15
}
```

## Best Practices

### 1. Background Resolution

**Recommended:** Backgrounds should be **larger than your typical canvas size**

```yaml
# If canvas sizes are typically 800x600
# Use backgrounds that are at least 1000x800 or larger
```

**Why:** Backgrounds smaller than canvas are rejected with moderate penalty.

### 2. Directory Organization

Organize backgrounds by quality/type:

```
data/backgrounds/
├── high_quality/     # Clean scans, high res
├── paper_textures/   # Various paper backgrounds
└── aged_documents/   # Weathered, vintage looks
```

Then use weights:
```yaml
background_weights:
  data/backgrounds/high_quality: 2.0
  data/backgrounds/paper_textures: 1.5
  data/backgrounds/aged_documents: 1.0
```

### 3. Image Formats

- **PNG**: Lossless, good for high-quality backgrounds
- **JPEG**: Lossy but smaller files, good for large datasets
- **Avoid**: TIFF, BMP (not supported)

### 4. Validation Before Generation

Check that your backgrounds are suitable:

```python
from background_manager import BackgroundImageManager

manager = BackgroundImageManager(
    background_dirs=['data/backgrounds'],
    pattern='*.png'
)

print(f"Found {len(manager.backgrounds)} backgrounds")
stats = manager.get_statistics()
print(f"Average score: {stats['avg_score']:.2f}")
```

### 5. Fallback Strategy

Always enable fallback for production:
```yaml
use_solid_background_fallback: true
```

This ensures generation continues even if:
- No backgrounds available
- All backgrounds too small
- Images corrupted

### 6. Performance Considerations

**Large Background Collections:**
- Discovery is done once at initialization
- Memory usage scales with number of images
- For 10K+ backgrounds, consider splitting into weighted directories

**Canvas Size Variation:**
- If canvas sizes vary significantly, session-only scoring is better
- Persistent scoring works best with consistent canvas sizes

## Examples

### Example 1: Simple Paper Backgrounds

```yaml
total_images: 5000

batches:
  - name: handwritten
    proportion: 1.0
    text_direction: left_to_right
    background_dirs:
      - data/backgrounds/notebook_paper
      - data/backgrounds/lined_paper
    background_pattern: "*.jpg"
```

### Example 2: Mixed Quality Backgrounds

```yaml
total_images: 10000

batches:
  - name: training_data
    proportion: 1.0
    background_dirs:
      - data/backgrounds/pristine     # Clean scans
      - data/backgrounds/worn         # Aged, stained
      - data/backgrounds/damaged      # Creases, tears
    background_pattern: "*.{png,jpg}"
    background_weights:
      data/backgrounds/pristine: 2.0
      data/backgrounds/worn: 1.5
      data/backgrounds/damaged: 0.5
    use_solid_background_fallback: true
```

### Example 3: Language-Specific Backgrounds

```yaml
total_images: 8000

batches:
  - name: english_documents
    proportion: 0.5
    text_direction: left_to_right
    corpus_file: data/corpus/english.txt
    background_dirs:
      - data/backgrounds/us_forms
      - data/backgrounds/business_docs

  - name: arabic_documents
    proportion: 0.5
    text_direction: right_to_left
    corpus_file: data/corpus/arabic.txt
    background_dirs:
      - data/backgrounds/arabic_calligraphy_paper
      - data/backgrounds/middle_east_forms
```

## Troubleshooting

### Issue: No backgrounds selected

**Symptoms:**
- All images have solid color backgrounds
- Logs show "No background images available"

**Causes & Solutions:**

1. **Directory doesn't exist:**
   ```bash
   # Check directories exist
   ls -la data/backgrounds/
   ```

2. **Pattern doesn't match:**
   ```yaml
   # Too restrictive
   background_pattern: "scan_*.tif"  # No .tif support

   # Fix: Use supported formats
   background_pattern: "scan_*.{png,jpg}"
   ```

3. **All backgrounds too small:**
   - Check background dimensions
   - Ensure backgrounds ≥ canvas size (typically 800-2000px)

### Issue: Same backgrounds repeated

**Cause:** Limited background pool or heavy scoring penalties

**Solutions:**

1. **Add more backgrounds:**
   ```bash
   # Check count
   ls data/backgrounds/*.png | wc -l
   ```

2. **Reset scores (if using persistence):**
   ```bash
   rm .background_scores_*.json
   ```

3. **Adjust canvas size:**
   - Smaller canvas = more backgrounds valid
   - Or get larger background images

### Issue: Performance degradation

**Cause:** Very large number of backgrounds

**Solution:**

1. **Split into directories:**
   ```yaml
   background_dirs:
     - data/backgrounds/set_A  # 5000 images
     - data/backgrounds/set_B  # 5000 images
   ```

2. **Use specific patterns:**
   ```yaml
   # Good: targets specific subset
   background_pattern: "high_res_*.png"

   # Avoid: matches everything
   background_pattern: "*.*"
   ```

### Issue: Backgrounds look stretched/distorted

**This should NOT happen** - backgrounds are cropped, not resized.

If you see distortion:
1. Check the canvas_placement code
2. Verify background images aren't corrupted
3. Report as a bug

## API Reference

### BackgroundImageManager

```python
from background_manager import BackgroundImageManager

manager = BackgroundImageManager(
    background_dirs=['data/backgrounds'],  # List of directories
    pattern='*.{png,jpg}',                 # Glob pattern
    weights={'data/backgrounds': 1.0},     # Directory weights
    score_file='.background_scores.json',  # Score persistence file
    enable_persistence=False                # Enable score persistence
)

# Select background
bg_path = manager.select_background()

# Validate background
is_valid, reason, penalty = manager.validate_background(
    bg_path,
    canvas_size=(800, 600),
    text_bbox=(100, 100, 400, 300)
)

# Load and crop
bg_image = manager.load_and_crop_background(
    bg_path,
    canvas_size=(800, 600)
)

# Update score
manager.update_score(bg_path, penalty=0.1)

# Save scores (if persistence enabled)
manager.finalize()

# Get statistics
stats = manager.get_statistics()
# Returns: {
#   'total_backgrounds': int,
#   'avg_score': float,
#   'min_score': float,
#   'max_score': float
# }
```

## Migration from Old System

### Before: add_background() Augmentation

**Old approach** (DEPRECATED):
```python
from augmentations import add_background

# This resized backgrounds to text size - WRONG!
augmented = add_background(text_image, background_list)
```

### After: BackgroundImageManager + Canvas Placement

**New approach**:
```yaml
# Configured per-batch in YAML
batches:
  - name: my_batch
    background_dirs: ['data/backgrounds']
    background_pattern: '*.png'
```

**What changed:**
1. Backgrounds applied at **canvas placement stage**, not augmentation
2. Text rendered with **transparent background** first
3. Background composited **underneath** text
4. Backgrounds **cropped to canvas size**, not resized to text size
5. **Validation** ensures backgrounds are suitable
6. **Scoring** adapts to background performance

## Advanced Topics

### Custom Score Functions

By default, scores decrease with penalties. For custom scoring:

```python
class CustomBackgroundManager(BackgroundImageManager):
    def update_score(self, bg_path, penalty):
        # Custom scoring logic
        if penalty > 0.5:
            # Severe penalty: reduce score more aggressively
            self.backgrounds[bg_path] *= 0.5
        else:
            # Minor penalty: small reduction
            self.backgrounds[bg_path] *= 0.9
```

### Integration with Other Systems

**Font Health + Background Scoring:**
```python
# Both systems use similar patterns
font_health_manager = FontHealthManager(...)
background_manager = BackgroundImageManager(...)

# Both track performance and adapt selection
```

**Batch-Level Independence:**
Each batch has its own BackgroundImageManager, so different batches can:
- Use different background pools
- Have independent scoring
- Optimize independently

## Performance Metrics

**Background Manager Overhead:**
- Discovery: O(n) where n = number of images
- Selection: O(1) - constant time weighted random
- Validation: O(1) - single image open + dimension check
- Cropping: O(wh) where w×h = canvas size

**Memory Usage:**
- ~100 bytes per background (path + score)
- 10,000 backgrounds ≈ 1 MB

**Recommended Limits:**
- **< 10,000 backgrounds per directory**: Optimal performance
- **10,000 - 50,000**: Good performance, higher memory
- **> 50,000**: Consider splitting into multiple directories

## Future Enhancements

- [ ] Background categories/tags for semantic selection
- [ ] Automatic background augmentation (blur, distortion)
- [ ] Multi-layer backgrounds (texture + noise)
- [ ] Background color analysis for better text contrast
- [ ] Persistent scoring enabled by default for long-term training
- [ ] Background health reports (similar to font health)
- [ ] Parallel background loading for large datasets
- [ ] Background caching for frequently used images

## See Also

- [Batch Generation Documentation](BATCH_GENERATION.md)
- [Deterministic Generation](DETERMINISTIC_GENERATION.md)
- [Canvas Placement](../src/canvas_placement.py)
- [Background Manager Source](../src/background_manager.py)
