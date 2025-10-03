# Batch Generation System

## Overview

The batch generation system allows you to create large datasets with precise control over proportions, text directions, fonts, and other parameters - all in a single efficient run.

## Quick Start

### Basic Usage

```bash
python3 src/main.py \
  --batch-config batch_configs/simple_example.yaml \
  --output-dir output \
  --fonts-dir data/fonts \
  --text-file data/raw_text/corpus.txt
```

### Generate 1000 Multilingual Images

```bash
python3 src/main.py \
  --batch-config batch_configs/multilingual.yaml \
  --output-dir output \
  --fonts-dir data/fonts \
  --text-file data/raw_text/corpus.txt \
  --clear-output --force
```

## Key Features

### ✅ Proportional Allocation
Specify exact proportions for each batch:
- 40% English LTR
- 30% Arabic RTL
- 20% Japanese vertical
- 10% Mixed

### ✅ Per-Batch Configuration
Each batch can have:
- Custom text direction
- Different corpus file
- Font filters and weights
- Text length ranges
- Augmentation parameters (future)

### ✅ Interleaved Generation
Images are generated in a round-robin fashion across batches, ensuring balanced output even if interrupted.

### ✅ Font Validation
Automatic glyph coverage checking ensures fonts can render the corpus text (90% threshold).

### ✅ Reproducibility
Set a `seed` value for deterministic generation across runs.

### ✅ Single Script Run
No more shell scripts calling the generator 40+ times. One command, one process.

## Architecture

### Components

1. **BatchSpecification** - Defines a single batch with parameters
2. **BatchConfig** - Contains all batches and global settings
3. **BatchManager** - Orchestrates interleaved generation
4. **YAML Loader** - Parses configuration files

### Workflow

```
Load YAML Config
     ↓
Validate & Normalize Proportions
     ↓
Load & Filter Fonts (once)
     ↓
Allocate Images to Batches
     ↓
Interleaved Generation Loop
     ↓
Progress Tracking & Output
```

## Configuration Format

### Complete Example

```yaml
total_images: 1000
seed: 42

batches:
  - name: english_ltr
    proportion: 0.4
    text_direction: left_to_right
    corpus_file: data/raw_text/corpus.txt
    font_filter: "*.ttf"
    font_weights:
      "*sans*": 1.5
      "*mono*": 2.0
    min_text_length: 10
    max_text_length: 30

  - name: arabic_rtl
    proportion: 0.3
    text_direction: right_to_left
    corpus_file: data/raw_text/arabic_corpus.txt
    font_filter: "*{arabic,amiri}*.ttf"
    min_text_length: 8
    max_text_length: 25

  - name: japanese_vertical
    proportion: 0.3
    text_direction: top_to_bottom
    corpus_file: data/raw_text/japanese_corpus.txt
    font_filter: "*{cjk,noto*jp}*.{ttf,otf}"
    font_weights:
      "*noto*": 2.0
    min_text_length: 5
    max_text_length: 20
```

### Parameters Reference

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `total_images` | int | Yes | - | Total number of images to generate |
| `seed` | int | No | None | Random seed for reproducibility |
| **Batch Parameters** |
| `name` | string | Yes | - | Unique batch identifier |
| `proportion` | float | Yes | - | Fraction of total (0.0-1.0) |
| `text_direction` | string | No | "left_to_right" | "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "random" |
| `corpus_file` | string | No | CLI default | Path to text corpus |
| `font_filter` | string | No | "*.{ttf,otf}" | Glob pattern for fonts |
| `font_weights` | dict | No | {} | Pattern → weight mapping |
| `min_text_length` | int | No | 5 | Minimum characters |
| `max_text_length` | int | No | 25 | Maximum characters |

## Font Filters

### Syntax

Use shell glob patterns:

```yaml
# All TTF fonts
font_filter: "*.ttf"

# Arabic fonts
font_filter: "*{arabic,amiri,naskh}*.ttf"

# CJK fonts (TTF or OTF)
font_filter: "*{cjk,noto*jp,mincho}*.{ttf,otf}"

# Sans-serif regular weights
font_filter: "*sans*regular*.ttf"
```

### Font Weights

Assign relative weights to prioritize fonts:

```yaml
font_weights:
  "*noto*": 2.0      # 2x more likely
  "*bold*": 0.5      # Half as likely
  "*mono*": 1.0      # Normal probability
```

## Best Practices

### 1. Match Corpus to Fonts
Ensure each batch's corpus is compatible with its fonts:
- ✅ Arabic corpus → Arabic fonts
- ✅ Japanese corpus → CJK fonts
- ❌ Arabic corpus → Latin fonts (will skip)

### 2. Normalize Proportions
Proportions will auto-normalize if they don't sum to 1.0:
```yaml
# These get normalized to 0.4, 0.4, 0.2
proportions: [2, 2, 1]
```

### 3. Use Specific Font Filters
Better performance with targeted filters:
```yaml
# Good - specific pattern
font_filter: "*{noto,cjk}*.otf"

# Less ideal - matches everything
font_filter: "*.*"
```

### 4. Set Appropriate Text Lengths
Adjust per language/script:
- English: 10-30 chars
- Arabic: 8-25 chars (often shorter)
- CJK: 5-20 chars (more information per char)

### 5. Test with Small Batches
Before generating 10K images, test with 100:
```yaml
total_images: 100  # Test first
```

## Performance

### Optimizations

1. **Single font validation** - Fonts validated once at startup
2. **Interleaved generation** - Balanced output from start
3. **Lazy corpus loading** - Only load needed corpora
4. **Font-name optimization** - Skip validation when specific font used

### Comparison

**Old approach (run_battery_tests.sh):**
- 40 script invocations
- 40× font validation (~10 minutes)
- Sequential batch execution

**New approach (batch config):**
- 1 script invocation
- 1× font validation (~30 seconds)
- Interleaved generation
- **~20x faster**

## Troubleshooting

### Issue: "No fonts match filter"

**Cause:** Font filter pattern doesn't match any fonts

**Solution:**
- Check pattern syntax
- List available fonts: `ls data/fonts/`
- Use broader pattern: `"*.ttf"`

### Issue: "Insufficient glyph coverage"

**Cause:** Font can't render corpus characters

**Solution:**
- Use corpus-appropriate fonts
- Check font supports language/script
- Lower coverage threshold (not recommended)

### Issue: "Generated fewer images than expected"

**Cause:** Font-corpus incompatibilities skipped

**Solution:**
- Check warnings in output
- Ensure font filters match corpus
- Verify fonts have required glyphs

## Examples

See `batch_configs/` directory:
- `simple_example.yaml` - Basic 2-batch config
- `multilingual.yaml` - 5-batch multilingual
- `advanced.yaml` - Font weights & varied lengths

## Migration Guide

### From Shell Script to Batch Config

**Before (run_battery_tests.sh):**
```bash
for direction in ltr rtl ttb btt; do
  for i in {1..10}; do
    python3 src/main.py \
      --text-file corpus.txt \
      --num-images 1 \
      --text-direction $direction
  done
done
```

**After (batch config):**
```yaml
total_images: 40
batches:
  - {name: ltr, proportion: 0.25, text_direction: left_to_right}
  - {name: rtl, proportion: 0.25, text_direction: right_to_left}
  - {name: ttb, proportion: 0.25, text_direction: top_to_bottom}
  - {name: btt, proportion: 0.25, text_direction: bottom_to_top}
```

```bash
python3 src/main.py --batch-config batches.yaml --output-dir output ...
```

## Future Enhancements

- [ ] Per-batch augmentation parameters
- [ ] Multiple corpus files per batch (random selection)
- [ ] Background image filters per batch
- [ ] Font size ranges per batch
- [ ] Progress callbacks/hooks
- [ ] Resume interrupted generation
- [ ] Metadata export (batch info per image)
