# Batch Configuration Examples

This directory contains example YAML batch configuration files for OCR synthetic data generation.

## Quick Start

Generate images using a batch configuration:

```bash
python3 src/main.py \
  --batch-config batch_configs/simple_example.yaml \
  --output-dir output \
  --fonts-dir data/fonts \
  --text-file data/raw_text/corpus.txt
```

## Configuration Format

### Basic Structure

```yaml
total_images: 1000        # Total number of images to generate
seed: 42                  # Optional: random seed for reproducibility

batches:
  - name: batch_name      # Identifier for this batch
    proportion: 0.6       # Proportion of total images (0.0-1.0)
    # ... batch-specific parameters
```

### Batch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Unique identifier for the batch |
| `proportion` | float | required | Fraction of total images (will be normalized if sum ≠ 1.0) |
| `text_direction` | string | "left_to_right" | Direction: "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", or "random" |
| `corpus_file` | string | CLI default | Path to text corpus file |
| `font_filter` | string | "*.{ttf,otf}" | Glob pattern to match fonts |
| `font_weights` | dict | {} | Font weight multipliers (pattern: weight) |
| `min_text_length` | int | 5 | Minimum text length in characters |
| `max_text_length` | int | 25 | Maximum text length in characters |

### Font Filters

Use glob patterns to select fonts:

```yaml
font_filter: "*.ttf"                           # All TTF fonts
font_filter: "*{arabic,amiri}*.ttf"           # Arabic fonts
font_filter: "*{cjk,noto*jp}*.{ttf,otf}"      # CJK fonts (TTF or OTF)
font_filter: "*{sans,mono}*regular*.ttf"       # Sans/mono regular weights
```

### Font Weights

Assign weights to prioritize certain fonts:

```yaml
font_weights:
  "*noto*": 2.0          # 2x weight for Noto fonts
  "*bold*": 0.5          # 0.5x weight (de-emphasize bold)
  "*amiri*": 3.0         # 3x weight for Amiri fonts
```

Higher weights = more likely to be selected.

## Example Configurations

### 1. Simple Example (`simple_example.yaml`)
- 100 images total
- 60% English (LTR)
- 40% Arabic (RTL)
- Basic font filtering

### 2. Multilingual (`multilingual.yaml`)
- 1000 images total
- 35% English horizontal
- 25% Arabic RTL (with font weights)
- 20% Japanese vertical (top-to-bottom)
- 10% Japanese vertical (bottom-to-top)
- 10% Random mixed

### 3. Advanced (`advanced.yaml`)
- 500 images total
- Varied text lengths per batch
- Font weight customization
- Specialty font handling
- Direction randomization

## Features

### Interleaved Generation
Images are generated in an interleaved fashion across all batches, ensuring balanced output even if generation is interrupted.

### Best-Effort Proportions
The system attempts to match specified proportions approximately. If a font cannot render a corpus (insufficient glyph coverage), it will skip that combination and continue.

### Font Validation
Each font is automatically validated against the corpus to ensure it can render the required characters (90% coverage threshold).

### Reproducibility
Set `seed` for reproducible image generation across runs.

## Tips

1. **Corpus Files**: Ensure each batch's `corpus_file` contains text in the appropriate script/language
2. **Font Filters**: Use specific patterns to target the right fonts (e.g., Arabic fonts for Arabic text)
3. **Proportions**: They will be normalized if they don't sum to 1.0
4. **Text Lengths**: Adjust per batch based on typical text patterns (Arabic often shorter, CJK can be longer)
5. **Direction**: Use "random" for varied training data within a batch

## Troubleshooting

**No fonts match filter:**
- Check font filter pattern syntax
- Verify fonts exist in fonts directory
- System falls back to all fonts if none match

**Low coverage warnings:**
- Font cannot render corpus characters
- Ensure corpus matches font capabilities (e.g., Arabic text needs Arabic fonts)

**Proportion mismatches:**
- Best-effort allocation - exact counts may vary slightly
- Check logs for skipped combinations
