# Multi-Line Text Generation

This guide explains how to configure and use the multi-line text generation feature to create training data with text ranging from single characters to full paragraphs.

## Overview

The multi-line text generation feature allows you to create synthetic OCR training data with varying amounts of text on a single image, from single characters up to full paragraphs with multiple lines. This feature is fully configurable and works with all text directions and augmentations.

## Key Concepts

### Line Count

- **`min_lines`**: Minimum number of text lines to generate (default: 1)
- **`max_lines`**: Maximum number of text lines to generate (default: 1)

When `max_lines > 1`, the system enters multi-line mode and breaks the text across multiple lines.

### Line Breaking Modes

The system supports two line breaking strategies, configurable for any text direction:

- **`word`**: Respects word boundaries (breaks at whitespace) - suitable for languages with word separators
- **`character`**: Breaks text at any character position - suitable for continuous scripts or when precise character distribution is needed

### Line Spacing

- **`line_spacing_min`**: Minimum line spacing multiplier (e.g., 1.0 = single spacing)
- **`line_spacing_max`**: Maximum line spacing multiplier (e.g., 1.5 = 1.5x spacing)
- **`line_spacing_distribution`**: Distribution type for sampling line spacing (default: "uniform")

Line spacing controls the vertical (for horizontal text) or horizontal (for vertical text) distance between lines.

### Text Alignment

Controls how lines are aligned relative to each other:

- For horizontal text (LTR/RTL): `"left"`, `"center"`, `"right"`
- For vertical text (TTB/BTT): `"top"`, `"center"`, `"bottom"`

## Configuration

### Basic Multi-Line Configuration

Here's a basic example for generating 2-5 lines of text:

```yaml
specifications:
  - name: "multiline_example"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "english_text.txt"

    # Text length (total characters across all lines)
    min_text_length: 50
    max_text_length: 200

    # Multi-line parameters
    min_lines: 2
    max_lines: 5
    line_break_mode: "word"
    line_spacing_min: 1.0
    line_spacing_max: 1.5
    line_spacing_distribution: "uniform"
    text_alignment: "left"
```

### Single Character to Paragraph

To generate text ranging from single characters to full paragraphs:

```yaml
specifications:
  - name: "varied_length"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "corpus.txt"

    # Character range: 1 character to full paragraph
    min_text_length: 1
    max_text_length: 500

    # Line range: single line to multi-line paragraph
    min_lines: 1
    max_lines: 10

    line_break_mode: "word"
    line_spacing_min: 1.2
    line_spacing_max: 1.8
    text_alignment: "left"
```

### Character-Based Line Breaking

For scripts without word boundaries or when you need precise character distribution:

```yaml
specifications:
  - name: "character_breaking"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "text.txt"

    min_text_length: 30
    max_text_length: 100
    min_lines: 3
    max_lines: 7

    # Break at any character position
    line_break_mode: "character"

    line_spacing_min: 1.0
    line_spacing_max: 1.5
    text_alignment: "center"
```

### Vertical Multi-Line Text

Multi-line works with all text directions, including vertical:

```yaml
specifications:
  - name: "vertical_multiline"
    proportion: 1.0
    text_direction: "top_to_bottom"
    corpus_file: "vertical_text.txt"

    min_text_length: 20
    max_text_length: 100
    min_lines: 2
    max_lines: 5

    line_break_mode: "character"
    line_spacing_min: 1.2
    line_spacing_max: 1.5

    # Use alignment appropriate for vertical text
    text_alignment: "top"
```

## Effects and Augmentations

All effects and augmentations are applied uniformly across all lines of multi-line text:

```yaml
specifications:
  - name: "multiline_with_effects"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "text.txt"

    min_lines: 3
    max_lines: 5

    # Effects apply to all lines uniformly
    blur_radius_min: 0.0
    blur_radius_max: 1.0

    noise_amount_min: 0.0
    noise_amount_max: 0.2

    # Curves apply to each line individually
    curve_type: "arc"
    arc_radius_min: 50.0
    arc_radius_max: 200.0
```

### How Effects Are Applied

- **Image-level effects** (blur, noise, brightness, etc.): Applied to the entire multi-line image uniformly
- **Text-level effects** (ink bleed, shadows, glyph overlap): Applied before line composition
- **Curves** (arc, sine): Applied to each individual line with the same parameters

This ensures visual consistency - if the image is blurry, all lines have the same blur. If text is curved, each line curves the same way.

## Truth Data Format

For multi-line text, the truth data includes a `line_index` field for each character bounding box:

```json
{
  "text": "Full text content",
  "lines": ["Line 1", "Line 2", "Line 3"],
  "num_lines": 3,
  "line_spacing": 1.2,
  "line_break_mode": "word",
  "text_alignment": "left",
  "bboxes": [
    {
      "char": "L",
      "x0": 10,
      "y0": 5,
      "x1": 20,
      "y1": 35,
      "line_index": 0
    },
    {
      "char": "i",
      "x0": 22,
      "y0": 5,
      "x1": 28,
      "y1": 35,
      "line_index": 0
    },
    ...
  ]
}
```

### Using Line Index

The `line_index` field allows you to:
- Group characters by line
- Calculate per-line bounding boxes
- Train models that are line-aware
- Analyze text layout and structure

Single-line mode includes `line_index: 0` for all character bounding boxes for consistency with multi-line format.

## Best Practices

### 1. Text Length and Line Count Relationship

Ensure your text length ranges can accommodate your line counts:

```yaml
# Good: 100 chars / 5 lines = ~20 chars per line (reasonable)
min_text_length: 50
max_text_length: 100
min_lines: 2
max_lines: 5

# Problematic: 20 chars / 10 lines = 2 chars per line (too short)
min_text_length: 10
max_text_length: 20
min_lines: 5
max_lines: 10
```

### 2. Choose Appropriate Line Breaking Mode

- **Word mode**: Best for languages with clear word boundaries (English, French, etc.)
- **Character mode**: Best for continuous scripts (Chinese, Japanese), URLs, or when you need even character distribution

### 3. Line Spacing Considerations

- **Dense text (small spacing)**: Use 1.0-1.2x for challenging OCR scenarios
- **Standard text**: Use 1.2-1.5x for typical document layouts
- **Loose text (large spacing)**: Use 1.5-2.0x for easy-to-read layouts

### 4. Testing Multi-Line Configurations

Start with a small test batch to verify your configuration:

```yaml
total_images: 100  # Small test batch

specifications:
  - name: "test_multiline"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test_corpus.txt"
    min_lines: 2
    max_lines: 4
    line_break_mode: "word"
```

Inspect the generated images and truth data before scaling up to larger batches.

## Single-Line Mode

Single-line mode is the default behavior:

- **Default behavior**: `min_lines=1`, `max_lines=1` (single-line mode)
- **Single-line truth data**: Includes `line_index: 0` for all characters
- **Existing configurations**: Continue to work without modification

To enable multi-line mode, simply set `max_lines > 1` in your specification.

All truth data now uses a consistent format with `line_index` present for both single-line and multi-line images.

## Troubleshooting

### Lines Are Too Short

**Problem**: Text is broken into too many lines with very few characters per line.

**Solution**: Either decrease `max_lines` or increase `min_text_length`:

```yaml
# Before
min_text_length: 20
max_lines: 10

# After
min_text_length: 50
max_lines: 5
```

### Lines Are Not Breaking at Words

**Problem**: Words are split across lines.

**Solution**: Ensure you're using `line_break_mode: "word"`:

```yaml
line_break_mode: "word"  # Not "character"
```

### Vertical Lines Are Overlapping

**Problem**: Lines in vertical text are too close together.

**Solution**: Increase `line_spacing_min` and `line_spacing_max`:

```yaml
# For vertical text, spacing is horizontal
line_spacing_min: 1.5
line_spacing_max: 2.0
```

## Advanced Usage

### Mixed Single and Multi-Line Batches

You can create different specifications for different line counts:

```yaml
specifications:
  # Single-line examples (40%)
  - name: "single_line"
    proportion: 0.4
    min_lines: 1
    max_lines: 1
    min_text_length: 5
    max_text_length: 30

  # Multi-line paragraphs (60%)
  - name: "paragraphs"
    proportion: 0.6
    min_lines: 3
    max_lines: 8
    min_text_length: 100
    max_text_length: 400
```

### Variable Line Spacing

Use distributions to create more realistic line spacing variation:

```yaml
line_spacing_min: 1.0
line_spacing_max: 2.0
line_spacing_distribution: "normal"  # Cluster around midpoint (1.5)
```

## Example Configurations

See `configs/` directory for complete example configurations using multi-line text generation.
