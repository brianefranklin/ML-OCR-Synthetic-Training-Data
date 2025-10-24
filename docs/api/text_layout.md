# API Reference: `text_layout.py`

This module provides utilities for breaking text into multiple lines and calculating layout dimensions for multi-line text rendering.

## Functions

### `break_into_lines(text, max_chars_per_line, num_lines, break_mode="word")`

Breaks a text string into multiple lines according to the specified breaking strategy.

**Parameters:**
- **`text` (`str`)**: The text string to break into lines.
- **`max_chars_per_line` (`int`)**: Maximum characters per line (soft limit for word mode).
- **`num_lines` (`int`)**: Desired number of lines to create.
- **`break_mode` (`str`)**: Line breaking strategy - `"word"` or `"character"` (default: `"word"`).

**Returns:** `List[str]` - List of text strings, one per line.

**Behavior:**
- **Word mode (`"word"`)**: Respects word boundaries by breaking at whitespace. Tries to distribute text evenly across lines while keeping words intact.
- **Character mode (`"character"`)**: Breaks text at any character position, distributing characters evenly across lines.

**Special Cases:**
- If `num_lines == 1`, returns the original text as a single-element list.
- If text is shorter than `num_lines`, pads with empty strings.
- Empty text returns a list containing a single empty string.

**Examples:**
```python
from src.text_layout import break_into_lines

# Word mode - respects word boundaries
text = "Hello world testing"
lines = break_into_lines(text, 10, 2, "word")
# Returns: ["Hello world", "testing"]

# Character mode - breaks anywhere
text = "HelloWorld"
lines = break_into_lines(text, 5, 2, "character")
# Returns: ["Hello", "World"]

# Single line
text = "Hello"
lines = break_into_lines(text, 100, 1, "word")
# Returns: ["Hello"]
```

---

### `calculate_multiline_dimensions(lines, font, line_spacing, direction, glyph_overlap_intensity=0.0)`

Calculates the total pixel dimensions needed to render multi-line text.

**Parameters:**
- **`lines` (`List[str]`)**: List of text strings (one per line).
- **`font` (`PIL.ImageFont.FreeTypeFont`)**: The PIL font object to use for measurements.
- **`line_spacing` (`float`)**: Line spacing multiplier (e.g., 1.0 = single spacing, 1.5 = 1.5x spacing).
- **`direction` (`str`)**: Text direction - `"left_to_right"`, `"right_to_left"`, `"top_to_bottom"`, or `"bottom_to_top"`.
- **`glyph_overlap_intensity` (`float`)**: Character overlap intensity (0.0-1.0) (default: 0.0).

**Returns:** `Tuple[int, int]` - Tuple of (width, height) in pixels.

**Behavior:**
- **Horizontal text (LTR/RTL)**: Height grows with number of lines (stacked vertically), width is the maximum line width.
- **Vertical text (TTB/BTT)**: Width grows with number of lines (arranged side by side), height is the maximum line height.
- Line spacing affects the distance between lines.
- Glyph overlap reduces effective width/height by allowing characters to overlap.

**Examples:**
```python
from PIL import ImageFont
from src.text_layout import calculate_multiline_dimensions

font = ImageFont.truetype("/path/to/font.ttf", 32)

# Horizontal text dimensions
lines = ["Hello", "World"]
width, height = calculate_multiline_dimensions(
    lines, font, 1.2, "left_to_right", 0.0
)
# Returns: (max_width, total_height_with_spacing)

# Vertical text dimensions
lines = ["Hello", "World"]
width, height = calculate_multiline_dimensions(
    lines, font, 1.5, "top_to_bottom", 0.0
)
# Returns: (total_width_with_spacing, max_height)
```

---

### `calculate_line_positions(lines, font, line_spacing, alignment, direction)`

Calculates the position offset for each line of text based on alignment and direction.

**Parameters:**
- **`lines` (`List[str]`)**: List of text strings (one per line).
- **`font` (`PIL.ImageFont.FreeTypeFont`)**: The PIL font object to use.
- **`line_spacing` (`float`)**: Line spacing multiplier.
- **`alignment` (`str`)**: Text alignment - `"left"`, `"center"`, `"right"` for horizontal; `"top"`, `"center"`, `"bottom"` for vertical.
- **`direction` (`str`)**: Text direction.

**Returns:** `List[Tuple[int, int]]` - List of (x_offset, y_offset) tuples for each line.

**Behavior:**
- **Horizontal text**: Returns relative x positions based on alignment and y positions based on line index.
- **Vertical text**: Returns relative y positions based on alignment and x positions based on line index.

**Examples:**
```python
from PIL import ImageFont
from src.text_layout import calculate_line_positions

font = ImageFont.truetype("/path/to/font.ttf", 32)
lines = ["Short", "Longer line", "Med"]

# Left-aligned horizontal text
positions = calculate_line_positions(
    lines, font, 1.2, "left", "left_to_right"
)
# Returns: [(0, 0), (0, line_height), (0, 2*line_height)]

# Center-aligned horizontal text
positions = calculate_line_positions(
    lines, font, 1.2, "center", "left_to_right"
)
# Returns: [(x_offset_to_center_line_0, 0),
#           (x_offset_to_center_line_1, line_height), ...]
```

---

## Internal Helper Functions

These functions are used internally by `break_into_lines()` and are not typically called directly:

### `_break_by_words(text, max_chars_per_line, num_lines)`

Breaks text into lines respecting word boundaries. Works for any text direction where whitespace separates words.

### `_break_by_characters(text, num_lines)`

Breaks text into lines by character count. Distributes characters evenly across lines, with remainder characters going to earlier lines.

### `_calculate_horizontal_positions(lines, font, line_spacing, alignment)`

Calculates positions for horizontal text (LTR/RTL). Used by `calculate_line_positions()`.

### `_calculate_vertical_positions(lines, font, line_spacing, alignment)`

Calculates positions for vertical text (TTB/BTT). Used by `calculate_line_positions()`.

---

## Design Principles

This module adheres to the project's core principles:

1. **Universality First**: All functions are language-agnostic and work with any writing system. Line breaking modes are configurable for all text directions.

2. **No Assumptions**: The module makes no assumptions about which text direction should use which line breaking mode - everything is explicitly configurable.

3. **Deterministic**: Given the same inputs, functions always produce the same outputs.

## Usage in Generator

The `OCRDataGenerator` class uses these functions during multi-line text generation:

1. **Planning phase** (`plan_generation()`):
   - Calls `break_into_lines()` to split text into lines
   - Calls `calculate_multiline_dimensions()` to size the canvas

2. **Rendering phase** (`_render_multiline_text()`):
   - Renders each line individually
   - Combines lines with proper spacing and alignment

See `docs/how-to/multiline_text.md` for usage examples and `docs/api/generator.md` for integration details.
