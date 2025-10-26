# API Reference: `generator.py`

This file contains the core `OCRDataGenerator` class that orchestrates the entire image generation pipeline.

## `OCRDataGenerator`

### `plan_generation(spec, text, font_path, background_manager)`

Creates a complete plan for generating a single image by randomly selecting parameter values from the ranges defined in the `BatchSpecification`.

- **`spec` (`BatchSpecification`):** The configuration object defining the ranges for all randomizable parameters.
- **`text` (`str`):** The text string to be rendered.
- **`font_path` (`str`):** The path to the `.ttf` font file to use.
- **`background_manager` (`BackgroundImageManager`, optional):** The manager to select a background from.

**Returns:** A `dict` containing the full, concrete generation plan with specific values for all parameters.

### `plan_generation_batch(tasks, background_manager)`

Creates multiple generation plans at once for batch processing. This method enables separation of planning from execution.

- **`tasks` (`List[Tuple[BatchSpecification, str, str]]`):** A list of tuples, each containing (spec, text, font_path).
- **`background_manager` (`BackgroundImageManager`, optional):** The manager to select backgrounds from.

**Returns:** A `List[Dict[str, Any]]` containing plan dictionaries, one for each input task.

**Use Cases:**
- **Pre-planning**: Generate all plans upfront, then execute generation separately
- **Analysis**: Examine parameter distributions before generating images
- **Debugging**: Inspect plans before committing to full generation
- **Separation of Concerns**: Decouple parameter selection from image rendering

**Note on Parallelization**: While this method can be used as part of a parallel workflow, actual parallel generation is handled by the `generate_image_from_task()` worker function in `src/main.py`. This method itself runs sequentially.

**Example:**
```python
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification

generator = OCRDataGenerator()
spec1 = BatchSpecification(name="batch1", proportion=1.0, ...)
spec2 = BatchSpecification(name="batch2", proportion=1.0, ...)

# Prepare tasks
tasks = [
    (spec1, "hello world", "/path/to/font1.ttf"),
    (spec2, "foo bar", "/path/to/font2.ttf")
]

# Generate all plans at once
plans = generator.plan_generation_batch(tasks, background_manager)

# Execute generation separately
for i, plan in enumerate(plans):
    image, bboxes = generator.generate_from_plan(plan)
    image.save(f"image_{i}.png")
```

### `generate_from_plan(plan)`

Deterministically generates an image based on a plan dictionary.

- **`plan` (`dict`):** The generation plan, typically loaded from a JSON label file.

**Returns:** A tuple `(image, bboxes)` where `image` is a PIL Image object and `bboxes` is a list of bounding box dictionaries.

### Internal Rendering Methods

- **`_render_multiline_text(...)`**: Renders multiple lines of text with proper spacing and alignment. Breaks text into lines, renders each line individually, and combines them into a single image. Applies effects uniformly across all lines. Returns bounding boxes with `line_index` field for multi-line awareness. Supports all text directions, curve types, and alignment options.

- **`_render_text(...)`**: A dispatcher that calls the correct rendering function for single-line text based on the text `direction` and `curve_type`. When curve parameters exceed threshold values (arc_radius > 1.0 or sine_amplitude > 0.1), dispatches to curved text renderers.

- **`_render_arc_text(...)`**: Renders text along a circular arc. Characters are positioned on a circle and rotated tangent to the curve. Uses transform-based bounding box calculation. Works with all 4 text directions.

- **`_render_sine_text(...)`**: Renders text along a sine wave pattern. Characters oscillate vertically (horizontal text) or horizontally (vertical text) following a sine function. Characters are rotated according to the wave's tangent. Works with all 4 text directions.

- **`_render_left_to_right(...)`**: Renders straight left-to-right text.
- **`_render_right_to_left(...)`**: Renders straight right-to-left text after BiDi reshaping.
- **`_render_top_to_bottom(...)`**: Renders straight vertical text from top to bottom.
- **`_render_bottom_to_top(...)`**: Renders straight vertical text from bottom to top.
- **`_render_text_surface(...)`**: Common helper for rendering horizontal straight text.
- **`_render_vertical_text(...)`**: Common helper for rendering vertical straight text.

## Multi-Line Text Generation

The generator supports multi-line text generation for creating training data with text ranging from single characters to full paragraphs. When `num_lines > 1` in the generation plan, the system automatically uses multi-line rendering.

### Multi-Line Parameters in Plan

Plans for multi-line generation include these additional fields:

- **`num_lines`**: Number of lines to generate
- **`lines`**: List of text strings (one per line)
- **`line_spacing`**: Line spacing multiplier
- **`line_break_mode`**: "word" or "character"
- **`text_alignment`**: Text alignment ("left", "center", "right" for horizontal; "top", "center", "bottom" for vertical)

### Bounding Box Format for Multi-Line

Multi-line text includes a `line_index` field in each character bounding box:

```python
{
    "char": "H",
    "x0": 10,
    "y0": 5,
    "x1": 25,
    "y1": 35,
    "line_index": 0  # Which line this character belongs to
}
```

Single-line mode (backward compatible) does not include the `line_index` field.

### Effects Application in Multi-Line Mode

- **Text-level effects** (glyph overlap, ink bleed, shadows): Applied before line composition
- **Curves**: Applied to each line individually with the same parameters
- **Image-level effects** (blur, noise, brightness, etc.): Applied uniformly to the entire multi-line image

See `docs/how-to/multiline_text.md` for detailed usage guide.