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

Creates multiple generation plans at once for batch processing. This method enables separation of planning from execution, which is useful for pre-generating all parameters before starting generation, analyzing parameter distributions, or parallelizing generation workflows.

- **`tasks` (`List[Tuple[BatchSpecification, str, str]]`):** A list of tuples, each containing (spec, text, font_path).
- **`background_manager` (`BackgroundImageManager`, optional):** The manager to select backgrounds from.

**Returns:** A `List[Dict[str, Any]]` containing plan dictionaries, one for each input task.

**Use Cases:**
- **Pre-planning**: Generate all plans upfront, then execute generation separately
- **Analysis**: Examine parameter distributions before generating images
- **Parallelization**: Enable parallel generation across multiple processes
- **Debugging**: Inspect plans before committing to full generation

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

- **`_render_text(...)`**: A dispatcher that calls the correct rendering function based on the text `direction` and `curve_type`. When curve parameters exceed threshold values (arc_radius > 1.0 or sine_amplitude > 0.1), dispatches to curved text renderers.
- **`_render_arc_text(...)`**: Renders text along a circular arc. Characters are positioned on a circle and rotated tangent to the curve. Uses transform-based bounding box calculation. Works with all 4 text directions.
- **`_render_sine_text(...)`**: Renders text along a sine wave pattern. Characters oscillate vertically (horizontal text) or horizontally (vertical text) following a sine function. Characters are rotated according to the wave's tangent. Works with all 4 text directions.
- **`_render_left_to_right(...)`**: Renders straight left-to-right text.
- **`_render_right_to_left(...)`**: Renders straight right-to-left text after BiDi reshaping.
- **`_render_top_to_bottom(...)`**: Renders straight vertical text from top to bottom.
- **`_render_bottom_to_top(...)`**: Renders straight vertical text from bottom to top.
- **`_render_text_surface(...)`**: Common helper for rendering horizontal straight text.
- **`_render_vertical_text(...)`**: Common helper for rendering vertical straight text.