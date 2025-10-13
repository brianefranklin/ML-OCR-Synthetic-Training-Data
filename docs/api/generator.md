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