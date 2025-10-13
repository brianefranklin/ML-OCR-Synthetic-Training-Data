# API Reference: `generator.py`

This file contains the core `OCRDataGenerator` class that orchestrates the entire image generation pipeline.

## `OCRDataGenerator`

### `plan_generation(spec, text, font_path, background_manager)`

Creates a complete plan for generating a single image.

- **`spec` (`BatchSpecification`):** The configuration object defining the ranges for this batch.
- **`text` (`str`):** The text string to be rendered.
- **`font_path` (`str`):** The path to the `.ttf` font file to use.
- **`background_manager` (`BackgroundImageManager`, optional):** The manager to select a background from.

**Returns:** A `dict` containing the full generation plan.

### `generate_from_plan(plan)`

Deterministically generates an image based on a plan dictionary.

- **`plan` (`dict`):** The generation plan, typically loaded from a JSON label file.

**Returns:** A tuple `(image, bboxes)` where `image` is a PIL Image object and `bboxes` is a list of bounding box dictionaries.

### Internal Rendering Methods

- **`_render_text(...)`**: A dispatcher that calls the correct rendering function based on the text `direction`.
- **`_render_text_surface(...)`**: Renders horizontal (LTR, RTL) text.
- **`_render_vertical_text(...)`**: Renders vertical (TTB, BTT) text.
