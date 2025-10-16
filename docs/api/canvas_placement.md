# API Reference: Canvas Placement

This module contains functions for placing the rendered text surface onto a larger, final canvas.

## `canvas_placement.py`

### `generate_random_canvas_size(image_w, image_h, ...)`
- **Description:** Calculates a new, larger canvas size by adding random padding to the dimensions of the rendered text surface.

### `calculate_text_placement(canvas_w, canvas_h, ...)`
- **Description:** Calculates the `(x, y)` coordinates where the top-left corner of the text surface should be placed on the final canvas. Currently supports a `uniform_random` strategy.

### `place_on_canvas(text_image, ..., background_path)`
- **Description:** The core function that performs the placement.
- **Process:**
    1.  If a `background_path` is provided, it loads and crops the background image to the target canvas size.
    2.  If no background is provided, it creates a new transparent canvas.
    3.  It pastes the rendered text surface onto the canvas at the specified coordinates.
    4.  Crucially, it adjusts the coordinates of all character bounding boxes by the placement offset, ensuring they are accurate relative to the final canvas.
