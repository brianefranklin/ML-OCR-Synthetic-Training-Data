# API Reference: `effects.py`

This module contains functions for applying various image effects, typically to the rendered text surface *before* it is placed on the final canvas.

### `apply_ink_bleed(image, radius)`
- **Description:** Simulates ink spreading on paper.
- **Implementation:** Applies a `GaussianBlur` filter.

### `apply_drop_shadow(image, offset, radius, color)`
- **Description:** Adds a per-glyph drop shadow.
- **Implementation:** Creates a blurred, offset copy of the text's alpha channel.

### `apply_block_shadow(image, offset, radius, color)`
- **Description:** Adds a soft shadow to the entire text block.
- **Implementation:** Similar to drop shadow but applied to the whole text surface.

### `add_noise(image, amount)`
- **Description:** Adds salt-and-pepper noise.
- **Implementation:** Randomly sets pixels to black or white.

### `apply_blur(image, radius)`
- **Description:** Applies a standard Gaussian blur.
- **Implementation:** Uses `ImageFilter.GaussianBlur`.

### `apply_brightness_contrast(image, brightness_factor, contrast_factor)`
- **Description:** Adjusts brightness and contrast.
- **Implementation:** Uses `ImageEnhance.Brightness` and `ImageEnhance.Contrast`.

### `apply_erosion_dilation(image, mode, kernel_size)`
- **Description:** Makes text thinner or thicker.
- **Implementation:** Uses `cv2.erode` or `cv2.dilate`.

### `apply_cutout(image, cutout_size)`
- **Description:** Erases a random rectangular portion of the image.
- **Implementation:** Draws a filled rectangle at a random position.
