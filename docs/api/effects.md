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
- **Description:** Adds salt-and-pepper noise to an image using vectorized NumPy operations.
- **Implementation:** NumPy-optimized implementation that randomly sets pixels to black (0) or white (255).
- **Performance:** 10-50x faster than loop-based approaches through vectorization.
- **Parameters:**
  - `image` (PIL.Image.Image): The source image (grayscale or multi-channel).
  - `amount` (float): Proportion of pixels to affect (0.0 to 1.0).
- **Returns:** PIL.Image.Image with salt-and-pepper noise applied.
- **Key Features:**
  - **Deterministic**: Uses `np.random` for reproducible results with `np.random.seed()`.
  - **Exact Count**: Modifies exactly `floor(amount * width * height)` pixels without duplicates.
  - **Multi-channel Support**: Works with grayscale, RGB, and RGBA images.
- **Example:**
  ```python
  import numpy as np
  from PIL import Image
  from src.effects import add_noise

  # Load image
  img = Image.open("text.png")

  # Add 10% noise with deterministic seed
  np.random.seed(42)
  noisy_img = add_noise(img, amount=0.1)
  ```

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
