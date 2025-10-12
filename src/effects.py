"""
This module contains functions for applying various image effects and augmentations.
"""

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import cv2

def apply_ink_bleed(image: Image.Image, radius: float) -> Image.Image:
    """
    Simulates an ink bleed effect by applying a Gaussian blur.

    Args:
        image: The source PIL Image.
        radius: The radius of the Gaussian blur.

    Returns:
        The processed PIL Image with the blur applied.
    """
    return apply_blur(image, radius)

def apply_drop_shadow(
    image: Image.Image, 
    offset: tuple[int, int], 
    radius: float, 
    color: tuple[int, int, int, int]
) -> Image.Image:
    """
    Applies a drop shadow effect to an image.

    Args:
        image: The source PIL Image (must be RGBA).
        offset: A tuple (x, y) for how far to offset the shadow.
        radius: The radius of the Gaussian blur for the shadow.
        color: The color of the shadow as an (R, G, B, A) tuple.

    Returns:
        A new PIL Image with the drop shadow applied.
    """
    # Create a new image for the shadow, using the alpha channel of the original
    shadow = Image.new("RGBA", image.size, color)
    shadow.putalpha(image.split()[3]) # Use original image's alpha

    # Blur the shadow
    if radius > 0:
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius))

    # Create a new canvas large enough for the image and shadow
    new_width = image.width + abs(offset[0]) + int(radius * 2)
    new_height = image.height + abs(offset[1]) + int(radius * 2)
    new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

    # Paste the shadow, then the original image on top
    shadow_pos = (int(radius) + max(0, offset[0]), int(radius) + max(0, offset[1]))
    image_pos = (int(radius) - min(0, offset[0]), int(radius) - min(0, offset[1]))

    new_image.paste(shadow, shadow_pos, shadow)
    new_image.paste(image, image_pos, image)

    return new_image

def add_noise(image: Image.Image, amount: float) -> Image.Image:
    """
    Adds salt-and-pepper noise to an image.

    Args:
        image: The source PIL Image.
        amount: The proportion of pixels to be affected by noise (0.0 to 1.0).

    Returns:
        The processed PIL Image with noise applied.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    num_pixels = int(amount * w * h)

    for _ in range(num_pixels):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        
        # Salt or pepper
        if random.random() > 0.5:
            img_np[y, x] = 255
        else:
            img_np[y, x] = 0

    return Image.fromarray(img_np)

def apply_blur(image: Image.Image, radius: float) -> Image.Image:
    """
    Applies a Gaussian blur to an image.

    Args:
        image: The source PIL Image.
        radius: The radius of the Gaussian blur.

    Returns:
        The processed PIL Image with the blur applied.
    """
    if radius <= 0:
        return image
    
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_brightness_contrast(image: Image.Image, brightness_factor: float, contrast_factor: float) -> Image.Image:
    """
    Adjusts the brightness and contrast of an image.

    Args:
        image: The source PIL Image.
        brightness_factor: The brightness enhancement factor. 1.0 is original.
        contrast_factor: The contrast enhancement factor. 1.0 is original.

    Returns:
        The processed PIL Image with adjustments applied.
    """
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    
    return image

def apply_erosion_dilation(image: Image.Image, mode: str, kernel_size: int) -> Image.Image:
    """
    Applies erosion or dilation to an image.

    Args:
        image: The source PIL Image.
        mode: 'erode' or 'dilate'.
        kernel_size: The size of the kernel for the operation.

    Returns:
        The processed PIL Image.
    """
    img_np = np.array(image)
    # Invert the image so the object is white and background is black
    img_np = cv2.bitwise_not(img_np)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if mode == 'erode':
        result_np = cv2.erode(img_np, kernel, iterations=1)
    elif mode == 'dilate':
        result_np = cv2.dilate(img_np, kernel, iterations=1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Invert back
    result_np = cv2.bitwise_not(result_np)

    return Image.fromarray(result_np)
