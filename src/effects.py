"""
This module contains functions for applying various image effects and augmentations.
"""

from PIL import Image, ImageFilter

def apply_ink_bleed(image: Image.Image, radius: float) -> Image.Image:
    """
    Simulates an ink bleed effect by applying a Gaussian blur.

    Args:
        image: The source PIL Image.
        radius: The radius of the Gaussian blur.

    Returns:
        The processed PIL Image with the blur applied.
    """
    if radius <= 0:
        return image
    
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

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
