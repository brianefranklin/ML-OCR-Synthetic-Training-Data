"""
Tests for image effects.
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np
from src.effects import (
    apply_ink_bleed, 
    apply_drop_shadow, 
    add_noise, 
    apply_blur, 
    apply_brightness_contrast,
    apply_erosion_dilation
)

def test_ink_bleed_is_applied():
    """Tests that applying ink bleed modifies the source image."""
    # Create an image with a black box on a transparent background to provide edges
    original_image = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(original_image)
    draw.rectangle((20, 10, 80, 40), fill="black")
    original_array = np.array(original_image)

    # Apply the ink bleed effect
    bled_image = apply_ink_bleed(original_image, radius=2)
    bled_array = np.array(bled_image)

    # The images should not be identical
    assert not np.array_equal(original_array, bled_array)

def test_drop_shadow_is_applied():
    """Tests that applying a drop shadow modifies the source image."""
    # Create an image with a black box on a transparent background
    original_image = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(original_image)
    draw.rectangle((20, 10, 80, 40), fill="black")
    original_array = np.array(original_image)

    # Apply the drop shadow effect
    shadow_image = apply_drop_shadow(original_image, offset=(5, 5), radius=2, color=(0, 0, 0, 128))
    shadow_array = np.array(shadow_image)

    # The images should not be identical
    assert not np.array_equal(original_array, shadow_array)

def test_add_noise_is_applied():
    """Tests that adding noise modifies the source image."""
    image = Image.new("RGBA", (100, 50), "white")
    original_array = np.array(image)

    # Apply noise
    noisy_image = add_noise(image, amount=0.1)
    noisy_array = np.array(noisy_image)

    assert not np.array_equal(original_array, noisy_array)

def test_blur_is_applied():
    """Tests that applying blur modifies the source image."""
    image = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 80, 40), fill="black")
    original_array = np.array(image)

    # Apply blur
    blurred_image = apply_blur(image, radius=2)
    blurred_array = np.array(blurred_image)

    assert not np.array_equal(original_array, blurred_array)

def test_brightness_contrast_is_applied():
    """Tests that adjusting brightness and contrast modifies the source image."""
    image = Image.new("RGB", (100, 50), "gray")
    original_array = np.array(image)

    # Apply brightness and contrast
    adjusted_image = apply_brightness_contrast(image, brightness_factor=1.5, contrast_factor=1.5)
    adjusted_array = np.array(adjusted_image)

    assert not np.array_equal(original_array, adjusted_array)

def test_erosion_dilation_is_applied():
    """Tests that erosion and dilation modify the image as expected."""
    image = Image.new("L", (100, 50), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 80, 40), fill="black")
    original_black_pixels = np.sum(np.array(image) == 0)

    # Test Erosion
    eroded_image = apply_erosion_dilation(image, mode='erode', kernel_size=3)
    eroded_black_pixels = np.sum(np.array(eroded_image) == 0)
    assert eroded_black_pixels < original_black_pixels

    # Test Dilation
    dilated_image = apply_erosion_dilation(image, mode='dilate', kernel_size=3)
    dilated_black_pixels = np.sum(np.array(dilated_image) == 0)
    assert dilated_black_pixels > original_black_pixels
