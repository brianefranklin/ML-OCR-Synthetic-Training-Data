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
    apply_erosion_dilation,
    apply_block_shadow,
    apply_cutout
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

def test_block_shadow_is_applied():
    """Tests that a block shadow is applied correctly."""
    image = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 10, 80, 40), fill="black")
    original_array = np.array(image)

    # Apply block shadow
    shadow_image = apply_block_shadow(image, offset=(5, 5), radius=2, color=(0, 0, 0, 128))
    shadow_array = np.array(shadow_image)

    assert not np.array_equal(original_array, shadow_array)

def test_cutout_is_applied():
    """Tests that applying cutout modifies the source image."""
    image = Image.new("RGBA", (100, 50), "white")
    original_array = np.array(image)

    # Apply cutout
    cutout_image = apply_cutout(image, cutout_size=(20, 10))
    cutout_array = np.array(cutout_image)

    assert not np.array_equal(original_array, cutout_array)

def test_add_noise_deterministic():
    """Tests that add_noise produces deterministic results with same numpy seed."""
    image1 = Image.new("RGBA", (100, 50), "white")
    image2 = Image.new("RGBA", (100, 50), "white")

    # Apply noise with same seed
    np.random.seed(42)
    noisy1 = add_noise(image1, amount=0.1)

    np.random.seed(42)
    noisy2 = add_noise(image2, amount=0.1)

    # Should produce identical results
    assert np.array_equal(np.array(noisy1), np.array(noisy2))

def test_add_noise_respects_amount_exactly():
    """Tests that add_noise modifies the exact number of pixels specified by amount."""
    image = Image.new("L", (100, 50), 128)  # Grayscale with value 128
    original_array = np.array(image)

    amount = 0.1
    noisy_image = add_noise(image, amount=amount)
    noisy_array = np.array(noisy_image)

    # Count pixels that changed
    changed_pixels = np.sum(original_array != noisy_array)
    expected_pixels = int(amount * image.width * image.height)

    # Should match exactly
    assert changed_pixels == expected_pixels

def test_add_noise_only_uses_black_and_white():
    """Tests that add_noise only sets pixels to 0 (black) or 255 (white)."""
    image = Image.new("L", (100, 50), 128)  # Grayscale with value 128

    noisy_image = add_noise(image, amount=0.2)
    noisy_array = np.array(noisy_image)

    # Get all unique values in the noisy image
    unique_values = np.unique(noisy_array)

    # Should only contain 0, 128 (original), and 255
    for val in unique_values:
        assert val in [0, 128, 255], f"Found unexpected pixel value: {val}"

    # Should have at least some 0s and 255s (noise was applied)
    assert 0 in unique_values or 255 in unique_values
