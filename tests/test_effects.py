"""
Tests for image effects.
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np
from src.effects import apply_ink_bleed, apply_drop_shadow

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
