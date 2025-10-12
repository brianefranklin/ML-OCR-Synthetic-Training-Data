"""
Tests for image augmentations.
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np
import random
from src.augmentations import (
    apply_rotation, 
    apply_perspective_warp, 
    apply_elastic_distortion,
    apply_grid_distortion,
    apply_optical_distortion
)

def test_rotation_ltr_updates_bboxes():
    """Tests that rotation correctly transforms bounding boxes for LTR text."""
    # Create a non-symmetrical image and a bounding box within it
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    bboxes = [
        {"char": "a", "x0": 50, "y0": 30, "x1": 80, "y1": 70}
    ]

    # Apply rotation
    rotated_image, rotated_bboxes = apply_rotation(image, bboxes, angle=90)

    # Check that the image size has changed (due to expand=True)
    assert rotated_image.size != image.size

    # Check that the bounding box has been transformed
    original_bbox = bboxes[0]
    rotated_bbox = rotated_bboxes[0]
    assert original_bbox["x0"] != rotated_bbox["x0"]
    assert original_bbox["y0"] != rotated_bbox["y0"]

def test_rotation_rtl_updates_bboxes():
    """Tests that rotation correctly transforms bounding boxes for RTL text."""
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    # Simulate a box on the right side of the image
    bboxes = [
        {"char": "ش", "x0": 120, "y0": 30, "x1": 150, "y1": 70}
    ]
    rotated_image, rotated_bboxes = apply_rotation(image, bboxes, angle=45)
    assert rotated_image.size != image.size
    assert bboxes[0]["x0"] != rotated_bboxes[0]["x0"]

def test_rotation_ttb_updates_bboxes():
    """Tests that rotation correctly transforms bounding boxes for TTB text."""
    image = Image.new("RGBA", (100, 200), (0, 0, 0, 0))
    # Simulate a tall box for vertical text
    bboxes = [
        {"char": "V", "x0": 30, "y0": 50, "x1": 70, "y1": 80}
    ]
    rotated_image, rotated_bboxes = apply_rotation(image, bboxes, angle=-30)
    assert rotated_image.size != image.size
    assert bboxes[0]["y0"] != rotated_bboxes[0]["y0"]

def test_perspective_warp_is_deterministic():
    """Tests that perspective warp with fixed points is deterministic."""
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    bboxes = [
        {"char": "a", "x0": 50, "y0": 30, "x1": 80, "y1": 70}
    ]
    
    # Define a fixed, non-random perspective transform
    dst_points = np.float32([
        [10, 10],  # Top-left
        [190, 0],   # Top-right
        [180, 90],  # Bottom-right
        [0, 80]     # Bottom-left
    ])

    _, warped_bboxes = apply_perspective_warp(image, bboxes, magnitude=0, dst_points=dst_points)

    # Pre-calculated expected coordinates for the given transform
    expected_bbox = {'x0': 40, 'y0': 29, 'x1': 68, 'y1': 60}
    warped_bbox = warped_bboxes[0]

    assert warped_bbox['x0'] == expected_bbox['x0']
    assert warped_bbox['y0'] == expected_bbox['y0']
    assert warped_bbox['x1'] == expected_bbox['x1']
    assert warped_bbox['y1'] == expected_bbox['y1']

def test_elastic_distortion_is_applied():
    """Tests that elastic distortion modifies the source image."""
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, 150, 70), fill="black")
    original_array = np.array(image)

    # Apply distortion
    distorted_image, _ = apply_elastic_distortion(image, [], alpha=34, sigma=4)
    distorted_array = np.array(distorted_image)

    assert not np.array_equal(original_array, distorted_array)

def test_grid_distortion_is_applied():
    """Tests that grid distortion modifies the source image."""
    random.seed(0)
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, 150, 70), fill="black")
    original_array = np.array(image)

    # Apply distortion
    distorted_image, _ = apply_grid_distortion(image, [], num_steps=5, distort_limit=10)
    distorted_array = np.array(distorted_image)

    assert not np.array_equal(original_array, distorted_array)

def test_optical_distortion_is_applied():
    """Tests that optical distortion modifies the source image."""
    image = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, 150, 70), fill="black")
    original_array = np.array(image)

    # Apply distortion
    distorted_image, _ = apply_optical_distortion(image, [], distort_limit=0.1)
    distorted_array = np.array(distorted_image)

    assert not np.array_equal(original_array, distorted_array)
