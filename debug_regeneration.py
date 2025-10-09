#!/usr/bin/env python3
"""Debug script to identify where determinism breaks in regeneration."""

import sys
from pathlib import Path
import random
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generator import OCRDataGenerator
from PIL import Image

# Test parameters (from the failing test)
font_path = "/tmp/test_font.ttf"  # Will use a real font path
text = "he qu"
font_size = 31
direction = "left_to_right"
seed = 42
canvas_size = (365, 166)
augmentations = {
    'perspective_transform': True,
    'elastic_distortion': True,
    'background': True,
    'brightness_contrast': True
}

# Find a font to use
font_dir = Path(__file__).parent / "data.nosync" / "fonts"
font_files = list(font_dir.glob("**/*.ttf"))
if not font_files:
    print("No fonts found!")
    sys.exit(1)

font_path = str(font_files[0])
print(f"Using font: {font_path}")

# Create generator
generator = OCRDataGenerator(font_files=[font_path], background_images=[])

# Generate twice with same seed
print("\n=== FIRST GENERATION ===")
random.seed(42)
np.random.seed(42)
image1, metadata1, _, _ = generator.generate_image(
    text=text,
    font_path=font_path,
    font_size=font_size,
    direction=direction,
    seed=seed,
    canvas_size=canvas_size,
    augmentations=augmentations
)

print(f"Image 1 size: {image1.size}")
print(f"Image 1 mode: {image1.mode}")
# Print first few pixels
arr1 = np.array(image1)
print(f"Image 1 first pixel: {arr1[0, 0]}")
print(f"Image 1 shape: {arr1.shape}")

print("\n=== SECOND GENERATION ===")
random.seed(42)
np.random.seed(42)
image2, metadata2, _, _ = generator.generate_image(
    text=text,
    font_path=font_path,
    font_size=font_size,
    direction=direction,
    seed=seed,
    canvas_size=canvas_size,
    augmentations=augmentations
)

print(f"Image 2 size: {image2.size}")
print(f"Image 2 mode: {image2.mode}")
arr2 = np.array(image2)
print(f"Image 2 first pixel: {arr2[0, 0]}")
print(f"Image 2 shape: {arr2.shape}")

# Compare
print("\n=== COMPARISON ===")
if np.array_equal(arr1, arr2):
    print("✓ Images are IDENTICAL (deterministic generation working!)")
else:
    print("✗ Images are DIFFERENT (non-deterministic!)")
    # Find first difference
    diff = np.where(arr1 != arr2)
    if len(diff[0]) > 0:
        y, x, c = diff[0][0], diff[1][0], diff[2][0]
        print(f"  First difference at pixel ({x}, {y}) channel {c}:")
        print(f"    Image 1: {arr1[y, x]}")
        print(f"    Image 2: {arr2[y, x]}")

        # Count total differences
        n_diff = len(diff[0])
        total_pixels = arr1.shape[0] * arr1.shape[1] * arr1.shape[2]
        print(f"  Total different values: {n_diff} / {total_pixels} ({100*n_diff/total_pixels:.2f}%)")
