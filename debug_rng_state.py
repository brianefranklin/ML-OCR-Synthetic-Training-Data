#!/usr/bin/env python3
"""Debug RNG state to understand the difference."""

import sys
from pathlib import Path
import random
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generator import OCRDataGenerator

# Find a font
font_dir = Path(__file__).parent / "data.nosync" / "fonts"
font_files = list(font_dir.glob("**/*.ttf"))
font_path = str(font_files[0])

# Test parameters
text = "he qu"
font_size = 31
direction = "left_to_right"
seed = 42
canvas_size = (365, 166)
augmentations = {
    'brightness_contrast': True
}

# Create generator
generator = OCRDataGenerator(font_files=[font_path], background_images=[])

# Test 1: NO external seed before generate_image
print("=== TEST 1: No external seed ===")
image1, _, _, _ = generator.generate_image(
    text=text,
    font_path=font_path,
    font_size=font_size,
    direction=direction,
    seed=seed,
    canvas_size=canvas_size,
    augmentations=augmentations
)
print(f"Random state after: {random.getstate()[1][:5]}")

# Test 2: External seed before generate_image (like in my debug script)
print("\n=== TEST 2: External seed before generate_image ===")
random.seed(42)
np.random.seed(42)
image2, _, _, _ = generator.generate_image(
    text=text,
    font_path=font_path,
    font_size=font_size,
    direction=direction,
    seed=seed,
    canvas_size=canvas_size,
    augmentations=augmentations
)
print(f"Random state after: {random.getstate()[1][:5]}")

# Test 3: Some random calls before generate_image (simulating main.py)
print("\n=== TEST 3: Random calls before generate_image ===")
random.seed(100)  # Different seed initially
for _ in range(10):
    random.random()  # Consume some random state

image3, _, _, _ = generator.generate_image(
    text=text,
    font_path=font_path,
    font_size=font_size,
    direction=direction,
    seed=seed,  # generate_image sets its own seed
    canvas_size=canvas_size,
    augmentations=augmentations
)
print(f"Random state after: {random.getstate()[1][:5]}")

# Compare
arr1 = np.array(image1)
arr2 = np.array(image2)
arr3 = np.array(image3)

print("\n=== COMPARISONS ===")
print(f"Image 1 vs 2 (no ext seed vs ext seed): {np.array_equal(arr1, arr2)}")
print(f"Image 1 vs 3 (no ext seed vs random before): {np.array_equal(arr1, arr3)}")
print(f"Image 2 vs 3 (ext seed vs random before): {np.array_equal(arr2, arr3)}")
