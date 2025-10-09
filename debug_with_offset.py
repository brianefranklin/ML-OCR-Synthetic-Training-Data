#!/usr/bin/env python3
"""Test generation with explicit text_offset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generator import OCRDataGenerator
import numpy as np

# Find font
font_dir = Path(__file__).parent / "data.nosync" / "fonts"
font_files = list(font_dir.glob("**/*.ttf"))
font_path = str(font_files[0])

params = {
    'text': 'he qu',
    'font_path': font_path,
    'font_size': 31,
    'direction': 'left_to_right',
    'seed': 42,
    'canvas_size': (365, 166),
    'text_offset': (85, 67),
    'augmentations': {
        'perspective_transform': True,
        'elastic_distortion': True,
        'background': True,
        'brightness_contrast': True
    }
}

generator = OCRDataGenerator(font_files=[font_path], background_images=[])

# Generate twice
print("=== FIRST ===")
img1, _, _, _ = generator.generate_image(**params)
print(f"Size: {img1.size}")

print("\n=== SECOND ===")
img2, _, _, _ = generator.generate_image(**params)
print(f"Size: {img2.size}")

# Compare
arr1 = np.array(img1)
arr2 = np.array(img2)

print("\n=== COMPARISON ===")
if np.array_equal(arr1, arr2):
    print("✓✓✓ IDENTICAL ✓✓✓")
else:
    print("✗✗✗ DIFFERENT ✗✗✗")
    diff = np.where(arr1 != arr2)
    n_diff = len(diff[0])
    total = arr1.size
    print(f"  Different: {n_diff} / {total} ({100*n_diff/total:.2f}%)")
