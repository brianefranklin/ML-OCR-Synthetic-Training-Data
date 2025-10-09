#!/usr/bin/env python3
"""Debug script to save both images from the test and compare them."""

import sys
from pathlib import Path
import random
import numpy as np
import subprocess
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generator import OCRDataGenerator
from PIL import Image
import json
import shutil

# Setup directories
tmp_path = Path("/tmp/debug_regen_test")
tmp_path.mkdir(exist_ok=True)
input_dir = tmp_path / "input"
output_dir = tmp_path / "output"
fonts_dir = input_dir / "fonts"
text_dir = input_dir / "text"

# Clean and recreate
if output_dir.exists():
    shutil.rmtree(output_dir)
fonts_dir.mkdir(parents=True, exist_ok=True)
text_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir()

# Create corpus
corpus_path = text_dir / "corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write("The quick brown fox jumps over the lazy dog. " * 50)

# Copy a font
source_font_dir = Path(__file__).parent / "data.nosync" / "fonts"
font_files = list(source_font_dir.glob("**/*.ttf"))
shutil.copy(font_files[0], fonts_dir)

# Create batch config
batch_config_data = {
    "total_images": 1,
    "seed": 42,
    "batches": [
        {
            "name": "regeneration_test_batch",
            "proportion": 1.0,
            "text_direction": "left_to_right",
            "effect_type": "none",
            "corpus_file": str(corpus_path)
        }
    ]
}

batch_config_path = tmp_path / "batch_config.yaml"
with open(batch_config_path, "w") as f:
    yaml.dump(batch_config_data, f)

# Run first generation via subprocess
print("=== FIRST GENERATION (subprocess) ===")
script_path = Path(__file__).parent / "src" / "main.py"
command = [
    "python3", str(script_path),
    "--batch-config", str(batch_config_path),
    "--fonts-dir", str(fonts_dir),
    "--output-dir", str(output_dir)
]

result = subprocess.run(command, capture_output=True, text=True, check=False)
if result.returncode != 0:
    print(f"First generation failed: {result.stderr}")
    sys.exit(1)

# Load the JSON and image
json_files = list(output_dir.glob("image_*.json"))
with open(json_files[0], 'r') as f:
    first_pass_data = json.load(f)

params = first_pass_data['generation_params']
print(f"First pass params: {params}")

original_image_path = output_dir / first_pass_data['image_file']
original_image = Image.open(original_image_path)
original_image.save("/tmp/image1_subprocess.png")
print(f"Saved first generation to /tmp/image1_subprocess.png")

# Run second generation directly
print("\n=== SECOND GENERATION (direct call) ===")
generator = OCRDataGenerator(font_files=[params['font_path']], background_images=[])

# Get text_offset from first pass for deterministic regeneration
text_offset = tuple(first_pass_data.get('text_placement', [None, None]))
if text_offset[0] is None:
    text_offset = None
print(f"Using text_offset: {text_offset}")

regen_image, regen_metadata, _, _ = generator.generate_image(
    text=params['text'],
    font_path=params['font_path'],
    font_size=params['font_size'],
    direction=params['text_direction'],
    seed=params['seed'],
    canvas_size=first_pass_data['canvas_size'],
    text_offset=text_offset,
    augmentations=params.get('augmentations'),
    curve_type=params.get('curve_type', 'none'),
    curve_intensity=params.get('curve_intensity', 0.0),
    overlap_intensity=params.get('overlap_intensity', 0.0),
    ink_bleed_intensity=params.get('ink_bleed_intensity', 0.0),
    effect_type=params.get('effect_type', 'none'),
    effect_depth=params.get('effect_depth', 0.5),
    light_azimuth=params.get('light_azimuth', 135.0),
    light_elevation=params.get('light_elevation', 45.0),
    text_color_mode=params.get('text_color_mode', 'uniform'),
    color_palette=params.get('color_palette', 'realistic_dark'),
    custom_colors=params.get('custom_colors'),
    background_color=params.get('background_color', 'auto')
)

regen_image.save("/tmp/image2_direct.png")
print(f"Saved second generation to /tmp/image2_direct.png")

# Compare
arr1 = np.array(original_image)
arr2 = np.array(regen_image)

print("\n=== COMPARISON ===")
print(f"Image 1: {original_image.size} {original_image.mode}")
print(f"Image 2: {regen_image.size} {regen_image.mode}")

if np.array_equal(arr1, arr2):
    print("✓ Images are IDENTICAL")
else:
    print("✗ Images are DIFFERENT")
    diff = np.where(arr1 != arr2)
    if len(diff[0]) > 0:
        n_diff = len(diff[0])
        total_pixels = arr1.shape[0] * arr1.shape[1] * arr1.shape[2]
        print(f"  Different values: {n_diff} / {total_pixels} ({100*n_diff/total_pixels:.2f}%)")

        # Save difference image
        diff_img = np.abs(arr1.astype(int) - arr2.astype(int)).astype(np.uint8)
        Image.fromarray(diff_img).save("/tmp/image_diff.png")
        print("  Saved difference to /tmp/image_diff.png")
