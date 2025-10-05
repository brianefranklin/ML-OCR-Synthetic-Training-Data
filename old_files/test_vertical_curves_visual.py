#!/usr/bin/env python3
"""Visual test for vertical curved text."""
import sys
import os
from pathlib import Path
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).parent / "src"))
from main import OCRDataGenerator

# Find a font
font_dir = Path("data/fonts")
font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
if not font_files:
    print("No fonts found!")
    sys.exit(1)

font_path = str(font_files[0])
generator = OCRDataGenerator([font_path])
font = ImageFont.truetype(font_path, size=40)

# Create output directory
output_dir = Path("test_output/vertical_curves")
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating vertical curved text examples...\n")

# Test cases
test_cases = [
    ("TopToBottom", "Test", "arc", 0.3, "top_to_bottom"),
    ("TopToBottom", "Vertical", "sine", 0.4, "top_to_bottom"),
    ("BottomToTop", "Test", "arc", 0.3, "bottom_to_top"),
    ("BottomToTop", "Vertical", "sine", 0.4, "bottom_to_top"),
]

for i, (name, text, curve_type, intensity, direction) in enumerate(test_cases, 1):
    if direction == "top_to_bottom":
        img, char_boxes = generator.render_top_to_bottom_curved(
            text, font, curve_type=curve_type, curve_intensity=intensity
        )
    else:
        img, char_boxes = generator.render_bottom_to_top_curved(
            text, font, curve_type=curve_type, curve_intensity=intensity
        )

    filename = f"{i}_{direction}_{curve_type}_{intensity}.png"
    output_path = output_dir / filename
    img.save(output_path)

    print(f"{i}. {name} - '{text}' ({curve_type}, intensity={intensity})")
    print(f"   Size: {img.width}x{img.height}")
    print(f"   Chars: {len(char_boxes)}")
    print(f"   Saved: {output_path}\n")

print("✓ All vertical curve examples generated!")
