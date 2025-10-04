#!/usr/bin/env python3
"""Visual demonstration of glyph overlap feature."""
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
output_dir = Path("test_output/overlap_examples")
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating glyph overlap examples...\n")

# Test different overlap intensities
text = "Overlapping Text"
overlap_levels = [0.0, 0.25, 0.5, 0.75]

for i, overlap in enumerate(overlap_levels, 1):
    img, char_boxes = generator.render_left_to_right(
        text, font, overlap_intensity=overlap, ink_bleed_intensity=0.0
    )

    filename = f"{i}_overlap_{int(overlap*100)}pct.png"
    output_path = output_dir / filename
    img.save(output_path)

    print(f"{i}. Overlap {int(overlap*100)}%")
    print(f"   Size: {img.width}x{img.height}")
    print(f"   Width reduction: {(1 - img.width/215)*100:.1f}% (from baseline)")
    print(f"   Saved: {output_path}\n")

# Test with ink bleed
print("Testing ink bleed effect:")
img, _ = generator.render_left_to_right(
    text, font, overlap_intensity=0.5, ink_bleed_intensity=0.5
)
output_path = output_dir / "overlap_50pct_ink_bleed.png"
img.save(output_path)
print(f"   Overlap 50% + Ink Bleed 50%")
print(f"   Saved: {output_path}\n")

# Test vertical text with overlap
print("Testing vertical overlap:")
vert_text = "Vertical"
img, _ = generator.render_top_to_bottom(
    vert_text, font, overlap_intensity=0.6, ink_bleed_intensity=0.0
)
output_path = output_dir / "vertical_overlap_60pct.png"
img.save(output_path)
print(f"   Vertical TTB with 60% overlap")
print(f"   Size: {img.width}x{img.height}")
print(f"   Saved: {output_path}\n")

# Test curved text with overlap
print("Testing curved + overlap:")
curved_text = "Curved"
img, _ = generator.render_curved_text(
    curved_text, font, curve_type='arc', curve_intensity=0.4,
    overlap_intensity=0.5, ink_bleed_intensity=0.0
)
output_path = output_dir / "curved_arc_overlap_50pct.png"
img.save(output_path)
print(f"   Arc curve + 50% overlap")
print(f"   Size: {img.width}x{img.height}")
print(f"   Saved: {output_path}\n")

print("✓ All glyph overlap examples generated!")
print(f"✓ Check {output_dir} for visual results")
