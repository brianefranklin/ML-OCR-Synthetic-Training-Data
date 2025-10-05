#!/usr/bin/env python3
"""
Test script to verify bbox accuracy for curved text with rotation.
Generates a curved text image and visualizes the bboxes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import OCRDataGenerator
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def test_bbox_accuracy():
    """Test bbox accuracy by rendering curved text and drawing bboxes."""
    # Load a font
    font_path = "data/fonts/NotoSans-Regular.ttf"
    if not os.path.exists(font_path):
        # Try to find any font
        font_files = [f for f in os.listdir("data/fonts") if f.endswith(('.ttf', '.otf'))]
        if font_files:
            font_path = os.path.join("data/fonts", font_files[0])
        else:
            print("No fonts found!")
            return
    else:
        font_files = [font_path]

    generator = OCRDataGenerator(font_files)
    font = ImageFont.truetype(font_path, size=40)

    # Test with different curve types and intensities
    test_cases = [
        ("Hello World", "arc", 0.3),
        ("Testing 123", "arc", 0.5),
        ("Curved Text", "sine", 0.4),
    ]

    os.makedirs("test_output/bbox_accuracy", exist_ok=True)

    for i, (text, curve_type, intensity) in enumerate(test_cases):
        print(f"\nTest {i+1}: '{text}' - {curve_type} @ {intensity}")

        # Generate curved text
        img, char_boxes = generator.render_curved_text(text, font, curve_type, intensity)

        # Draw bboxes on image
        img_with_boxes = img.copy()
        draw = ImageDraw.Draw(img_with_boxes)

        for char_box in char_boxes:
            bbox = char_box.bbox
            # Draw bbox as red rectangle
            draw.rectangle(bbox, outline='red', width=2)

            # Calculate bbox area
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            print(f"  '{char_box.char}': bbox={bbox}, size={width:.1f}x{height:.1f}, area={area:.0f}")

            # Verify bbox is reasonable
            assert width > 0, f"Width should be positive for '{char_box.char}'"
            assert height > 0, f"Height should be positive for '{char_box.char}'"
            assert bbox[0] >= 0, f"x0 should be non-negative"
            assert bbox[1] >= 0, f"y0 should be non-negative"
            assert bbox[2] <= img.width, f"x1 should be within image"
            assert bbox[3] <= img.height, f"y1 should be within image"

        # Save visualization
        output_path = f"test_output/bbox_accuracy/test_{i+1}_{curve_type}_{intensity}.png"
        img_with_boxes.save(output_path)
        print(f"  Saved: {output_path}")

    print("\n✓ All bbox accuracy tests passed!")

if __name__ == "__main__":
    test_bbox_accuracy()
