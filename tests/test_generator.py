import pytest
from PIL import Image
import numpy as np
from src.generator import OCRDataGenerator


def test_ocr_data_generator_initialization():
    """Tests that the OCRDataGenerator class can be initialized.""" 
    try:
        generator = OCRDataGenerator()
    except Exception as e:
        pytest.fail(f"OCRDataGenerator initialization failed: {e}")
    
    assert generator is not None

def test_plan_generation_creates_truth_data():
    """Tests that the plan_generation method returns a dictionary with essential truth data."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"

    plan = generator.plan_generation(text=text, font_path=font_path, direction=direction)

    assert isinstance(plan, dict)
    assert plan["text"] == text
    assert plan["font_path"] == font_path
    assert plan["direction"] == direction
    assert "seed" in plan
    assert isinstance(plan["seed"], int)

def test_generate_from_plan_ltr_places_on_canvas():
    """Tests that generate_from_plan creates a valid image and bboxes on a larger canvas."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"

    plan = generator.plan_generation(text=text, font_path=font_path, direction=direction)
    
    image, bboxes = generator.generate_from_plan(plan)
    
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0
    
    # Check bounding boxes
    assert isinstance(bboxes, list)
    assert len(bboxes) == len(text)
    
    # Check that the bounding box has been offset from the original margin (10)
    bbox_h = bboxes[0]
    assert bbox_h["char"] == "h"
    assert bbox_h["x0"] > 10 # This will fail until placement is integrated
    assert bbox_h["y1"] > bbox_h["y0"]

def test_generate_from_plan_rtl():
    """Tests that generate_from_plan creates a valid image for RTL text."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    plan = generator.plan_generation(text="שלום", font_path=font_path, direction="right_to_left")
    
    image, bboxes = generator.generate_from_plan(plan)
    
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0

def test_generate_from_plan_ttb():
    """Tests that generate_from_plan creates a valid image and bboxes for TTB text."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "VERTICAL"
    plan = generator.plan_generation(text=text, font_path=font_path, direction="top_to_bottom")
    
    image, bboxes = generator.generate_from_plan(plan)
    
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0
    assert len(bboxes) == len(text)

def test_generate_from_plan_btt():
    """Tests that generate_from_plan creates a valid image and bboxes for BTT text."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "VERTICAL"
    plan = generator.plan_generation(text=text, font_path=font_path, direction="bottom_to_top")
    
    image, bboxes = generator.generate_from_plan(plan)
    
    assert isinstance(image, Image.Image)
    assert image.width > 0
    assert image.height > 0
    assert len(bboxes) == len(text)

def test_glyph_overlap_reduces_width():
    """Tests that a glyph_overlap_intensity > 0 reduces the final image width."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"

    # Plan with overlap
    plan_overlap = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, glyph_overlap_intensity=0.5
    )
    image_overlap, _ = generator.generate_from_plan(plan_overlap)

    # Plan without overlap (control)
    plan_no_overlap = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, glyph_overlap_intensity=0.0
    )
    # Use the same seed for a fair comparison of canvas size, etc.
    plan_no_overlap["seed"] = plan_overlap["seed"]
    image_no_overlap, _ = generator.generate_from_plan(plan_no_overlap)

    assert image_overlap.width < image_no_overlap.width

def test_vertical_glyph_overlap_reduces_height():
    """Tests that glyph_overlap_intensity > 0 reduces the height of vertical text."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "VERTICAL"
    direction = "top_to_bottom"

    # Render with overlap
    image_overlap, _ = generator._render_text(
        text, font_path, direction, glyph_overlap_intensity=0.5
    )

    # Render without overlap (control)
    image_no_overlap, _ = generator._render_text(
        text, font_path, direction, glyph_overlap_intensity=0.0
    )

    assert image_overlap.height < image_no_overlap.height

def test_ink_bleed_integration():
    """Tests that the ink_bleed effect is correctly integrated into the generator."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"

    # Plan with ink bleed
    plan_bleed = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, ink_bleed_radius=2.0
    )
    image_bleed, _ = generator.generate_from_plan(plan_bleed)

    # Plan without ink bleed (control)
    plan_no_bleed = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, ink_bleed_radius=0.0
    )
    plan_no_bleed["seed"] = plan_bleed["seed"]
    image_no_bleed, _ = generator.generate_from_plan(plan_no_bleed)

    # The images should not be identical
    assert not np.array_equal(np.array(image_bleed), np.array(image_no_bleed))
