import pytest
from PIL import Image, ImageDraw
import numpy as np
from src.generator import OCRDataGenerator
from src.background_manager import BackgroundImageManager
from src.batch_config import BatchSpecification
from pathlib import Path

@pytest.fixture
def background_manager(tmp_path):
    """Creates a dummy background image and a BackgroundImageManager."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_image = Image.new("RGB", (300, 200), "red")
    bg_path = bg_dir / "background.png"
    bg_image.save(bg_path)
    return BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

def test_ocr_data_generator_initialization():
    """Tests that the OCRDataGenerator class can be initialized.""" 
    try:
        generator = OCRDataGenerator()
    except Exception as e:
        pytest.fail(f"OCRDataGenerator initialization failed: {e}")
    
    assert generator is not None

def test_generate_from_plan_ltr_places_on_canvas():
    """Tests that generate_from_plan creates a valid image and bboxes on a larger canvas."""
    generator = OCRDataGenerator()
    spec = BatchSpecification(name="test", proportion=1.0, text_direction="left_to_right", corpus_file="test.txt")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    plan = generator.plan_generation(spec, text, font_path)
    
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
    assert bbox_h["x0"] > 10
    assert bbox_h["y1"] > bbox_h["y0"]

def test_background_image_is_applied(background_manager):
    """Tests that a background image from the manager is used in the final image."""
    generator = OCRDataGenerator()
    spec = BatchSpecification(name="test", proportion=1.0, text_direction="left_to_right", corpus_file="test.txt")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    plan = generator.plan_generation(spec, text, font_path, background_manager=background_manager)
    image, _ = generator.generate_from_plan(plan)

    # Check a corner pixel for the background color
    corner_pixel_color = image.getpixel((0, 0))
    assert corner_pixel_color == (255, 0, 0, 255) # Red

def test_text_surface_handles_descenders():
    """Tests that the rendered text surface is tall enough for glyphs with descenders."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # The letter 'g' has a descender
    text_with_descender = "glyph"
    # The letter 'a' does not
    text_without_descender = "alpha"

    # Render both texts directly to surfaces
    surface_desc, _ = generator._render_text(text_with_descender, font_path, "left_to_right")
    surface_no_desc, _ = generator._render_text(text_without_descender, font_path, "left_to_right")

    # Find the actual height of the rendered pixels by checking the alpha channel
    alpha_desc = np.array(surface_desc.split()[3])
    rendered_height_desc = np.max(np.where(alpha_desc > 0)[0]) - np.min(np.where(alpha_desc > 0)[0])

    alpha_no_desc = np.array(surface_no_desc.split()[3])
    rendered_height_no_desc = np.max(np.where(alpha_no_desc > 0)[0]) - np.min(np.where(alpha_no_desc > 0)[0])

    # The text with a descender should produce a taller rendered surface
    assert rendered_height_desc > rendered_height_no_desc
