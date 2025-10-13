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

def test_plan_generation_uses_spec_ranges():
    """Tests that plan_generation correctly uses the ranges in the BatchSpecification."""
    generator = OCRDataGenerator()
    spec = BatchSpecification(
        name="test", 
        proportion=1.0, 
        text_direction="left_to_right", 
        corpus_file="test.txt",
        rotation_angle_min=-10,
        rotation_angle_max=10,
        glyph_overlap_intensity_min=0.1,
        glyph_overlap_intensity_max=0.5,
        elastic_distortion_alpha_min=20.0,
        elastic_distortion_alpha_max=30.0,
        elastic_distortion_sigma_min=6.0,
        elastic_distortion_sigma_max=8.0,
        grid_distortion_steps_min=3,
        grid_distortion_steps_max=6,
        grid_distortion_limit_min=5,
        grid_distortion_limit_max=15,
        optical_distortion_limit_min=0.1,
        optical_distortion_limit_max=0.5,
        noise_amount_min=0.05,
        noise_amount_max=0.15,
        blur_radius_min=1.0,
        blur_radius_max=3.0,
        brightness_factor_min=1.2,
        brightness_factor_max=1.8,
        contrast_factor_min=1.2,
        contrast_factor_max=1.8,
        erosion_dilation_kernel_min=2,
        erosion_dilation_kernel_max=4,
        cutout_width_min=10,
        cutout_width_max=30,
        cutout_height_min=5,
        cutout_height_max=15,
    )
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    plan = generator.plan_generation(spec, text, font_path)

    assert -10 <= plan["rotation_angle"] <= 10
    assert 0.1 <= plan["glyph_overlap_intensity"] <= 0.5
    assert 20.0 <= plan["elastic_distortion_options"]["alpha"] <= 30.0
    assert 6.0 <= plan["elastic_distortion_options"]["sigma"] <= 8.0
    assert 3 <= plan["grid_distortion_options"]["num_steps"] <= 6
    assert 5 <= plan["grid_distortion_options"]["distort_limit"] <= 15
    assert 0.1 <= plan["optical_distortion_options"]["distort_limit"] <= 0.5
    assert 0.05 <= plan["noise_amount"] <= 0.15
    assert 1.0 <= plan["blur_radius"] <= 3.0
    assert 1.2 <= plan["brightness_factor"] <= 1.8
    assert 1.2 <= plan["contrast_factor"] <= 1.8
    assert 2 <= plan["erosion_dilation_options"]["kernel_size"] <= 4
    assert 10 <= plan["cutout_options"]["cutout_size"][0] <= 30
    assert 5 <= plan["cutout_options"]["cutout_size"][1] <= 15

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
