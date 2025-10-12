import pytest
from PIL import Image
import numpy as np
from src.generator import OCRDataGenerator
from src.background_manager import BackgroundImageManager


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

    # Render with overlap
    image_overlap, _ = generator._render_text(
        text, font_path, direction, glyph_overlap_intensity=0.5
    )

    # Render without overlap
    image_no_overlap, _ = generator._render_text(
        text, font_path, direction, glyph_overlap_intensity=0.0
    )

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

def test_drop_shadow_integration():
    """Tests that the drop_shadow effect is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"
    shadow_options = {"offset": (5, 5), "radius": 2, "color": (0, 0, 0, 128)}

    # Plan with shadow
    plan_shadow = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, drop_shadow_options=shadow_options
    )
    image_shadow, _ = generator.generate_from_plan(plan_shadow)

    # Plan without shadow (control)
    plan_no_shadow = generator.plan_generation(
        text=text, font_path=font_path, direction=direction, drop_shadow_options=None
    )
    plan_no_shadow["seed"] = plan_shadow["seed"]
    image_no_shadow, _ = generator.generate_from_plan(plan_no_shadow)

    # The images should not be identical
    assert not np.array_equal(np.array(image_shadow), np.array(image_no_shadow))

def test_per_glyph_color_is_applied():
    """Tests that per-glyph coloring is applied correctly."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "ab"
    direction = "left_to_right"
    colors = [(255, 0, 0, 255), (0, 255, 0, 255)] # Red, Green

    # We test the internal rendering method directly to check the text surface
    text_surface, bboxes = generator._render_text(
        text, 
        font_path, 
        direction, 
        color_mode='per_glyph',
        color_palette=colors
    )

    # Check color of first character 'a'
    bbox_a = bboxes[0]
    found_color_a = False
    # Scan the entire bounding box for the correct color
    for x in range(bbox_a['x0'], bbox_a['x1']):
        for y in range(bbox_a['y0'], bbox_a['y1']):
            if text_surface.getpixel((x, y)) == colors[0]:
                found_color_a = True
                break
        if found_color_a:
            break
    assert found_color_a, "Did not find correct color for first character"

    # Check color of second character 'b'
    bbox_b = bboxes[1]
    found_color_b = False
    for x in range(bbox_b['x0'], bbox_b['x1']):
        for y in range(bbox_b['y0'], bbox_b['y1']):
            if text_surface.getpixel((x, y)) == colors[1]:
                found_color_b = True
                break
        if found_color_b:
            break
    assert found_color_b, "Did not find correct color for second character"


def test_horizontal_gradient_color_is_applied():
    """Tests that a horizontal gradient is applied correctly."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    direction = "left_to_right"
    colors = [(255, 0, 0, 255), (0, 0, 255, 255)] # Red to Blue

    text_surface, bboxes = generator._render_text(
        text, 
        font_path, 
        direction, 
        color_mode='gradient',
        color_palette=colors
    )

    # Check color of first character 'h'
    bbox_first = bboxes[0]
    found_color_first = False
    for x in range(bbox_first['x0'], bbox_first['x1']):
        for y in range(bbox_first['y0'], bbox_first['y1']):
            if text_surface.getpixel((x, y)) == colors[0]:
                found_color_first = True
                break
        if found_color_first:
            break
    assert found_color_first, "Did not find red pixel in first character"

    # Check color of last character 'o'
    bbox_last = bboxes[-1]
    found_color_last = False
    for x in range(bbox_last['x0'], bbox_last['x1']):
        for y in range(bbox_last['y0'], bbox_last['y1']):
            if text_surface.getpixel((x, y)) == colors[1]:
                found_color_last = True
                break
        if found_color_last:
            break
    assert found_color_last, "Did not find blue pixel in last character"

def test_vertical_gradient_color_is_applied():
    """Tests that a vertical gradient is applied correctly."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "VERTICAL"
    direction = "top_to_bottom"
    colors = [(255, 0, 0, 255), (0, 0, 255, 255)] # Red to Blue

    text_surface, bboxes = generator._render_text(
        text, 
        font_path, 
        direction, 
        color_mode='gradient',
        color_palette=colors
    )

    # Check color of first character 'V'
    bbox_first = bboxes[0]
    found_color_first = False
    for x in range(bbox_first['x0'], bbox_first['x1']):
        for y in range(bbox_first['y0'], bbox_first['y1']):
            if text_surface.getpixel((x, y)) == colors[0]:
                found_color_first = True
                break
        if found_color_first:
            break
    assert found_color_first, "Did not find red pixel in first character"

    # Check color of last character 'L'
    bbox_last = bboxes[-1]
    found_color_last = False
    for x in range(bbox_last['x0'], bbox_last['x1']):
        for y in range(bbox_last['y0'], bbox_last['y1']):
            if text_surface.getpixel((x, y)) == colors[1]:
                found_color_last = True
                break
        if found_color_last:
            break
    assert found_color_last, "Did not find blue pixel in last character"


def test_rotation_integration():
    """Tests that the rotation augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Plan with rotation
    plan_rotated = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', rotation_angle=30)
    image_rotated, _ = generator.generate_from_plan(plan_rotated)

    # Plan without rotation
    plan_no_rotation = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', rotation_angle=0)
    plan_no_rotation["seed"] = plan_rotated["seed"]
    image_no_rotation, _ = generator.generate_from_plan(plan_no_rotation)

    assert image_rotated.size != image_no_rotation.size

def test_perspective_warp_integration():
    """Tests that the perspective warp augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Plan with warp
    plan_warped = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', perspective_warp_magnitude=0.2)
    _, bboxes_warped = generator.generate_from_plan(plan_warped)

    # Plan without warp
    plan_no_warp = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', perspective_warp_magnitude=0.0)
    plan_no_warp["seed"] = plan_warped["seed"]
    _, bboxes_no_warp = generator.generate_from_plan(plan_no_warp)

    assert bboxes_warped[0]["x0"] != bboxes_no_warp[0]["x0"]

def test_elastic_distortion_integration():
    """Tests that the elastic distortion augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    elastic_options = {"alpha": 34, "sigma": 4}

    # Plan with distortion
    plan_distorted = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', elastic_distortion_options=elastic_options)
    image_distorted, _ = generator.generate_from_plan(plan_distorted)

    # Plan without distortion
    plan_no_distortion = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', elastic_distortion_options=None)
    plan_no_distortion["seed"] = plan_distorted["seed"]
    image_no_distortion, _ = generator.generate_from_plan(plan_no_distortion)

    assert not np.array_equal(np.array(image_distorted), np.array(image_no_distortion))

def test_noise_integration():
    """Tests that the noise augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Plan with noise
    plan_noisy = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', noise_amount=0.1)
    image_noisy, _ = generator.generate_from_plan(plan_noisy)

    # Plan without noise
    plan_no_noise = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', noise_amount=0.0)
    plan_no_noise["seed"] = plan_noisy["seed"]
    image_no_noise, _ = generator.generate_from_plan(plan_no_noise)

    assert not np.array_equal(np.array(image_noisy), np.array(image_no_noise))

def test_blur_integration():
    """Tests that the blur augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Plan with blur
    plan_blurred = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', blur_radius=2.0)
    image_blurred, _ = generator.generate_from_plan(plan_blurred)

    # Plan without blur
    plan_no_blur = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', blur_radius=0.0)
    plan_no_blur["seed"] = plan_blurred["seed"]
    image_no_blur, _ = generator.generate_from_plan(plan_no_blur)

    assert not np.array_equal(np.array(image_blurred), np.array(image_no_blur))

from src.background_manager import BackgroundImageManager

@pytest.fixture
def background_manager(tmp_path):
    """Creates a dummy background image and a BackgroundImageManager."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_image = Image.new("RGB", (300, 200), "red")
    bg_path = bg_dir / "background.png"
    bg_image.save(bg_path)
    return BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

def test_brightness_contrast_integration():
    """Tests that brightness and contrast are correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Plan with adjustments
    plan_adjusted = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', brightness_factor=1.5, contrast_factor=1.5
    )
    image_adjusted, _ = generator.generate_from_plan(plan_adjusted)

    # Plan without adjustments
    plan_no_adjust = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', brightness_factor=1.0, contrast_factor=1.0
    )
    plan_no_adjust["seed"] = plan_adjusted["seed"]
    image_no_adjust, _ = generator.generate_from_plan(plan_no_adjust)

    assert not np.array_equal(np.array(image_adjusted), np.array(image_no_adjust))

def test_erosion_dilation_integration():
    """Tests that erosion/dilation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    erosion_options = {"mode": "erode", "kernel_size": 3}

    # Plan with erosion
    plan_eroded = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', erosion_dilation_options=erosion_options
    )
    image_eroded, _ = generator.generate_from_plan(plan_eroded)

    # Plan without erosion
    plan_no_erosion = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', erosion_dilation_options=None
    )
    plan_no_erosion["seed"] = plan_eroded["seed"]
    image_no_erosion, _ = generator.generate_from_plan(plan_no_erosion)

    assert not np.array_equal(np.array(image_eroded), np.array(image_no_erosion))

def test_grid_distortion_integration():
    """Tests that the grid distortion augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    grid_options = {"num_steps": 5, "distort_limit": 10}

    # Plan with distortion
    plan_distorted = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', grid_distortion_options=grid_options
    )
    image_distorted, _ = generator.generate_from_plan(plan_distorted)

    # Plan without distortion
    plan_no_distortion = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', grid_distortion_options=None
    )
    plan_no_distortion["seed"] = plan_distorted["seed"]
    image_no_distortion, _ = generator.generate_from_plan(plan_no_distortion)

    assert not np.array_equal(np.array(image_distorted), np.array(image_no_distortion))

def test_optical_distortion_integration():
    """Tests that the optical distortion augmentation is correctly integrated."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"
    optical_options = {"distort_limit": 0.1}

    # Plan with distortion
    plan_distorted = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', optical_distortion_options=optical_options
    )
    image_distorted, _ = generator.generate_from_plan(plan_distorted)

    # Plan without distortion
    plan_no_distortion = generator.plan_generation(
        text=text, font_path=font_path, direction='left_to_right', optical_distortion_options=None
    )
    plan_no_distortion["seed"] = plan_distorted["seed"]
    image_no_distortion, _ = generator.generate_from_plan(plan_no_distortion)

    assert not np.array_equal(np.array(image_distorted), np.array(image_no_distortion))

def test_background_image_is_applied(background_manager):
    """Tests that a background image from the manager is used in the final image."""
    generator = OCRDataGenerator()
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    plan = generator.plan_generation(text=text, font_path=font_path, direction='left_to_right', background_manager=background_manager)
    image, _ = generator.generate_from_plan(plan)

    # Check a corner pixel for the background color
    corner_pixel_color = image.getpixel((0, 0))
    assert corner_pixel_color == (255, 0, 0, 255) # Red
