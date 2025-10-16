"""
Tests for curved text rendering (arcs and sine waves).
"""

import pytest
from PIL import Image, ImageDraw
import numpy as np
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification


def test_arc_text_ltr_produces_curved_output():
    """Tests that arc rendering with LTR text produces output different from straight text."""
    generator = OCRDataGenerator()

    # Create a spec with arc curve
    arc_spec = BatchSpecification(
        name="arc_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=200.0,
        arc_radius_max=200.0,  # Fixed radius for deterministic test
        arc_concave=True
    )

    # Create a spec with no curve for comparison
    straight_spec = BatchSpecification(
        name="straight_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    # Generate plans
    arc_plan = generator.plan_generation(arc_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)

    # Use same seed and canvas settings to isolate the curve effect
    straight_plan["seed"] = arc_plan["seed"]
    straight_plan["canvas_w"] = arc_plan["canvas_w"]
    straight_plan["canvas_h"] = arc_plan["canvas_h"]
    straight_plan["placement_x"] = arc_plan["placement_x"]
    straight_plan["placement_y"] = arc_plan["placement_y"]

    # Generate images
    arc_image, arc_bboxes = generator.generate_from_plan(arc_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # Convert to numpy for comparison
    arc_array = np.array(arc_image)
    straight_array = np.array(straight_image)

    # The images should be different (curved vs straight)
    assert not np.array_equal(arc_array, straight_array), \
        "Arc text should produce different output from straight text"

    # Both should have bboxes for each character
    assert len(arc_bboxes) == len(text)
    assert len(straight_bboxes) == len(text)

    # Arc text bboxes should have different positions than straight text
    # (at minimum, the last character should be in a different vertical position)
    arc_last_char_y = arc_bboxes[-1]["y0"]
    straight_last_char_y = straight_bboxes[-1]["y0"]

    # For concave arc, last character should be higher (smaller y) than straight
    assert arc_last_char_y != straight_last_char_y, \
        "Arc text should position characters differently from straight text"


def test_arc_text_bbox_accuracy():
    """Tests that bounding boxes for arc text are reasonably tight."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="arc_bbox_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=150.0,
        arc_radius_max=150.0,
        arc_concave=True
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "test"

    plan = generator.plan_generation(spec, text, font_path)
    image, bboxes = generator.generate_from_plan(plan)

    # Basic sanity checks for bboxes
    assert len(bboxes) == len(text), "Should have one bbox per character"

    for i, bbox in enumerate(bboxes):
        # Ensure bbox has the correct character
        assert bbox['char'] == text[i], f"Bbox {i} should contain character '{text[i]}'"

        # Extract the bbox coordinates
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']

        # Ensure bbox is within image bounds
        assert 0 <= x0 < image.width, f"Bbox {i} x0={x0} is out of bounds (width={image.width})"
        assert 0 <= y0 < image.height, f"Bbox {i} y0={y0} is out of bounds (height={image.height})"
        assert 0 < x1 <= image.width, f"Bbox {i} x1={x1} is out of bounds (width={image.width})"
        assert 0 < y1 <= image.height, f"Bbox {i} y1={y1} is out of bounds (height={image.height})"

        # Ensure bbox has positive area
        assert x1 > x0, f"Bbox {i} has invalid width: x0={x0}, x1={x1}"
        assert y1 > y0, f"Bbox {i} has invalid height: y0={y0}, y1={y1}"


def test_arc_zero_radius_equals_straight():
    """Tests that arc_radius=0.0 produces output equivalent to straight text."""
    generator = OCRDataGenerator()

    # Arc with radius=0 (should be straight)
    arc_spec = BatchSpecification(
        name="arc_zero",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=0.0,
        arc_radius_max=0.0
    )

    # Explicitly straight
    straight_spec = BatchSpecification(
        name="straight",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "test"

    # Use same seed and canvas for both to ensure identical placement
    arc_plan = generator.plan_generation(arc_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)
    straight_plan["seed"] = arc_plan["seed"]
    straight_plan["canvas_w"] = arc_plan["canvas_w"]
    straight_plan["canvas_h"] = arc_plan["canvas_h"]
    straight_plan["placement_x"] = arc_plan["placement_x"]
    straight_plan["placement_y"] = arc_plan["placement_y"]

    arc_image, arc_bboxes = generator.generate_from_plan(arc_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # Images should now be identical (or very similar)
    arc_array = np.array(arc_image)
    straight_array = np.array(straight_image)

    # Verify same dimensions
    assert arc_array.shape == straight_array.shape, \
        f"Images should have same dimensions (arc: {arc_array.shape}, straight: {straight_array.shape})"

    # Allow for minor rendering differences, but should be >98% similar
    diff_pixels = np.sum(arc_array != straight_array)
    total_pixels = arc_array.size
    similarity = 1.0 - (diff_pixels / total_pixels)

    assert similarity > 0.98, \
        f"Arc with radius=0 should produce nearly identical output to straight text (similarity: {similarity:.1%})"


def test_arc_text_rtl():
    """Tests that arc rendering works correctly with right-to-left text."""
    generator = OCRDataGenerator()

    arc_spec = BatchSpecification(
        name="arc_rtl_test",
        proportion=1.0,
        text_direction="right_to_left",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=200.0,
        arc_radius_max=200.0,
        arc_concave=True
    )

    straight_spec = BatchSpecification(
        name="straight_rtl_test",
        proportion=1.0,
        text_direction="right_to_left",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    arc_plan = generator.plan_generation(arc_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)
    straight_plan["seed"] = arc_plan["seed"]
    straight_plan["canvas_w"] = arc_plan["canvas_w"]
    straight_plan["canvas_h"] = arc_plan["canvas_h"]
    straight_plan["placement_x"] = arc_plan["placement_x"]
    straight_plan["placement_y"] = arc_plan["placement_y"]

    arc_image, arc_bboxes = generator.generate_from_plan(arc_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # The images should be different (curved vs straight)
    arc_array = np.array(arc_image)
    straight_array = np.array(straight_image)
    assert not np.array_equal(arc_array, straight_array), \
        "RTL arc text should produce different output from straight text"

    # Should have bboxes for each character
    assert len(arc_bboxes) == len(text)
    assert len(straight_bboxes) == len(text)


def test_arc_text_ttb():
    """Tests that arc rendering works correctly with top-to-bottom text."""
    generator = OCRDataGenerator()

    arc_spec = BatchSpecification(
        name="arc_ttb_test",
        proportion=1.0,
        text_direction="top_to_bottom",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=200.0,
        arc_radius_max=200.0,
        arc_concave=True
    )

    straight_spec = BatchSpecification(
        name="straight_ttb_test",
        proportion=1.0,
        text_direction="top_to_bottom",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    arc_plan = generator.plan_generation(arc_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)
    straight_plan["seed"] = arc_plan["seed"]
    straight_plan["canvas_w"] = arc_plan["canvas_w"]
    straight_plan["canvas_h"] = arc_plan["canvas_h"]
    straight_plan["placement_x"] = arc_plan["placement_x"]
    straight_plan["placement_y"] = arc_plan["placement_y"]

    arc_image, arc_bboxes = generator.generate_from_plan(arc_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # The images should be different (curved vs straight)
    arc_array = np.array(arc_image)
    straight_array = np.array(straight_image)
    assert not np.array_equal(arc_array, straight_array), \
        "TTB arc text should produce different output from straight text"

    # Should have bboxes for each character
    assert len(arc_bboxes) == len(text)
    assert len(straight_bboxes) == len(text)


def test_arc_text_btt():
    """Tests that arc rendering works correctly with bottom-to-top text."""
    generator = OCRDataGenerator()

    arc_spec = BatchSpecification(
        name="arc_btt_test",
        proportion=1.0,
        text_direction="bottom_to_top",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=200.0,
        arc_radius_max=200.0,
        arc_concave=True
    )

    straight_spec = BatchSpecification(
        name="straight_btt_test",
        proportion=1.0,
        text_direction="bottom_to_top",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    arc_plan = generator.plan_generation(arc_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)
    straight_plan["seed"] = arc_plan["seed"]
    straight_plan["canvas_w"] = arc_plan["canvas_w"]
    straight_plan["canvas_h"] = arc_plan["canvas_h"]
    straight_plan["placement_x"] = arc_plan["placement_x"]
    straight_plan["placement_y"] = arc_plan["placement_y"]

    arc_image, arc_bboxes = generator.generate_from_plan(arc_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # The images should be different (curved vs straight)
    arc_array = np.array(arc_image)
    straight_array = np.array(straight_image)
    assert not np.array_equal(arc_array, straight_array), \
        "BTT arc text should produce different output from straight text"

    # Should have bboxes for each character
    assert len(arc_bboxes) == len(text)
    assert len(straight_bboxes) == len(text)


def test_sine_text_ltr_produces_curved_output():
    """Tests that sine wave rendering with LTR text produces output different from straight text."""
    generator = OCRDataGenerator()

    # Create a spec with sine wave curve
    sine_spec = BatchSpecification(
        name="sine_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="sine",
        sine_amplitude_min=15.0,
        sine_amplitude_max=15.0,  # Fixed amplitude for deterministic test
        sine_frequency_min=0.05,
        sine_frequency_max=0.05,  # Fixed frequency
        sine_phase_min=0.0,
        sine_phase_max=0.0
    )

    # Create a spec with no curve for comparison
    straight_spec = BatchSpecification(
        name="straight_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello world"

    # Generate plans
    sine_plan = generator.plan_generation(sine_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)

    # Use same seed and canvas settings to isolate the curve effect
    straight_plan["seed"] = sine_plan["seed"]
    straight_plan["canvas_w"] = sine_plan["canvas_w"]
    straight_plan["canvas_h"] = sine_plan["canvas_h"]
    straight_plan["placement_x"] = sine_plan["placement_x"]
    straight_plan["placement_y"] = sine_plan["placement_y"]

    # Generate images
    sine_image, sine_bboxes = generator.generate_from_plan(sine_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # Convert to numpy for comparison
    sine_array = np.array(sine_image)
    straight_array = np.array(straight_image)

    # The images should be different (curved vs straight)
    assert not np.array_equal(sine_array, straight_array), \
        "Sine wave text should produce different output from straight text"

    # Both should have bboxes for each character
    assert len(sine_bboxes) == len(text)
    assert len(straight_bboxes) == len(text)

    # Sine wave text should have vertical variation in character positions
    # Check that not all characters have the same Y position
    y_positions = [bbox["y0"] for bbox in sine_bboxes]
    assert len(set(y_positions)) > 1, \
        "Sine wave text should have characters at different vertical positions"


def test_sine_zero_amplitude_equals_straight():
    """Tests that sine_amplitude=0.0 produces output equivalent to straight text."""
    generator = OCRDataGenerator()

    # Sine with amplitude=0 (should be straight)
    sine_spec = BatchSpecification(
        name="sine_zero",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="sine",
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.05,
        sine_frequency_max=0.05,
        sine_phase_min=0.0,
        sine_phase_max=0.0
    )

    # Explicitly straight
    straight_spec = BatchSpecification(
        name="straight",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="none"
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "test"

    # Use same seed and canvas for both to ensure identical placement
    sine_plan = generator.plan_generation(sine_spec, text, font_path)
    straight_plan = generator.plan_generation(straight_spec, text, font_path)
    straight_plan["seed"] = sine_plan["seed"]
    straight_plan["canvas_w"] = sine_plan["canvas_w"]
    straight_plan["canvas_h"] = sine_plan["canvas_h"]
    straight_plan["placement_x"] = sine_plan["placement_x"]
    straight_plan["placement_y"] = sine_plan["placement_y"]

    sine_image, sine_bboxes = generator.generate_from_plan(sine_plan)
    straight_image, straight_bboxes = generator.generate_from_plan(straight_plan)

    # Images should now be identical (or very similar)
    sine_array = np.array(sine_image)
    straight_array = np.array(straight_image)

    # Verify same dimensions
    assert sine_array.shape == straight_array.shape, \
        f"Images should have same dimensions (sine: {sine_array.shape}, straight: {straight_array.shape})"

    # Allow for minor rendering differences, but should be >98% similar
    diff_pixels = np.sum(sine_array != straight_array)
    total_pixels = sine_array.size
    similarity = 1.0 - (diff_pixels / total_pixels)

    assert similarity > 0.98, \
        f"Sine with amplitude=0 should produce nearly identical output to straight text (similarity: {similarity:.1%})"


def test_arc_text_with_rotation():
    """Tests that arc text can be combined with rotation augmentation."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="arc_rotation_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=200.0,
        arc_radius_max=200.0,
        arc_concave=True,
        rotation_angle_min=15.0,
        rotation_angle_max=15.0  # Fixed rotation for deterministic test
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "test"

    plan = generator.plan_generation(spec, text, font_path)
    image, bboxes = generator.generate_from_plan(plan)

    # Should successfully generate an image
    assert image is not None
    assert image.width > 0 and image.height > 0

    # Should have correct number of bboxes
    assert len(bboxes) == len(text)

    # All bboxes should be within image bounds
    for i, bbox in enumerate(bboxes):
        assert 0 <= bbox["x0"] < image.width, f"Bbox {i} x0 out of bounds"
        assert 0 <= bbox["y0"] < image.height, f"Bbox {i} y0 out of bounds"
        assert 0 < bbox["x1"] <= image.width, f"Bbox {i} x1 out of bounds"
        assert 0 < bbox["y1"] <= image.height, f"Bbox {i} y1 out of bounds"
        assert bbox["x1"] > bbox["x0"], f"Bbox {i} has invalid width"
        assert bbox["y1"] > bbox["y0"], f"Bbox {i} has invalid height"


def test_sine_text_with_perspective_warp():
    """Tests that sine wave text can be combined with perspective warp augmentation."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="sine_warp_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="sine",
        sine_amplitude_min=10.0,
        sine_amplitude_max=10.0,
        sine_frequency_min=0.05,
        sine_frequency_max=0.05,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
        perspective_warp_magnitude_min=0.2,
        perspective_warp_magnitude_max=0.2  # Fixed magnitude for deterministic test
    )

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    text = "hello"

    plan = generator.plan_generation(spec, text, font_path)
    image, bboxes = generator.generate_from_plan(plan)

    # Should successfully generate an image
    assert image is not None
    assert image.width > 0 and image.height > 0

    # Should have correct number of bboxes
    assert len(bboxes) == len(text)

    # All bboxes should be within image bounds
    for i, bbox in enumerate(bboxes):
        assert 0 <= bbox["x0"] < image.width, f"Bbox {i} x0 out of bounds"
        assert 0 <= bbox["y0"] < image.height, f"Bbox {i} y0 out of bounds"
        assert 0 < bbox["x1"] <= image.width, f"Bbox {i} x1 out of bounds"
        assert 0 < bbox["y1"] <= image.height, f"Bbox {i} y1 out of bounds"
        assert bbox["x1"] > bbox["x0"], f"Bbox {i} has invalid width"
        assert bbox["y1"] > bbox["y0"], f"Bbox {i} has invalid height"
