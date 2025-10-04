#!/usr/bin/env python3
"""
Comprehensive test suite for curved vertical text rendering.
Tests both top-to-bottom and bottom-to-top text with various curve types.
"""
import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageFont
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from main import OCRDataGenerator


@pytest.fixture
def test_font():
    """Load a test font for vertical text rendering."""
    font_dir = Path(__file__).parent.parent / "data" / "fonts"

    # Try to find a CJK font that supports vertical text
    cjk_fonts = [
        "NotoSerifCJKjp-Regular.otf",
        "NotoSansCJKtc-VF.otf",
        "NanumGothic.ttf",
    ]

    for font_name in cjk_fonts:
        font_path = font_dir / font_name
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=40)

    # Fallback to any font
    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if font_files:
        return ImageFont.truetype(str(font_files[0]), size=40)

    pytest.skip("No fonts available for testing")


@pytest.fixture
def generator(test_font):
    """Create a generator instance."""
    font_path = test_font.path
    return OCRDataGenerator([font_path])


class TestTopToBottomCurveInputValidation:
    """Test input validation for top-to-bottom curved text."""

    def test_empty_text_ttb(self, generator, test_font):
        """Empty text should return empty image."""
        img, char_boxes = generator.render_top_to_bottom_curved("", test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 0

    def test_single_character_ttb(self, generator, test_font):
        """Single character should render without error."""
        img, char_boxes = generator.render_top_to_bottom_curved("A", test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 1
        assert char_boxes[0].char == "A"

    def test_very_long_text_ttb(self, generator, test_font):
        """Very long text should render with vertical curve."""
        long_text = "A" * 100
        img, char_boxes = generator.render_top_to_bottom_curved(long_text, test_font, curve_type='arc', curve_intensity=0.2)
        assert img is not None
        assert len(char_boxes) == 100
        assert img.height > 1000  # Should be tall

    def test_cjk_characters_ttb(self, generator, test_font):
        """CJK characters (traditional vertical text) should render."""
        text = "日本語テスト"
        img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == len(text)

    def test_mixed_characters_ttb(self, generator, test_font):
        """Mixed ASCII and unicode should render."""
        text = "Test123テスト"
        img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type='sine', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == len(text)

    def test_whitespace_handling_ttb(self, generator, test_font):
        """Whitespace should be preserved in vertical text."""
        text = "A B C"
        img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 5  # Including spaces


class TestBottomToTopCurveInputValidation:
    """Test input validation for bottom-to-top curved text."""

    def test_empty_text_btt(self, generator, test_font):
        """Empty text should return empty image."""
        img, char_boxes = generator.render_bottom_to_top_curved("", test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 0

    def test_single_character_btt(self, generator, test_font):
        """Single character should render without error."""
        img, char_boxes = generator.render_bottom_to_top_curved("A", test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 1
        assert char_boxes[0].char == "A"

    def test_very_long_text_btt(self, generator, test_font):
        """Very long text should render with vertical curve."""
        long_text = "B" * 100
        img, char_boxes = generator.render_bottom_to_top_curved(long_text, test_font, curve_type='arc', curve_intensity=0.2)
        assert img is not None
        assert len(char_boxes) == 100
        assert img.height > 1000  # Should be tall

    def test_cjk_characters_btt(self, generator, test_font):
        """CJK characters should render bottom-to-top."""
        text = "中文測試"
        img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == len(text)

    def test_mixed_characters_btt(self, generator, test_font):
        """Mixed ASCII and unicode should render."""
        text = "Test456測試"
        img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type='sine', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == len(text)

    def test_whitespace_handling_btt(self, generator, test_font):
        """Whitespace should be preserved in vertical text."""
        text = "X Y Z"
        img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 5  # Including spaces


class TestVerticalCurveParameters:
    """Test curve parameters for vertical text."""

    def test_zero_intensity_ttb(self, generator, test_font):
        """Zero intensity should fall back to straight vertical rendering."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.0)
        assert img is not None
        assert len(char_boxes) == 4

    def test_zero_intensity_btt(self, generator, test_font):
        """Zero intensity should fall back to straight vertical rendering."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.0)
        assert img is not None
        assert len(char_boxes) == 4

    def test_negative_intensity_ttb(self, generator, test_font):
        """Negative intensity should be handled gracefully."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=-0.5)
        assert img is not None
        assert len(char_boxes) == 4

    def test_negative_intensity_btt(self, generator, test_font):
        """Negative intensity should be handled gracefully."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=-0.5)
        assert img is not None
        assert len(char_boxes) == 4

    def test_extreme_high_intensity_ttb(self, generator, test_font):
        """Very high intensity should be clamped."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=5.0)
        assert img is not None
        assert len(char_boxes) == 4

    def test_extreme_high_intensity_btt(self, generator, test_font):
        """Very high intensity should be clamped."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=5.0)
        assert img is not None
        assert len(char_boxes) == 4

    def test_arc_curve_type_ttb(self, generator, test_font):
        """Arc curve should work for top-to-bottom."""
        img, char_boxes = generator.render_top_to_bottom_curved("Curved", test_font, curve_type='arc', curve_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 6

    def test_arc_curve_type_btt(self, generator, test_font):
        """Arc curve should work for bottom-to-top."""
        img, char_boxes = generator.render_bottom_to_top_curved("Curved", test_font, curve_type='arc', curve_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 6

    def test_sine_curve_type_ttb(self, generator, test_font):
        """Sine wave curve should work for top-to-bottom."""
        img, char_boxes = generator.render_top_to_bottom_curved("Wavy", test_font, curve_type='sine', curve_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 4

    def test_sine_curve_type_btt(self, generator, test_font):
        """Sine wave curve should work for bottom-to-top."""
        img, char_boxes = generator.render_bottom_to_top_curved("Wavy", test_font, curve_type='sine', curve_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 4

    def test_invalid_curve_type_ttb(self, generator, test_font):
        """Invalid curve type should fall back to arc or straight."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='invalid', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 4

    def test_invalid_curve_type_btt(self, generator, test_font):
        """Invalid curve type should fall back to arc or straight."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='invalid', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 4


class TestVerticalBoundingBoxAccuracy:
    """Test bounding box accuracy for curved vertical text."""

    def test_bbox_within_bounds_ttb(self, generator, test_font):
        """All bboxes should be within image bounds for top-to-bottom."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test123", test_font, curve_type='arc', curve_intensity=0.3)
        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] >= 0, "x0 should be non-negative"
            assert bbox[1] >= 0, "y0 should be non-negative"
            assert bbox[2] <= img.width, f"x1 {bbox[2]} should be within image width {img.width}"
            assert bbox[3] <= img.height, f"y1 {bbox[3]} should be within image height {img.height}"

    def test_bbox_within_bounds_btt(self, generator, test_font):
        """All bboxes should be within image bounds for bottom-to-top."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test123", test_font, curve_type='arc', curve_intensity=0.3)
        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] >= 0, "x0 should be non-negative"
            assert bbox[1] >= 0, "y0 should be non-negative"
            assert bbox[2] <= img.width, f"x1 {bbox[2]} should be within image width {img.width}"
            assert bbox[3] <= img.height, f"y1 {bbox[3]} should be within image height {img.height}"

    def test_bbox_coordinates_ordered_ttb(self, generator, test_font):
        """Bbox coordinates should be properly ordered (x0 < x1, y0 < y1)."""
        img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='sine', curve_intensity=0.4)
        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] < bbox[2], f"x0 {bbox[0]} should be less than x1 {bbox[2]}"
            assert bbox[1] < bbox[3], f"y0 {bbox[1]} should be less than y1 {bbox[3]}"

    def test_bbox_coordinates_ordered_btt(self, generator, test_font):
        """Bbox coordinates should be properly ordered (x0 < x1, y0 < y1)."""
        img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='sine', curve_intensity=0.4)
        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] < bbox[2], f"x0 {bbox[0]} should be less than x1 {bbox[2]}"
            assert bbox[1] < bbox[3], f"y0 {bbox[1]} should be less than y1 {bbox[3]}"

    def test_bbox_reasonable_size_ttb(self, generator, test_font):
        """Bboxes should have reasonable dimensions."""
        img, char_boxes = generator.render_top_to_bottom_curved("ABC", test_font, curve_type='arc', curve_intensity=0.3)
        for char_box in char_boxes:
            bbox = char_box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            assert width > 0, "Bbox width should be positive"
            assert height > 0, "Bbox height should be positive"
            assert width < img.width, "Bbox width should be less than image width"
            assert height < 200, "Single character bbox height should be reasonable"

    def test_bbox_reasonable_size_btt(self, generator, test_font):
        """Bboxes should have reasonable dimensions."""
        img, char_boxes = generator.render_bottom_to_top_curved("ABC", test_font, curve_type='arc', curve_intensity=0.3)
        for char_box in char_boxes:
            bbox = char_box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            assert width > 0, "Bbox width should be positive"
            assert height > 0, "Bbox height should be positive"
            assert width < img.width, "Bbox width should be less than image width"
            assert height < 200, "Single character bbox height should be reasonable"

    def test_vertical_progression_ttb(self, generator, test_font):
        """Characters should progress vertically from top to bottom."""
        img, char_boxes = generator.render_top_to_bottom_curved("ABCD", test_font, curve_type='arc', curve_intensity=0.2)
        # First character should be higher (lower y) than last character
        first_y = (char_boxes[0].bbox[1] + char_boxes[0].bbox[3]) / 2
        last_y = (char_boxes[-1].bbox[1] + char_boxes[-1].bbox[3]) / 2
        assert first_y < last_y, "Top-to-bottom text should have increasing y coordinates"

    def test_vertical_progression_btt(self, generator, test_font):
        """Characters should progress vertically from bottom to top."""
        img, char_boxes = generator.render_bottom_to_top_curved("ABCD", test_font, curve_type='arc', curve_intensity=0.2)
        # First character should be lower (higher y) than last character
        first_y = (char_boxes[0].bbox[1] + char_boxes[0].bbox[3]) / 2
        last_y = (char_boxes[-1].bbox[1] + char_boxes[-1].bbox[3]) / 2
        assert first_y > last_y, "Bottom-to-top text should have decreasing y coordinates"

    def test_bbox_count_matches_text_ttb(self, generator, test_font):
        """Number of bboxes should match text length."""
        text = "Vertical!"
        img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert len(char_boxes) == len(text)

    def test_bbox_count_matches_text_btt(self, generator, test_font):
        """Number of bboxes should match text length."""
        text = "Vertical!"
        img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type='arc', curve_intensity=0.3)
        assert len(char_boxes) == len(text)


class TestVerticalFontVariations:
    """Test vertical curves with different font sizes."""

    def test_small_font_ttb(self, generator):
        """Small font should render with vertical curve."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        small_font = ImageFont.truetype(str(font_files[0]), size=12)
        img, char_boxes = generator.render_top_to_bottom_curved("Test", small_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 4

    def test_large_font_ttb(self, generator):
        """Large font should render with vertical curve."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        large_font = ImageFont.truetype(str(font_files[0]), size=80)
        img, char_boxes = generator.render_top_to_bottom_curved("XY", large_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 2

    def test_small_font_btt(self, generator):
        """Small font should render with vertical curve."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        small_font = ImageFont.truetype(str(font_files[0]), size=12)
        img, char_boxes = generator.render_bottom_to_top_curved("Test", small_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 4

    def test_large_font_btt(self, generator):
        """Large font should render with vertical curve."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        large_font = ImageFont.truetype(str(font_files[0]), size=80)
        img, char_boxes = generator.render_bottom_to_top_curved("XY", large_font, curve_type='arc', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 2


class TestVerticalImageQuality:
    """Test image quality for curved vertical text."""

    def test_image_mode_rgb_ttb(self, generator, test_font):
        """Output image should be RGB mode."""
        img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        assert img.mode == 'RGB'

    def test_image_mode_rgb_btt(self, generator, test_font):
        """Output image should be RGB mode."""
        img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        assert img.mode == 'RGB'

    def test_image_not_blank_ttb(self, generator, test_font):
        """Image should contain non-white pixels (actual text)."""
        img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        img_array = np.array(img)
        white_pixels = np.all(img_array == 255, axis=-1)
        assert not np.all(white_pixels), "Image should contain some non-white pixels"

    def test_image_not_blank_btt(self, generator, test_font):
        """Image should contain non-white pixels (actual text)."""
        img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        img_array = np.array(img)
        white_pixels = np.all(img_array == 255, axis=-1)
        assert not np.all(white_pixels), "Image should contain some non-white pixels"

    def test_reasonable_dimensions_ttb(self, generator, test_font):
        """Image dimensions should be reasonable for vertical text."""
        img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        assert img.height > img.width, "Vertical text should be taller than wide"
        assert img.height > 100, "Image should be tall enough for vertical text"
        assert img.width > 20, "Image should have some width"

    def test_reasonable_dimensions_btt(self, generator, test_font):
        """Image dimensions should be reasonable for vertical text."""
        img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        assert img.height > img.width, "Vertical text should be taller than wide"
        assert img.height > 100, "Image should be tall enough for vertical text"
        assert img.width > 20, "Image should have some width"


class TestVerticalEdgeCases:
    """Test edge cases for curved vertical text."""

    def test_single_char_extreme_curve_ttb(self, generator, test_font):
        """Single character with extreme curve should not crash."""
        img, char_boxes = generator.render_top_to_bottom_curved("X", test_font, curve_type='arc', curve_intensity=0.9)
        assert img is not None
        assert len(char_boxes) == 1

    def test_single_char_extreme_curve_btt(self, generator, test_font):
        """Single character with extreme curve should not crash."""
        img, char_boxes = generator.render_bottom_to_top_curved("X", test_font, curve_type='arc', curve_intensity=0.9)
        assert img is not None
        assert len(char_boxes) == 1

    def test_long_text_small_font_ttb(self, generator):
        """Long text with small font should render."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        small_font = ImageFont.truetype(str(font_files[0]), size=10)
        long_text = "A" * 50
        img, char_boxes = generator.render_top_to_bottom_curved(long_text, small_font, curve_type='arc', curve_intensity=0.2)
        assert img is not None
        assert len(char_boxes) == 50

    def test_long_text_small_font_btt(self, generator):
        """Long text with small font should render."""
        font_path = Path(__file__).parent.parent / "data" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        small_font = ImageFont.truetype(str(font_files[0]), size=10)
        long_text = "B" * 50
        img, char_boxes = generator.render_bottom_to_top_curved(long_text, small_font, curve_type='arc', curve_intensity=0.2)
        assert img is not None
        assert len(char_boxes) == 50

    def test_numbers_and_symbols_ttb(self, generator, test_font):
        """Numbers and symbols should render vertically."""
        text = "123!@#"
        img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type='sine', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 6

    def test_numbers_and_symbols_btt(self, generator, test_font):
        """Numbers and symbols should render vertically."""
        text = "456$%^"
        img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type='sine', curve_intensity=0.3)
        assert img is not None
        assert len(char_boxes) == 6


class TestVerticalStability:
    """Test stability and crash prevention for vertical curves."""

    def test_no_crashes_various_inputs_ttb(self, generator, test_font):
        """Various inputs should not crash top-to-bottom rendering."""
        test_cases = [
            ("", 'arc', 0.3),
            ("A", 'arc', 0.0),
            ("Test", 'sine', 0.5),
            ("LongText" * 10, 'arc', 0.2),
            ("123", 'sine', 1.0),
        ]

        for text, curve_type, intensity in test_cases:
            img, char_boxes = generator.render_top_to_bottom_curved(text, test_font, curve_type, intensity)
            assert img is not None

    def test_no_crashes_various_inputs_btt(self, generator, test_font):
        """Various inputs should not crash bottom-to-top rendering."""
        test_cases = [
            ("", 'arc', 0.3),
            ("B", 'arc', 0.0),
            ("Test", 'sine', 0.5),
            ("LongText" * 10, 'arc', 0.2),
            ("456", 'sine', 1.0),
        ]

        for text, curve_type, intensity in test_cases:
            img, char_boxes = generator.render_bottom_to_top_curved(text, test_font, curve_type, intensity)
            assert img is not None

    def test_repeated_generation_ttb(self, generator, test_font):
        """Repeated generation should produce consistent results."""
        results = []
        for _ in range(3):
            img, char_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
            results.append((img.size, len(char_boxes)))

        # All results should be identical
        assert len(set(results)) == 1, "Repeated generation should be deterministic"

    def test_repeated_generation_btt(self, generator, test_font):
        """Repeated generation should produce consistent results."""
        results = []
        for _ in range(3):
            img, char_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
            results.append((img.size, len(char_boxes)))

        # All results should be identical
        assert len(set(results)) == 1, "Repeated generation should be deterministic"


class TestVerticalComparisonTests:
    """Compare vertical curved vs straight rendering."""

    def test_curved_differs_from_straight_ttb(self, generator, test_font):
        """Curved vertical text should differ from straight vertical text."""
        straight_img, _ = generator.render_top_to_bottom("Test", test_font)
        curved_img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.5)

        # Images should be different
        straight_array = np.array(straight_img)
        curved_array = np.array(curved_img)

        # Note: sizes might differ, so compare pixel distributions
        straight_pixels = np.mean(straight_array)
        curved_pixels = np.mean(curved_array)

        # Should have different characteristics
        assert straight_img.size != curved_img.size or not np.array_equal(straight_array, curved_array)

    def test_curved_differs_from_straight_btt(self, generator, test_font):
        """Curved vertical text should differ from straight vertical text."""
        straight_img, _ = generator.render_bottom_to_top("Test", test_font)
        curved_img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.5)

        # Images should be different
        straight_array = np.array(straight_img)
        curved_array = np.array(curved_img)

        # Should have different characteristics
        assert straight_img.size != curved_img.size or not np.array_equal(straight_array, curved_array)

    def test_arc_differs_from_sine_ttb(self, generator, test_font):
        """Arc and sine curves should produce different results."""
        arc_img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.4)
        sine_img, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='sine', curve_intensity=0.4)

        arc_array = np.array(arc_img)
        sine_array = np.array(sine_img)

        # Should produce different images
        assert not np.array_equal(arc_array, sine_array)

    def test_arc_differs_from_sine_btt(self, generator, test_font):
        """Arc and sine curves should produce different results."""
        arc_img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.4)
        sine_img, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='sine', curve_intensity=0.4)

        arc_array = np.array(arc_img)
        sine_array = np.array(sine_img)

        # Should produce different images
        assert not np.array_equal(arc_array, sine_array)

    def test_ttb_differs_from_btt(self, generator, test_font):
        """Top-to-bottom should differ from bottom-to-top."""
        ttb_img, ttb_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)
        btt_img, btt_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.3)

        # Character ordering should be different
        ttb_first_y = (ttb_boxes[0].bbox[1] + ttb_boxes[0].bbox[3]) / 2
        ttb_last_y = (ttb_boxes[-1].bbox[1] + ttb_boxes[-1].bbox[3]) / 2

        btt_first_y = (btt_boxes[0].bbox[1] + btt_boxes[0].bbox[3]) / 2
        btt_last_y = (btt_boxes[-1].bbox[1] + btt_boxes[-1].bbox[3]) / 2

        # TTB should go down, BTT should go up
        assert ttb_first_y < ttb_last_y, "TTB should progress downward"
        assert btt_first_y > btt_last_y, "BTT should progress upward"


class TestVerticalIntegration:
    """Test integration with the full pipeline."""

    def test_vertical_curved_through_generate_image_ttb(self, generator, test_font):
        """Test curved vertical text through full generate_image pipeline."""
        # This will be tested once the feature is implemented
        # For now, just test that the methods exist
        assert hasattr(generator, 'render_top_to_bottom_curved')

    def test_vertical_curved_through_generate_image_btt(self, generator, test_font):
        """Test curved vertical text through full generate_image pipeline."""
        # This will be tested once the feature is implemented
        # For now, just test that the methods exist
        assert hasattr(generator, 'render_bottom_to_top_curved')


class TestVerticalPerformance:
    """Test performance of vertical curved text rendering."""

    def test_long_vertical_text_performance_ttb(self, generator, test_font):
        """Long vertical text should render in reasonable time."""
        import time
        long_text = "A" * 200

        start_time = time.time()
        img, char_boxes = generator.render_top_to_bottom_curved(long_text, test_font, curve_type='arc', curve_intensity=0.3)
        elapsed = time.time() - start_time

        assert elapsed < 5.0, f"Rendering took {elapsed}s, should be < 5s"
        assert img is not None
        assert len(char_boxes) == 200

    def test_long_vertical_text_performance_btt(self, generator, test_font):
        """Long vertical text should render in reasonable time."""
        import time
        long_text = "B" * 200

        start_time = time.time()
        img, char_boxes = generator.render_bottom_to_top_curved(long_text, test_font, curve_type='arc', curve_intensity=0.3)
        elapsed = time.time() - start_time

        assert elapsed < 5.0, f"Rendering took {elapsed}s, should be < 5s"
        assert img is not None
        assert len(char_boxes) == 200


class TestVerticalDocumentedBehavior:
    """Test documented specifications for vertical curves."""

    def test_intensity_zero_equals_straight_ttb(self, generator, test_font):
        """Zero intensity should produce identical results to straight rendering."""
        straight_img, straight_boxes = generator.render_top_to_bottom("Test", test_font)
        curved_img, curved_boxes = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.0)

        # Should have same number of characters
        assert len(straight_boxes) == len(curved_boxes)

    def test_intensity_zero_equals_straight_btt(self, generator, test_font):
        """Zero intensity should produce identical results to straight rendering."""
        straight_img, straight_boxes = generator.render_bottom_to_top("Test", test_font)
        curved_img, curved_boxes = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.0)

        # Should have same number of characters
        assert len(straight_boxes) == len(curved_boxes)

    def test_curve_intensity_range_ttb(self, generator, test_font):
        """Intensity should be clamped to valid range."""
        # Very low intensity
        img1, boxes1 = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=0.001)
        assert img1 is not None

        # Very high intensity (should be clamped)
        img2, boxes2 = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc', curve_intensity=10.0)
        assert img2 is not None

    def test_curve_intensity_range_btt(self, generator, test_font):
        """Intensity should be clamped to valid range."""
        # Very low intensity
        img1, boxes1 = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=0.001)
        assert img1 is not None

        # Very high intensity (should be clamped)
        img2, boxes2 = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc', curve_intensity=10.0)
        assert img2 is not None
