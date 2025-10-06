"""
Tests for right-to-left curved text rendering.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from PIL import Image, ImageFont
from main import OCRDataGenerator

@pytest.fixture
def generator():
    """Create OCR generator for testing."""
    return OCRDataGenerator([], [])

@pytest.fixture
def test_font():
    """Load test font."""
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)


class TestRTLCurveBasics:
    """Basic RTL curved text rendering tests."""

    def test_arc_curve_rtl(self, generator, test_font):
        """Test RTL arc curve rendering."""
        text = "Hello World"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3
        )
        assert isinstance(image, Image.Image)
        assert len(char_boxes) == len(text)

    def test_sine_curve_rtl(self, generator, test_font):
        """Test RTL sine curve rendering."""
        text = "Test123"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='sine', curve_intensity=0.5
        )
        assert isinstance(image, Image.Image)
        assert len(char_boxes) == len(text)

    def test_zero_intensity_rtl(self, generator, test_font):
        """Test zero intensity falls back to straight RTL."""
        text = "Fallback"
        img_curved, boxes_curved = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.0
        )
        img_straight, boxes_straight = generator.render_right_to_left(text, test_font)

        # Should produce same dimensions as straight rendering
        assert img_curved.size == img_straight.size

    def test_empty_text_rtl(self, generator, test_font):
        """Test empty text handling."""
        image, char_boxes = generator.render_right_to_left_curved(
            "", test_font, curve_type='arc', curve_intensity=0.3
        )
        assert isinstance(image, Image.Image)
        assert len(char_boxes) == 0


class TestRTLCurveWithOverlap:
    """Test RTL curves with glyph overlap."""

    def test_rtl_arc_with_overlap(self, generator, test_font):
        """Test RTL arc with overlap reduces width."""
        text = "Overlap"
        img_no_overlap, _ = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3,
            overlap_intensity=0.0
        )
        img_overlap, _ = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3,
            overlap_intensity=0.5
        )

        # Overlap should reduce width
        assert img_overlap.width < img_no_overlap.width

    def test_rtl_sine_with_overlap(self, generator, test_font):
        """Test RTL sine with overlap."""
        text = "SineWave"
        img_no_overlap, _ = generator.render_right_to_left_curved(
            text, test_font, curve_type='sine', curve_intensity=0.4,
            overlap_intensity=0.0
        )
        img_overlap, _ = generator.render_right_to_left_curved(
            text, test_font, curve_type='sine', curve_intensity=0.4,
            overlap_intensity=0.6
        )

        assert img_overlap.width < img_no_overlap.width

    def test_rtl_curve_with_ink_bleed(self, generator, test_font):
        """Test RTL curve with ink bleed effect."""
        text = "InkBleed"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3,
            overlap_intensity=0.3, ink_bleed_intensity=0.5
        )

        assert isinstance(image, Image.Image)
        assert len(char_boxes) == len(text)


class TestRTLCurveBboxes:
    """Test bounding boxes for RTL curved text."""

    def test_bbox_within_bounds_rtl(self, generator, test_font):
        """Test all bboxes within image bounds."""
        text = "BoundTest"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.4
        )

        for box in char_boxes:
            bbox = box.bbox
            assert bbox[0] >= -2, f"x0 {bbox[0]} should be >= -2"
            assert bbox[1] >= -2, f"y0 {bbox[1]} should be >= -2"
            assert bbox[2] <= image.width + 2
            assert bbox[3] <= image.height + 2

    def test_bbox_coordinates_ordered_rtl(self, generator, test_font):
        """Test bbox coordinates are properly ordered."""
        text = "OrderTest"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='sine', curve_intensity=0.3
        )

        for box in char_boxes:
            bbox = box.bbox
            assert bbox[0] <= bbox[2], f"x0 should be <= x1"
            assert bbox[1] <= bbox[3], f"y0 should be <= y1"


class TestRTLCurveIntegration:
    """Integration tests for RTL curves."""

    def test_rtl_curve_through_generate_image(self, generator, test_font):
        """Test RTL curve through generate_image()."""
        text = "Integration"
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        image, metadata, returned_text, augmentations_applied = generator.generate_image(
            text, font_path, 32, direction='right_to_left',
            curve_type='arc', curve_intensity=0.4
        )

        assert isinstance(image, Image.Image)
        assert isinstance(metadata, dict)
        assert 'char_bboxes' in metadata
        assert len(metadata['char_bboxes']) == len(text)
        assert returned_text == text

    def test_rtl_vs_ltr_curves_differ(self, generator, test_font):
        """Test RTL and LTR curves produce different images."""
        text = "Direction"

        img_ltr, _ = generator.render_curved_text(
            text, test_font, curve_type='arc', curve_intensity=0.3
        )
        img_rtl, _ = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3
        )

        # Images should differ (mirrored curves)
        assert img_ltr.tobytes() != img_rtl.tobytes()


class TestRTLCurveVariations:
    """Test various RTL curve configurations."""

    def test_high_intensity_rtl(self, generator, test_font):
        """Test high curve intensity."""
        text = "Extreme"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.9
        )

        assert isinstance(image, Image.Image)
        assert len(char_boxes) == len(text)

    def test_single_character_rtl(self, generator, test_font):
        """Test single character RTL curve."""
        text = "X"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='sine', curve_intensity=0.5
        )

        assert isinstance(image, Image.Image)
        assert len(char_boxes) == 1

    def test_long_text_rtl_curve(self, generator, test_font):
        """Test long text with RTL curve."""
        text = "This is a very long piece of text for testing"
        image, char_boxes = generator.render_right_to_left_curved(
            text, test_font, curve_type='arc', curve_intensity=0.3
        )

        assert isinstance(image, Image.Image)
        assert len(char_boxes) == len(text)
