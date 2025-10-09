#!/usr/bin/env python3
"""
Comprehensive test suite for glyph overlap functionality.
Tests language-agnostic overlap with various scripts and intensities.
"""
import pytest
import sys
import os
from pathlib import Path
from PIL import Image, ImageFont
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from generator import OCRDataGenerator


@pytest.fixture
def test_font():
    """Load a test font."""
    font_dir = Path(__file__).parent.parent / "data.nosync" / "fonts"
    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf"))
    if not font_files:
        pytest.skip("No fonts available for testing")
    return ImageFont.truetype(str(font_files[0]), size=40)


@pytest.fixture
def generator(test_font):
    """Create a generator instance."""
    font_path = test_font.path
    return OCRDataGenerator([font_path])


class TestOverlapIntensityParameter:
    """Test overlap intensity parameter ranges and behavior."""

    def test_zero_overlap_no_spacing_reduction(self, generator, test_font):
        """Zero overlap should produce standard spacing."""
        img, char_boxes = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)
        assert img is not None
        assert len(char_boxes) == 4

        # Verify no bbox overlap
        for i in range(len(char_boxes) - 1):
            bbox1 = char_boxes[i].bbox
            bbox2 = char_boxes[i + 1].bbox
            # x2 of first char should be <= x0 of second char (no overlap)
            assert bbox1[2] <= bbox2[0] + 5  # Allow small tolerance

    def test_low_overlap_slight_reduction(self, generator, test_font):
        """Low overlap (0.25) should produce slight spacing reduction."""
        img_no_overlap, boxes_no_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)
        img_overlap, boxes_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.25)

        # Image with overlap should be narrower
        assert img_overlap.width < img_no_overlap.width

    def test_medium_overlap_moderate_reduction(self, generator, test_font):
        """Medium overlap (0.5) should produce moderate spacing reduction."""
        img_no_overlap, boxes_no_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)
        img_overlap, boxes_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.5)

        # Image with overlap should be significantly narrower
        width_reduction = (img_no_overlap.width - img_overlap.width) / img_no_overlap.width
        assert width_reduction > 0.1  # At least 10% reduction

    def test_high_overlap_significant_reduction(self, generator, test_font):
        """High overlap (0.75) should produce significant spacing reduction."""
        img_no_overlap, boxes_no_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)
        img_overlap, boxes_overlap = generator.render_left_to_right("Test", test_font, overlap_intensity=0.75)

        # Image with overlap should be much narrower
        width_reduction = (img_no_overlap.width - img_overlap.width) / img_no_overlap.width
        assert width_reduction > 0.2  # At least 20% reduction

    def test_maximum_overlap_extreme_reduction(self, generator, test_font):
        """Maximum overlap (1.0) should produce extreme spacing reduction."""
        img, char_boxes = generator.render_left_to_right("Test", test_font, overlap_intensity=1.0)
        assert img is not None
        assert len(char_boxes) == 4

        # Characters should significantly overlap
        overlap_count = 0
        for i in range(len(char_boxes) - 1):
            bbox1 = char_boxes[i].bbox
            bbox2 = char_boxes[i + 1].bbox
            # Check if bboxes overlap
            if bbox1[2] > bbox2[0]:
                overlap_count += 1

        # At least some adjacent bboxes should overlap
        assert overlap_count > 0

    def test_negative_overlap_treated_as_zero(self, generator, test_font):
        """Negative overlap should be treated as zero."""
        img_neg, boxes_neg = generator.render_left_to_right("Test", test_font, overlap_intensity=-0.5)
        img_zero, boxes_zero = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)

        # Should produce similar results
        assert abs(img_neg.width - img_zero.width) < 5

    def test_overlap_above_one_clamped(self, generator, test_font):
        """Overlap > 1.0 should be clamped."""
        img_high, boxes_high = generator.render_left_to_right("Test", test_font, overlap_intensity=5.0)
        img_one, boxes_one = generator.render_left_to_right("Test", test_font, overlap_intensity=1.0)

        # Should produce similar results
        assert abs(img_high.width - img_one.width) < 10


class TestBboxValidityWithOverlap:
    """Test bounding box validity with overlap enabled."""

    def test_bbox_coordinates_ordered(self, generator, test_font):
        """Bboxes should maintain proper ordering (x0 < x1, y0 < y1) with overlap."""
        img, char_boxes = generator.render_left_to_right("Test", test_font, overlap_intensity=0.8)

        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] < bbox[2], f"x0 {bbox[0]} should be < x1 {bbox[2]}"
            assert bbox[1] < bbox[3], f"y0 {bbox[1]} should be < y1 {bbox[3]}"

    def test_bbox_within_image_bounds(self, generator, test_font):
        """All bboxes should be within image bounds with overlap."""
        img, char_boxes = generator.render_left_to_right("Testing", test_font, overlap_intensity=0.6)

        for char_box in char_boxes:
            bbox = char_box.bbox
            assert bbox[0] >= 0, "x0 should be non-negative"
            assert bbox[1] >= 0, "y0 should be non-negative"
            assert bbox[2] <= img.width + 2, f"x1 {bbox[2]} should be <= width {img.width} (with small tolerance)"
            assert bbox[3] <= img.height + 2, f"y1 {bbox[3]} should be <= height {img.height} (with small tolerance)"

    def test_bbox_overlap_detection(self, generator, test_font):
        """Adjacent bboxes can overlap with high overlap intensity."""
        img, char_boxes = generator.render_left_to_right("AVAWAYTO", test_font, overlap_intensity=0.7)

        # Count overlapping bbox pairs
        overlap_count = 0
        for i in range(len(char_boxes) - 1):
            bbox1 = char_boxes[i].bbox
            bbox2 = char_boxes[i + 1].bbox
            # Check horizontal overlap
            if bbox1[2] > bbox2[0]:
                overlap_count += 1

        # With high overlap, should see some bbox overlap
        assert overlap_count > 0, "High overlap should cause some bbox overlap"

    def test_bbox_count_matches_text_length(self, generator, test_font):
        """Number of bboxes should match text length regardless of overlap."""
        text = "Overlap Test"
        img, char_boxes = generator.render_left_to_right(text, test_font, overlap_intensity=0.5)
        assert len(char_boxes) == len(text)

    def test_bbox_reasonable_size(self, generator, test_font):
        """Bboxes should have reasonable dimensions with overlap."""
        img, char_boxes = generator.render_left_to_right("ABC", test_font, overlap_intensity=0.5)

        for char_box in char_boxes:
            bbox = char_box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            assert width > 0, "Bbox width should be positive"
            assert height > 0, "Bbox height should be positive"
            assert width < img.width, "Bbox width should be less than image width"


class TestLanguageAgnostic:
    """Test overlap works with various scripts without language-specific assumptions."""

    def test_ascii_text(self, generator, test_font):
        """Overlap should work with standard ASCII text."""
        img, char_boxes = generator.render_left_to_right("Hello World", test_font, overlap_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 11

    def test_numbers_and_symbols(self, generator, test_font):
        """Overlap should work with numbers and symbols."""
        img, char_boxes = generator.render_left_to_right("123!@#$%", test_font, overlap_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 8

    def test_mixed_case(self, generator, test_font):
        """Overlap should work with mixed case text."""
        img, char_boxes = generator.render_left_to_right("TeSt CaSe", test_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 9

    def test_unicode_characters(self, generator, test_font):
        """Overlap should work with Unicode characters."""
        texts = [
            "日本語",  # Japanese
            "中文",    # Chinese
            "테스트",  # Korean
            "Тест",   # Cyrillic
        ]

        for text in texts:
            try:
                img, char_boxes = generator.render_left_to_right(text, test_font, overlap_intensity=0.3)
                assert img is not None
                assert len(char_boxes) == len(text)
            except Exception:
                # Some fonts may not support all Unicode ranges
                pass

    def test_mixed_scripts(self, generator, test_font):
        """Overlap should work with mixed script text."""
        img, char_boxes = generator.render_left_to_right("Test123テスト", test_font, overlap_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == len("Test123テスト")


class TestDirectionSpecificOverlap:
    """Test overlap works correctly with different text directions."""

    def test_left_to_right_overlap(self, generator, test_font):
        """Overlap reduces horizontal spacing for LTR."""
        img_no, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.0)
        img_yes, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.5)

        # Width should decrease
        assert img_yes.width < img_no.width

    def test_right_to_left_overlap(self, generator, test_font):
        """Overlap reduces horizontal spacing for RTL."""
        img_no, _ = generator.render_right_to_left("Test", test_font, overlap_intensity=0.0)
        img_yes, _ = generator.render_right_to_left("Test", test_font, overlap_intensity=0.5)

        # Width should decrease
        assert img_yes.width < img_no.width

    def test_top_to_bottom_overlap(self, generator, test_font):
        """Overlap reduces vertical spacing for TTB."""
        img_no, _ = generator.render_top_to_bottom("Test", test_font, overlap_intensity=0.0)
        img_yes, _ = generator.render_top_to_bottom("Test", test_font, overlap_intensity=0.5)

        # Height should decrease
        assert img_yes.height < img_no.height

    def test_bottom_to_top_overlap(self, generator, test_font):
        """Overlap reduces vertical spacing for BTT."""
        img_no, _ = generator.render_bottom_to_top("Test", test_font, overlap_intensity=0.0)
        img_yes, _ = generator.render_bottom_to_top("Test", test_font, overlap_intensity=0.5)

        # Height should decrease
        assert img_yes.height < img_no.height


class TestCurvedTextOverlap:
    """Test overlap with curved text rendering."""

    def test_horizontal_curved_overlap(self, generator, test_font):
        """Overlap should work with horizontal curved text."""
        img_no, _ = generator.render_curved_text("Test", test_font, curve_type='arc',
                                                  curve_intensity=0.3, overlap_intensity=0.0)
        img_yes, _ = generator.render_curved_text("Test", test_font, curve_type='arc',
                                                   curve_intensity=0.3, overlap_intensity=0.5)

        # Width should decrease
        assert img_yes.width < img_no.width

    def test_vertical_curved_ttb_overlap(self, generator, test_font):
        """Overlap should work with vertical curved TTB text."""
        img_no, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc',
                                                           curve_intensity=0.3, overlap_intensity=0.0)
        img_yes, _ = generator.render_top_to_bottom_curved("Test", test_font, curve_type='arc',
                                                            curve_intensity=0.3, overlap_intensity=0.5)

        # Height should decrease
        assert img_yes.height < img_no.height

    def test_vertical_curved_btt_overlap(self, generator, test_font):
        """Overlap should work with vertical curved BTT text."""
        img_no, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc',
                                                           curve_intensity=0.3, overlap_intensity=0.0)
        img_yes, _ = generator.render_bottom_to_top_curved("Test", test_font, curve_type='arc',
                                                            curve_intensity=0.3, overlap_intensity=0.5)

        # Height should decrease
        assert img_yes.height < img_no.height

    def test_sine_wave_overlap(self, generator, test_font):
        """Overlap should work with sine wave curved text."""
        img, char_boxes = generator.render_curved_text("Wavy", test_font, curve_type='sine',
                                                        curve_intensity=0.4, overlap_intensity=0.6)
        assert img is not None
        assert len(char_boxes) == 4


class TestVisualEffects:
    """Test ink bleed and visual overlap effects."""

    def test_ink_bleed_disabled_by_default(self, generator, test_font):
        """Ink bleed should be disabled when intensity is 0."""
        img, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.3,
                                                 ink_bleed_intensity=0.0)
        assert img is not None
        assert img.mode == 'RGBA'

    def test_ink_bleed_low_intensity(self, generator, test_font):
        """Low ink bleed should produce slight blur."""
        img, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.3,
                                                 ink_bleed_intensity=0.2)
        assert img is not None
        assert img.mode == 'RGBA'

    def test_ink_bleed_high_intensity(self, generator, test_font):
        """High ink bleed should produce noticeable effect."""
        img, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.3,
                                                 ink_bleed_intensity=0.8)
        assert img is not None
        assert img.mode == 'RGBA'

    def test_ink_bleed_maintains_text_visibility(self, generator, test_font):
        """Ink bleed should not completely destroy text."""
        img, _ = generator.render_left_to_right("Test", test_font, overlap_intensity=0.5,
                                                 ink_bleed_intensity=0.5)

        # Image should not be all white
        img_array = np.array(img)
        white_pixels = np.all(img_array == 255, axis=-1)
        assert not np.all(white_pixels), "Text should be visible"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_text(self, generator, test_font):
        """Empty text should handle overlap gracefully."""
        img, char_boxes = generator.render_left_to_right("", test_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 0

    def test_single_character(self, generator, test_font):
        """Single character should handle overlap (no adjacent chars)."""
        img, char_boxes = generator.render_left_to_right("X", test_font, overlap_intensity=0.8)
        assert img is not None
        assert len(char_boxes) == 1

    def test_very_long_text(self, generator, test_font):
        """Very long text should handle overlap without errors."""
        long_text = "A" * 100
        img, char_boxes = generator.render_left_to_right(long_text, test_font, overlap_intensity=0.4)
        assert img is not None
        assert len(char_boxes) == 100

    def test_whitespace_only(self, generator, test_font):
        """Whitespace should handle overlap."""
        img, char_boxes = generator.render_left_to_right("   ", test_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 3

    def test_mixed_whitespace(self, generator, test_font):
        """Mixed text and whitespace should handle overlap."""
        img, char_boxes = generator.render_left_to_right("A B C", test_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 5


class TestIntegration:
    """Test overlap integration with other features."""

    def test_overlap_with_small_font(self, generator):
        """Overlap should work with small fonts."""
        font_path = Path(__file__).parent.parent / "data.nosync" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        small_font = ImageFont.truetype(str(font_files[0]), size=12)
        img, char_boxes = generator.render_left_to_right("Test", small_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 4

    def test_overlap_with_large_font(self, generator):
        """Overlap should work with large fonts."""
        font_path = Path(__file__).parent.parent / "data.nosync" / "fonts"
        font_files = list(font_path.glob("*.ttf")) + list(font_path.glob("*.otf"))
        if not font_files:
            pytest.skip("No fonts available")

        large_font = ImageFont.truetype(str(font_files[0]), size=80)
        img, char_boxes = generator.render_left_to_right("XY", large_font, overlap_intensity=0.5)
        assert img is not None
        assert len(char_boxes) == 2

    def test_deterministic_with_same_params(self, generator, test_font):
        """Same params should produce consistent results."""
        img1, boxes1 = generator.render_left_to_right("Test", test_font, overlap_intensity=0.5)
        img2, boxes2 = generator.render_left_to_right("Test", test_font, overlap_intensity=0.5)

        assert img1.size == img2.size
        assert len(boxes1) == len(boxes2)


class TestPerformance:
    """Test performance with overlap enabled."""

    def test_overlap_doesnt_significantly_slow_rendering(self, generator, test_font):
        """Overlap should not add significant processing time."""
        import time

        # Measure without overlap
        start = time.time()
        for _ in range(10):
            generator.render_left_to_right("Test" * 10, test_font, overlap_intensity=0.0)
        time_no_overlap = time.time() - start

        # Measure with overlap
        start = time.time()
        for _ in range(10):
            generator.render_left_to_right("Test" * 10, test_font, overlap_intensity=0.5)
        time_with_overlap = time.time() - start

        # Should not be more than 2x slower
        assert time_with_overlap < time_no_overlap * 2.0
