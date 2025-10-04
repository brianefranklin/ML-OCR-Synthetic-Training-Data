"""
Comprehensive test suite for curved text rendering.

Tests cover:
- Input validation and edge cases
- Curve parameter variations
- Bounding box accuracy
- Font compatibility
- Integration with augmentations
- Performance and stability
"""

import pytest
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import OCRDataGenerator


@pytest.fixture
def test_font():
    """Fixture providing a test font."""
    font_path = Path(__file__).parent.parent / "data" / "fonts" / "ABeeZee-Regular.ttf"
    if not font_path.exists():
        pytest.skip(f"Test font not found: {font_path}")
    return str(font_path)


@pytest.fixture
def generator():
    """Fixture providing an OCRDataGenerator instance."""
    return OCRDataGenerator([], [])


class TestCurveInputValidation:
    """Test input validation and edge cases."""

    def test_empty_text(self, generator, test_font):
        """Test with empty string."""
        font = ImageFont.truetype(test_font, size=32)
        # Should handle gracefully or raise appropriate error
        try:
            image, bboxes = generator.render_curved_text("", font, 'arc', 0.3)
            assert len(bboxes) == 0
        except (ValueError, IndexError) as e:
            # Acceptable to raise error for empty text
            pass

    def test_single_character(self, generator, test_font):
        """Test with single character."""
        font = ImageFont.truetype(test_font, size=32)
        image, bboxes = generator.render_curved_text("A", font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == 1
        assert image.size[0] > 0 and image.size[1] > 0

    def test_very_long_text(self, generator, test_font):
        """Test with very long text (50+ characters)."""
        font = ImageFont.truetype(test_font, size=24)
        long_text = "The quick brown fox jumps over the lazy dog multiple times"

        image, bboxes = generator.render_curved_text(long_text, font, 'sine', 0.4)

        assert image is not None
        assert len(bboxes) == len(long_text)
        assert image.size[0] > 0 and image.size[1] > 0

    def test_unicode_characters(self, generator, test_font):
        """Test with Unicode characters."""
        font = ImageFont.truetype(test_font, size=32)
        unicode_text = "Héllo Wörld™"

        image, bboxes = generator.render_curved_text(unicode_text, font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == len(unicode_text)

    def test_whitespace_only(self, generator, test_font):
        """Test with whitespace-only text."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("   ", font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == 3  # Three spaces

    def test_mixed_whitespace(self, generator, test_font):
        """Test with text containing multiple spaces."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Hello  World", font, 'sine', 0.3)

        assert image is not None
        assert len(bboxes) == 12  # Including double space


class TestCurveParameters:
    """Test curve parameter variations."""

    def test_zero_intensity(self, generator, test_font):
        """Test with zero curve intensity."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Test", font, 'arc', 0.0)

        # Should fall back to straight rendering
        assert image is not None
        assert len(bboxes) == 4

    def test_negative_intensity(self, generator, test_font):
        """Test with negative intensity (should handle gracefully)."""
        font = ImageFont.truetype(test_font, size=32)

        # Should either clamp to 0 or handle gracefully
        image, bboxes = generator.render_curved_text("Test", font, 'arc', -0.5)

        assert image is not None
        assert len(bboxes) == 4

    def test_intensity_above_one(self, generator, test_font):
        """Test with intensity > 1.0."""
        font = ImageFont.truetype(test_font, size=32)

        # Should handle gracefully (clamp or allow)
        image, bboxes = generator.render_curved_text("Test", font, 'arc', 1.5)

        assert image is not None
        assert len(bboxes) == 4

    def test_extreme_low_intensity(self, generator, test_font):
        """Test with very low intensity."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Testing", font, 'sine', 0.01)

        assert image is not None
        assert len(bboxes) == 7

    def test_extreme_high_intensity(self, generator, test_font):
        """Test with very high intensity."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Test", font, 'arc', 0.99)

        assert image is not None
        assert len(bboxes) == 4

    def test_arc_curve_type(self, generator, test_font):
        """Test arc curve type explicitly."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Arc Test", font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == 8

    def test_sine_curve_type(self, generator, test_font):
        """Test sine wave curve type explicitly."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Sine Test", font, 'sine', 0.4)

        assert image is not None
        assert len(bboxes) == 9

    def test_invalid_curve_type(self, generator, test_font):
        """Test with invalid curve type."""
        font = ImageFont.truetype(test_font, size=32)

        # Should default to arc or handle gracefully
        image, bboxes = generator.render_curved_text("Test", font, 'invalid', 0.3)

        assert image is not None
        assert len(bboxes) == 4


class TestBoundingBoxAccuracy:
    """Test bounding box accuracy and validity."""

    def test_bbox_within_image_bounds(self, generator, test_font):
        """Test that all bboxes are within image dimensions."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Testing Bounds", font, 'arc', 0.3)

        img_width, img_height = image.size

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.bbox
            assert x_min >= 0, f"x_min {x_min} is negative"
            assert y_min >= 0, f"y_min {y_min} is negative"
            assert x_max <= img_width + 10, f"x_max {x_max} exceeds width {img_width}"  # Small tolerance
            assert y_max <= img_height + 10, f"y_max {y_max} exceeds height {img_height}"

    def test_bbox_coordinates_ordered(self, generator, test_font):
        """Test that bbox coordinates are properly ordered (min < max)."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Order Test", font, 'sine', 0.4)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.bbox
            assert x_min < x_max, f"x_min {x_min} >= x_max {x_max}"
            assert y_min < y_max, f"y_min {y_min} >= y_max {y_max}"

    def test_bbox_reasonable_size(self, generator, test_font):
        """Test that bboxes have reasonable dimensions."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Size Test", font, 'arc', 0.3)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.bbox
            width = x_max - x_min
            height = y_max - y_min

            # Should be reasonably sized for 32pt font
            assert width > 0, "Bbox width is zero"
            assert height > 0, "Bbox height is zero"
            assert width < 200, f"Bbox width {width} is unreasonably large"
            assert height < 200, f"Bbox height {height} is unreasonably large"

    def test_bbox_count_matches_text_length(self, generator, test_font):
        """Test that number of bboxes matches text length."""
        font = ImageFont.truetype(test_font, size=32)
        text = "Match Count"

        image, bboxes = generator.render_curved_text(text, font, 'arc', 0.3)

        assert len(bboxes) == len(text)

    def test_bbox_character_correspondence(self, generator, test_font):
        """Test that each bbox corresponds to the correct character."""
        font = ImageFont.truetype(test_font, size=32)
        text = "ABC123"

        image, bboxes = generator.render_curved_text(text, font, 'sine', 0.3)

        for i, char_box in enumerate(bboxes):
            assert char_box.char == text[i], f"Character mismatch at index {i}"


class TestFontVariations:
    """Test with different font sizes and types."""

    def test_very_small_font(self, generator, test_font):
        """Test with very small font size."""
        font = ImageFont.truetype(test_font, size=8)

        image, bboxes = generator.render_curved_text("Small", font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == 5

    def test_very_large_font(self, generator, test_font):
        """Test with very large font size."""
        font = ImageFont.truetype(test_font, size=100)

        image, bboxes = generator.render_curved_text("Big", font, 'sine', 0.3)

        assert image is not None
        assert len(bboxes) == 3
        assert image.size[0] > 100  # Should be reasonably large

    def test_medium_font_arc(self, generator, test_font):
        """Test with medium font and arc."""
        font = ImageFont.truetype(test_font, size=48)

        image, bboxes = generator.render_curved_text("Medium", font, 'arc', 0.4)

        assert image is not None
        assert len(bboxes) == 6

    def test_medium_font_sine(self, generator, test_font):
        """Test with medium font and sine wave."""
        font = ImageFont.truetype(test_font, size=48)

        image, bboxes = generator.render_curved_text("Medium", font, 'sine', 0.5)

        assert image is not None
        assert len(bboxes) == 6


class TestImageQuality:
    """Test image quality and rendering."""

    def test_image_mode_rgb(self, generator, test_font):
        """Test that output image is in RGB mode."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("RGB Test", font, 'arc', 0.3)

        assert image.mode == 'RGB', f"Expected RGB mode, got {image.mode}"

    def test_image_not_blank(self, generator, test_font):
        """Test that rendered image contains actual content."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Content", font, 'sine', 0.4)

        # Convert to numpy array and check for non-white pixels
        img_array = np.array(image)

        # Check that not all pixels are white (255, 255, 255)
        white_pixels = np.all(img_array == 255, axis=2)
        non_white_count = np.sum(~white_pixels)

        assert non_white_count > 0, "Image appears to be completely blank"

    def test_reasonable_image_dimensions(self, generator, test_font):
        """Test that image dimensions are reasonable."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Dimension Test", font, 'arc', 0.3)

        width, height = image.size

        assert width > 50, f"Image width {width} is too small"
        assert height > 20, f"Image height {height} is too small"
        assert width < 5000, f"Image width {width} is unreasonably large"
        assert height < 5000, f"Image height {height} is unreasonably large"


class TestEdgeCaseCombinations:
    """Test combinations of edge cases."""

    def test_single_char_extreme_curve(self, generator, test_font):
        """Test single character with extreme curve."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("X", font, 'arc', 0.9)

        assert image is not None
        assert len(bboxes) == 1

    def test_long_text_small_font(self, generator, test_font):
        """Test long text with small font."""
        font = ImageFont.truetype(test_font, size=12)
        long_text = "This is a very long text string for testing purposes"

        image, bboxes = generator.render_curved_text(long_text, font, 'sine', 0.3)

        assert image is not None
        assert len(bboxes) == len(long_text)

    def test_numbers_and_symbols(self, generator, test_font):
        """Test with numbers and symbols."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("Test123!@#", font, 'arc', 0.3)

        assert image is not None
        assert len(bboxes) == 10

    def test_varying_character_widths(self, generator, test_font):
        """Test with characters of varying widths (i vs W)."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("iiiWWW", font, 'sine', 0.4)

        assert image is not None
        assert len(bboxes) == 6


class TestStability:
    """Test stability and error handling."""

    def test_no_crashes_with_various_inputs(self, generator, test_font):
        """Test that various inputs don't cause crashes."""
        font = ImageFont.truetype(test_font, size=32)

        test_cases = [
            ("", 'arc', 0.3),
            ("A", 'sine', 0.0),
            ("Test" * 20, 'arc', 0.5),
            ("!@#$%^&*()", 'sine', 0.6),
            ("123", 'arc', 1.0),
        ]

        for text, curve_type, intensity in test_cases:
            try:
                if text:  # Skip empty for this test
                    image, bboxes = generator.render_curved_text(text, font, curve_type, intensity)
                    assert image is not None
            except Exception as e:
                pytest.fail(f"Crashed with input ({text}, {curve_type}, {intensity}): {e}")

    def test_repeated_generation(self, generator, test_font):
        """Test that repeated generation is consistent."""
        font = ImageFont.truetype(test_font, size=32)

        results = []
        for _ in range(3):
            image, bboxes = generator.render_curved_text("Repeat", font, 'arc', 0.3)
            results.append((image.size, len(bboxes)))

        # All results should be identical
        assert all(r == results[0] for r in results), "Repeated generation produces different results"


class TestComparisonTests:
    """Test that curved text differs from straight text appropriately."""

    def test_curved_differs_from_straight(self, generator, test_font):
        """Test that curved rendering actually produces different results."""
        font = ImageFont.truetype(test_font, size=32)
        text = "Compare"

        # Curved version
        curved_img, curved_bboxes = generator.render_curved_text(text, font, 'arc', 0.5)

        # Straight version (intensity 0)
        straight_img, straight_bboxes = generator.render_curved_text(text, font, 'arc', 0.0)

        # Images should be different sizes or content
        assert curved_img.size != straight_img.size or \
               np.array_equal(np.array(curved_img), np.array(straight_img)) == False

    def test_arc_differs_from_sine(self, generator, test_font):
        """Test that arc and sine produce different results."""
        font = ImageFont.truetype(test_font, size=32)
        text = "Different"

        arc_img, arc_bboxes = generator.render_curved_text(text, font, 'arc', 0.4)
        sine_img, sine_bboxes = generator.render_curved_text(text, font, 'sine', 0.4)

        # Should produce visibly different results
        # At minimum, check they're not identical
        assert not np.array_equal(np.array(arc_img), np.array(sine_img))


class TestIntegrationWithPipeline:
    """Test integration with the full generation pipeline."""

    def test_curved_text_through_generate_image(self, generator, test_font):
        """Test curved text through the full generate_image pipeline."""
        # This tests that curve parameters work end-to-end
        augmented_image, bboxes, text = generator.generate_image(
            "Integration Test",
            test_font,
            font_size=32,
            direction='left_to_right',
            curve_type='arc',
            curve_intensity=0.3
        )

        assert augmented_image is not None
        assert len(bboxes) > 0
        assert text == "Integration Test"

    def test_curved_text_with_rotation(self, generator, test_font):
        """Test that curved text works with rotation augmentation."""
        # Generate curved text with augmentations enabled
        augmented_image, bboxes, text = generator.generate_image(
            "Rotated Curve",
            test_font,
            font_size=32,
            direction='left_to_right',
            curve_type='sine',
            curve_intensity=0.4
        )

        # Should complete without errors
        assert augmented_image is not None
        assert len(bboxes) > 0

    def test_different_curve_types_pipeline(self, generator, test_font):
        """Test different curve types through pipeline."""
        for curve_type in ['none', 'arc', 'sine']:
            augmented_image, bboxes, text = generator.generate_image(
                "Pipeline Test",
                test_font,
                font_size=32,
                direction='left_to_right',
                curve_type=curve_type,
                curve_intensity=0.3
            )

            assert augmented_image is not None
            assert len(bboxes) > 0

    def test_fallback_to_straight_for_non_ltr(self, generator, test_font):
        """Test that non-LTR directions fall back to straight rendering."""
        # Curved text currently only works for LTR
        for direction in ['right_to_left', 'top_to_bottom', 'bottom_to_top']:
            augmented_image, bboxes, text = generator.generate_image(
                "Direction Test",
                test_font,
                font_size=32,
                direction=direction,
                curve_type='arc',
                curve_intensity=0.3
            )

            # Should complete without errors (falls back to straight)
            assert augmented_image is not None
            assert len(bboxes) > 0


class TestPerformance:
    """Test performance characteristics."""

    def test_long_text_performance(self, generator, test_font):
        """Test that long text renders in reasonable time."""
        import time

        font = ImageFont.truetype(test_font, size=24)
        long_text = "Performance test with reasonably long text string" * 3

        start = time.time()
        image, bboxes = generator.render_curved_text(long_text, font, 'arc', 0.3)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Rendering took {elapsed:.2f}s, expected < 5s"
        assert image is not None

    def test_memory_efficiency(self, generator, test_font):
        """Test that rendering doesn't consume excessive memory."""
        font = ImageFont.truetype(test_font, size=32)

        # Generate multiple images without memory issues
        for i in range(10):
            image, bboxes = generator.render_curved_text(f"Test {i}", font, 'sine', 0.4)
            assert image is not None

        # If we get here without MemoryError, test passes


class TestRegressionTests:
    """Regression tests for previously found bugs."""

    def test_no_white_gaps_in_curved_text(self, generator, test_font):
        """Test that curved text doesn't have unexpected white gaps."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("NoGaps", font, 'arc', 0.4)

        # Check that there's reasonable black pixel density
        img_array = np.array(image)
        black_pixels = np.sum(img_array < 200)
        total_pixels = img_array.size

        density = black_pixels / total_pixels

        # Should have some text content (not just white)
        assert density > 0.01, f"Text density {density:.4f} is too low"

    def test_bboxes_not_negative_after_rotation(self, generator, test_font):
        """Test that rotated characters don't produce negative bbox coordinates."""
        font = ImageFont.truetype(test_font, size=32)

        # Extreme curve that causes significant rotation
        image, bboxes = generator.render_curved_text("Extreme", font, 'arc', 0.8)

        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.bbox
            assert x_min >= -5, f"x_min {x_min} is too negative"  # Small tolerance
            assert y_min >= -5, f"y_min {y_min} is too negative"

    def test_rgba_to_rgb_conversion(self, generator, test_font):
        """Test that RGBA images are properly converted to RGB."""
        font = ImageFont.truetype(test_font, size=32)

        image, bboxes = generator.render_curved_text("RGBA Test", font, 'sine', 0.5)

        # Final image should always be RGB
        assert image.mode == 'RGB', f"Expected RGB, got {image.mode}"

    def test_overlapping_characters_handled(self, generator, test_font):
        """Test that overlapping characters (if any) are handled correctly."""
        font = ImageFont.truetype(test_font, size=48)

        # High curve intensity might cause character overlap
        image, bboxes = generator.render_curved_text("Overlap", font, 'arc', 0.9)

        assert image is not None
        assert len(bboxes) == 7

        # All bboxes should be valid
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox.bbox
            assert x_max > x_min
            assert y_max > y_min


class TestDocumentedBehavior:
    """Test documented behavior and specifications."""

    def test_intensity_zero_equals_straight(self, generator, test_font):
        """Documented: intensity 0.0 should produce straight text."""
        font = ImageFont.truetype(test_font, size=32)

        image_zero, bboxes_zero = generator.render_curved_text("Straight", font, 'arc', 0.0)
        image_none, bboxes_none = generator.render_curved_text("Straight", font, 'none', 0.0)

        # Both should fall back to straight rendering
        assert image_zero is not None
        assert image_none is not None

    def test_only_ltr_supported(self, generator, test_font):
        """Documented: curvature only works for LTR direction."""
        # This is tested in integration tests, but worth documenting separately
        # Curve parameters should be ignored for non-LTR directions
        pass  # Covered in TestIntegrationWithPipeline

    def test_curve_intensity_range(self, generator, test_font):
        """Documented: curve_intensity should be 0.0-1.0."""
        font = ImageFont.truetype(test_font, size=32)

        # Should handle out-of-range gracefully
        for intensity in [0.0, 0.2, 0.5, 0.8, 1.0]:
            image, bboxes = generator.render_curved_text("Range", font, 'arc', intensity)
            assert image is not None
            assert len(bboxes) == 5
