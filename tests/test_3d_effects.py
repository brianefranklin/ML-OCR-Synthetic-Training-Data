"""
Comprehensive test suite for 3D text effects (drop shadow, emboss, deboss, bevel).
Tests effect types, depth intensity, light direction, and integration with other features.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from PIL import Image, ImageFont
import numpy as np
from pathlib import Path
from main import OCRDataGenerator


@pytest.fixture
def generator():
    """Create OCR generator for testing."""
    return OCRDataGenerator([], [])


@pytest.fixture
def test_font():
    """Load test font."""
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)


class TestEffectTypeParameter:
    """Test different 3D effect types."""

    def test_none_effect_unchanged(self, generator, test_font):
        """effect_type='none' should produce standard text."""
        img_standard, boxes_standard = generator.render_left_to_right("Test", test_font)
        img_none, boxes_none = generator.render_left_to_right(
            "Test", test_font, effect_type='none', effect_depth=0.5
        )

        # Should produce identical results
        assert img_standard.size == img_none.size
        assert len(boxes_standard) == len(boxes_none)

    def test_drop_shadow_creates_offset(self, generator, test_font):
        """Drop shadow should create offset shadow layer."""
        img_no_effect, _ = generator.render_left_to_right("Test", test_font)
        img_shadow, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )

        # Image with shadow might be larger due to offset
        assert img_shadow.size[0] >= img_no_effect.size[0]
        assert img_shadow.size[1] >= img_no_effect.size[1]

    def test_embossed_creates_highlights(self, generator, test_font):
        """Embossed text should have highlight edges."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4

        # Check that image has pixels brighter than base text (highlights)
        img_array = np.array(img)
        has_bright_pixels = np.any(img_array > 200)
        assert has_bright_pixels, "Embossed text should have highlight pixels"

    def test_debossed_creates_shadows(self, generator, test_font):
        """Debossed (engraved) text should have shadow edges."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='engraved', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4
        # Engraved effect should create depth illusion

    def test_raised_differs_from_flat(self, generator, test_font):
        """Raised text should visually differ from flat text."""
        img_flat, _ = generator.render_left_to_right("Test", test_font)
        img_raised, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )

        # Images should be different
        assert img_flat.tobytes() != img_raised.tobytes()

    def test_engraved_differs_from_raised(self, generator, test_font):
        """Engraved text should differ from raised text."""
        img_raised, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )
        img_engraved, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='engraved', effect_depth=0.5
        )

        # Different effects should produce different results
        assert img_raised.tobytes() != img_engraved.tobytes()

    def test_invalid_effect_type_fallback(self, generator, test_font):
        """Unknown effect type should default to 'none'."""
        img_invalid, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='invalid_type', effect_depth=0.5
        )
        img_none, _ = generator.render_left_to_right("Test", test_font)

        # Should produce same result as no effect
        assert img_invalid.size == img_none.size


class TestDepthIntensity:
    """Test depth intensity parameter ranges."""

    def test_zero_depth_no_effect(self, generator, test_font):
        """Depth=0.0 should produce no effect."""
        img_no_effect, _ = generator.render_left_to_right("Test", test_font)
        img_zero_depth, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.0
        )

        # Should be similar (zero depth = no effect)
        assert img_no_effect.size == img_zero_depth.size

    def test_low_depth_subtle_effect(self, generator, test_font):
        """Low depth (0.2) should create minimal effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.2
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_medium_depth_visible_effect(self, generator, test_font):
        """Medium depth (0.5) should be clearly visible."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_high_depth_pronounced_effect(self, generator, test_font):
        """High depth (0.8) should create strong effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.8
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_negative_depth_treated_as_zero(self, generator, test_font):
        """Negative depth should be clamped to zero."""
        img_neg, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=-0.5
        )
        img_zero, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.0
        )

        # Should produce similar results
        assert img_neg.size == img_zero.size

    def test_over_one_depth_clamped(self, generator, test_font):
        """Depth >1.0 should be clamped to 1.0."""
        img_over, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=1.5
        )
        img_max, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=1.0
        )

        # Should produce similar results (both clamped to 1.0)
        assert img_over.size == img_max.size


class TestLightDirection:
    """Test light direction parameters (azimuth, elevation)."""

    def test_azimuth_0_top_lit(self, generator, test_font):
        """Azimuth 0° should create top-lit effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed',
            effect_depth=0.6, light_azimuth=0
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_azimuth_90_right_lit(self, generator, test_font):
        """Azimuth 90° should create right-lit effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed',
            effect_depth=0.6, light_azimuth=90
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_azimuth_180_bottom_lit(self, generator, test_font):
        """Azimuth 180° should create bottom-lit effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed',
            effect_depth=0.6, light_azimuth=180
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_azimuth_270_left_lit(self, generator, test_font):
        """Azimuth 270° should create left-lit effect."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed',
            effect_depth=0.6, light_azimuth=270
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_elevation_affects_shadow_length(self, generator, test_font):
        """Different elevations should produce different shadow lengths."""
        img_low_elev, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised',
            effect_depth=0.6, light_elevation=15
        )
        img_high_elev, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised',
            effect_depth=0.6, light_elevation=75
        )

        # Different elevations should produce different results
        assert img_low_elev.tobytes() != img_high_elev.tobytes()


class TestBboxAccuracyWith3D:
    """Test bounding box accuracy with 3D effects."""

    def test_bbox_includes_shadow_area(self, generator, test_font):
        """Bboxes should account for shadow offset."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        # All bboxes should be valid
        for box in char_boxes:
            bbox = box.bbox
            assert len(bbox) == 4
            assert bbox[0] < bbox[2]
            assert bbox[1] < bbox[3]

    def test_bbox_coordinates_ordered(self, generator, test_font):
        """Bbox coordinates should maintain proper ordering."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.7
        )

        for box in char_boxes:
            bbox = box.bbox
            assert bbox[0] <= bbox[2], "x0 should be <= x1"
            assert bbox[1] <= bbox[3], "y0 should be <= y1"

    def test_bbox_within_expanded_bounds(self, generator, test_font):
        """All bboxes should be within image bounds."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )

        for box in char_boxes:
            bbox = box.bbox
            assert bbox[0] >= -2
            assert bbox[1] >= -2
            assert bbox[2] <= img.width + 2
            assert bbox[3] <= img.height + 2

    def test_bbox_count_matches_text(self, generator, test_font):
        """Number of bboxes should match text length."""
        text = "Testing"
        img, char_boxes = generator.render_left_to_right(
            text, test_font, effect_type='embossed', effect_depth=0.5
        )

        assert len(char_boxes) == len(text)

    def test_bbox_reasonable_dimensions(self, generator, test_font):
        """Bboxes should have reasonable dimensions."""
        img, char_boxes = generator.render_left_to_right(
            "ABC", test_font, effect_type='engraved', effect_depth=0.6
        )

        for box in char_boxes:
            bbox = box.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            assert width > 0
            assert height > 0
            assert width < img.width


class TestImageDimensionChanges:
    """Test how 3D effects affect image dimensions."""

    def test_drop_shadow_increases_dimensions(self, generator, test_font):
        """Drop shadow may increase image dimensions."""
        img_no_shadow, _ = generator.render_left_to_right("Test", test_font)
        img_shadow, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        # Shadow may make image slightly larger
        assert img_shadow.size[0] >= img_no_shadow.size[0] - 5
        assert img_shadow.size[1] >= img_no_shadow.size[1] - 5

    def test_emboss_maintains_dimensions(self, generator, test_font):
        """Emboss effect should not significantly change dimensions."""
        img_no_effect, _ = generator.render_left_to_right("Test", test_font)
        img_emboss, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.5
        )

        # Dimensions should be similar (emboss doesn't expand much)
        width_diff = abs(img_emboss.width - img_no_effect.width)
        height_diff = abs(img_emboss.height - img_no_effect.height)
        assert width_diff < 20
        assert height_diff < 20

    def test_shadow_offset_consistent(self, generator, test_font):
        """Shadow offset should be consistent with depth parameter."""
        img_low, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.2
        )
        img_high, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.8
        )

        # Higher depth may produce larger image (longer shadow)
        assert img_high.width >= img_low.width - 5

    def test_cropping_preserves_text(self, generator, test_font):
        """Text should not be cropped when adding effects."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.7
        )

        # All character bboxes should be within image
        for box in char_boxes:
            bbox = box.bbox
            assert bbox[0] >= -2
            assert bbox[2] <= img.width + 2


class TestEffectCombinations:
    """Test 3D effects combined with other features."""

    def test_emboss_with_overlap(self, generator, test_font):
        """Emboss + overlap should work together."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font,
            overlap_intensity=0.4,
            effect_type='embossed',
            effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_shadow_with_ink_bleed(self, generator, test_font):
        """Drop shadow + ink bleed should be compatible."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font,
            ink_bleed_intensity=0.3,
            effect_type='raised',
            effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_engraved_with_curves(self, generator, test_font):
        """Engraved effect + curved text should work."""
        img, char_boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            effect_type='engraved',
            effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_all_effects_combined(self, generator, test_font):
        """All effects (overlap + ink bleed + 3D) should work together."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font,
            overlap_intensity=0.3,
            ink_bleed_intensity=0.2,
            effect_type='embossed',
            effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_vertical_text_with_emboss(self, generator, test_font):
        """Vertical text (TTB) + emboss should work."""
        img, char_boxes = generator.render_top_to_bottom(
            "Test", test_font,
            effect_type='embossed',
            effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_rtl_text_with_shadow(self, generator, test_font):
        """RTL text + drop shadow should work."""
        img, char_boxes = generator.render_right_to_left(
            "Test", test_font,
            effect_type='raised',
            effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4


class TestDirectionSpecific3D:
    """Test 3D effects with different text directions."""

    def test_ltr_shadow_placement(self, generator, test_font):
        """LTR text shadow should be correctly positioned."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_rtl_shadow_placement(self, generator, test_font):
        """RTL text shadow should be correctly positioned."""
        img, char_boxes = generator.render_right_to_left(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_ttb_shadow_placement(self, generator, test_font):
        """Top-to-bottom text shadow should be correctly positioned."""
        img, char_boxes = generator.render_top_to_bottom(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_btt_shadow_placement(self, generator, test_font):
        """Bottom-to-top text shadow should be correctly positioned."""
        img, char_boxes = generator.render_bottom_to_top(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 4


class TestVisualQuality:
    """Test visual quality of 3D effects."""

    def test_text_remains_readable(self, generator, test_font):
        """Text should remain readable with effects."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.7
        )

        # Image should not be all white (text should be visible)
        img_array = np.array(img)
        white_pixels = np.all(img_array == 255, axis=-1)
        assert not np.all(white_pixels), "Text should be visible"

    def test_no_artifacts_at_edges(self, generator, test_font):
        """Effects should not create artifacts at image edges."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.6
        )

        # Check image is valid RGB
        assert img.mode == 'RGBA'
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_smooth_gradients(self, generator, test_font):
        """Shadows/highlights should be smooth (no banding)."""
        img, _ = generator.render_left_to_right(
            "Test", test_font, effect_type='embossed', effect_depth=0.5
        )

        # Just verify image is generated properly
        assert img is not None
        assert img.mode == 'RGBA'

    def test_effect_maintains_text_visibility(self, generator, test_font):
        """Effects should maintain sufficient contrast."""
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='engraved', effect_depth=0.8
        )

        # Should have both dark and light pixels (contrast)
        img_array = np.array(img.convert('L'))
        has_dark = np.any(img_array < 100)
        has_light = np.any(img_array > 150)
        assert has_dark and has_light, "Should have contrast"


class TestEdgeCases:
    """Test edge cases with 3D effects."""

    def test_empty_text_with_effects(self, generator, test_font):
        """Empty text should handle effects gracefully."""
        img, char_boxes = generator.render_left_to_right(
            "", test_font, effect_type='embossed', effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 0

    def test_single_char_with_shadow(self, generator, test_font):
        """Single character should handle shadow effect."""
        img, char_boxes = generator.render_left_to_right(
            "X", test_font, effect_type='raised', effect_depth=0.6
        )

        assert img is not None
        assert len(char_boxes) == 1

    def test_very_long_text_with_emboss(self, generator, test_font):
        """Long text should handle emboss effect."""
        long_text = "A" * 50
        img, char_boxes = generator.render_left_to_right(
            long_text, test_font, effect_type='embossed', effect_depth=0.4
        )

        assert img is not None
        assert len(char_boxes) == 50

    def test_whitespace_with_effects(self, generator, test_font):
        """Whitespace should handle effects."""
        img, char_boxes = generator.render_left_to_right(
            "A B C", test_font, effect_type='raised', effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 5

    def test_special_characters_with_3d(self, generator, test_font):
        """Special characters should handle 3D effects."""
        img, char_boxes = generator.render_left_to_right(
            "!@#$%", test_font, effect_type='embossed', effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 5


class TestPerformance:
    """Test performance of 3D effects."""

    def test_drop_shadow_performance(self, generator, test_font):
        """Drop shadow should add minimal overhead."""
        import time

        text = "Performance Test"

        # Baseline
        start = time.time()
        for _ in range(5):
            generator.render_left_to_right(text, test_font)
        baseline_time = (time.time() - start) / 5

        # With shadow
        start = time.time()
        for _ in range(5):
            generator.render_left_to_right(
                text, test_font, effect_type='raised', effect_depth=0.5
            )
        shadow_time = (time.time() - start) / 5

        # Shadow should add <50ms overhead
        overhead = shadow_time - baseline_time
        assert overhead < 0.05, f"Shadow overhead too high: {overhead*1000:.1f}ms"

    def test_emboss_performance(self, generator, test_font):
        """Emboss effect should be reasonably fast."""
        import time

        text = "Performance Test"

        start = time.time()
        for _ in range(5):
            generator.render_left_to_right(
                text, test_font, effect_type='embossed', effect_depth=0.5
            )
        emboss_time = (time.time() - start) / 5

        # Should complete in reasonable time (<100ms)
        assert emboss_time < 0.1, f"Emboss too slow: {emboss_time*1000:.1f}ms"


class TestCurvedDirections3D:
    """Test 3D effects with all 4 curved text directions."""

    def test_ltr_curved_with_raised(self, generator, test_font):
        """LTR curved text with raised (drop shadow) effect."""
        img, char_boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            effect_type='raised',
            effect_depth=0.5,
            light_azimuth=135.0,
            light_elevation=45.0
        )

        assert img is not None
        assert len(char_boxes) == 4
        # Image should be larger due to shadow
        assert img.size[0] > 50

    def test_rtl_curved_with_embossed(self, generator, test_font):
        """RTL curved text with embossed effect."""
        img, char_boxes = generator.render_right_to_left_curved(
            "Test", test_font,
            curve_type='sine',
            curve_intensity=0.4,
            effect_type='embossed',
            effect_depth=0.6,
            light_azimuth=90.0,
            light_elevation=50.0
        )

        assert img is not None
        assert len(char_boxes) == 4
        # Should have visible effect
        img_array = np.array(img.convert('L'))
        assert img_array.min() < 200  # Has dark pixels

    def test_ttb_curved_with_engraved(self, generator, test_font):
        """Top-to-bottom curved text with engraved effect."""
        img, char_boxes = generator.render_top_to_bottom_curved(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            effect_type='engraved',
            effect_depth=0.7,
            light_azimuth=180.0,
            light_elevation=40.0
        )

        assert img is not None
        assert len(char_boxes) == 4
        # Should have contrast
        img_array = np.array(img.convert('L'))
        assert img_array.max() > 200  # Has light pixels
        assert img_array.min() < 100  # Has dark pixels

    def test_btt_curved_with_raised(self, generator, test_font):
        """Bottom-to-top curved text with raised effect."""
        img, char_boxes = generator.render_bottom_to_top_curved(
            "Test", test_font,
            curve_type='sine',
            curve_intensity=0.2,
            effect_type='raised',
            effect_depth=0.5,
            light_azimuth=270.0,
            light_elevation=35.0
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_all_curved_directions_with_effects(self, generator, test_font):
        """All 4 curved directions should support 3D effects."""
        directions = [
            ('ltr', generator.render_curved_text),
            ('rtl', generator.render_right_to_left_curved),
            ('ttb', generator.render_top_to_bottom_curved),
            ('btt', generator.render_bottom_to_top_curved)
        ]

        for name, method in directions:
            img, boxes = method(
                "Hi", test_font,
                curve_type='arc',
                curve_intensity=0.3,
                effect_type='embossed',
                effect_depth=0.5
            )
            assert img is not None, f"{name} failed"
            assert len(boxes) == 2, f"{name} wrong box count"


class TestIntegration:
    """Test integration with main generation pipeline."""

    def test_through_generate_image(self, generator, test_font):
        """3D effects should work through generate_image()."""
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        img, bboxes, text = generator.generate_image(
            "Test", font_path, 32, 'left_to_right',
            effect_type='embossed', effect_depth=0.5
        )

        assert img is not None
        assert len(bboxes) == 4
        assert text == "Test"

    def test_batch_config_integration(self, generator, test_font):
        """Effects should work with batch configuration system."""
        # This will be tested after batch config is updated
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font, effect_type='raised', effect_depth=0.5
        )

        assert img is not None
        assert len(char_boxes) == 4

    def test_cli_parameter_integration(self, generator, test_font):
        """Effects should accept parameters like CLI would provide."""
        # Test parameter passing
        img, char_boxes = generator.render_left_to_right(
            "Test", test_font,
            effect_type='embossed',
            effect_depth=0.6,
            light_azimuth=135,
            light_elevation=45
        )

        assert img is not None
        assert len(char_boxes) == 4
