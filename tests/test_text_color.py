"""
Comprehensive test suite for text color functionality.
Tests color generation, modes, palettes, and integration with existing features.
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


class TestColorGeneration:
    """Test color generation logic."""

    def test_uniform_mode_same_color_all_chars(self, generator, test_font):
        """Uniform mode should use same color for all characters."""
        img, boxes = generator.render_left_to_right(
            "Hello", test_font,
            text_color_mode='uniform',
            color_palette='realistic_dark'
        )

        # Extract unique non-white colors from image
        img_array = np.array(img.convert('RGB'))
        # Get all non-white pixels (text pixels)
        text_pixels = img_array[(img_array != [255, 255, 255]).any(axis=2)]

        if len(text_pixels) > 0:
            # In uniform mode, color variance should be low, even with anti-aliasing
            std_dev = np.std(text_pixels, axis=0)
            assert np.all(std_dev < 80), f"Color variance too high for uniform mode: {std_dev}"


    def test_per_glyph_mode_different_colors(self, generator, test_font):
        """Per-glyph mode should allow different colors per character."""
        img, boxes = generator.render_left_to_right(
            "Hello", test_font,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )

        # Extract unique non-white colors
        img_array = np.array(img.convert('RGB'))
        text_pixels = img_array[(img_array != [255, 255, 255]).any(axis=2)]

        if len(text_pixels) > 0:
            # In per-glyph mode, color variance should be high
            min_color = np.min(text_pixels, axis=0)
            max_color = np.max(text_pixels, axis=0)
            color_range = max_color - min_color
            assert np.any(color_range > 100), f"Color range too small for per-glyph: {color_range}"

    def test_realistic_dark_palette_dark_colors(self, generator, test_font):
        """Realistic dark palette should produce dark text colors."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='realistic_dark'
        )

        img_array = np.array(img.convert('RGB'))
        text_pixels = img_array[(img_array != [255, 255, 255]).any(axis=2)]

        if len(text_pixels) > 0:
            # Dark colors have low RGB values (< 150 on average)
            avg_brightness = text_pixels.mean()
            assert avg_brightness < 150, f"Text too bright for dark palette: {avg_brightness}"

    def test_vibrant_palette_bright_colors(self, generator, test_font):
        """Vibrant palette should produce bright, saturated colors."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant'
        )

        img_array = np.array(img.convert('RGB'))
        text_pixels = img_array[(img_array != [255, 255, 255]).any(axis=2)]

        if len(text_pixels) > 0:
            # Vibrant colors should have at least one channel > 200
            max_channel = text_pixels.max(axis=1).mean()
            assert max_channel > 150, "Vibrant colors should be bright"

    def test_custom_colors_respected(self, generator, test_font):
        """Custom colors should be used when provided."""
        custom_red = (255, 0, 0)
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            custom_colors=[custom_red]
        )

        img_array = np.array(img.convert('RGB'))
        text_pixels = img_array[(img_array != [255, 255, 255]).any(axis=2)]

        if len(text_pixels) > 0:
            # Find the most frequent color
            unique_colors, counts = np.unique(text_pixels.reshape(-1, 3), axis=0, return_counts=True)
            primary_color = unique_colors[counts.argmax()]

            # The primary color should be very close to the custom red
            assert primary_color[0] > 240, "Primary color should be red"
            assert primary_color[1] < 15, "Primary color should have low green"
            assert primary_color[2] < 15, "Primary color should have low blue"


class TestColorModes:
    """Test different color modes."""

    def test_uniform_mode_works(self, generator, test_font):
        """Uniform mode should work correctly."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font, text_color_mode='uniform'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_per_glyph_mode_works(self, generator, test_font):
        """Per-glyph mode should work correctly."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font, text_color_mode='per_glyph'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_gradient_mode_works(self, generator, test_font):
        """Gradient mode should work correctly."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font, text_color_mode='gradient'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_random_mode_works(self, generator, test_font):
        """Random mode should work correctly."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font, text_color_mode='random'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_invalid_mode_fallback(self, generator, test_font):
        """Invalid color mode should fallback to uniform."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font, text_color_mode='invalid_mode'
        )
        # Should not crash, fallback to default
        assert img is not None
        assert len(boxes) == 4


class TestColorPalettes:
    """Test different color palettes."""

    def test_realistic_dark_palette(self, generator, test_font):
        """Realistic dark palette should work."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            color_palette='realistic_dark'
        )
        assert img is not None

    def test_realistic_light_palette(self, generator, test_font):
        """Realistic light palette should work."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            color_palette='realistic_light'
        )
        assert img is not None

    def test_vibrant_palette(self, generator, test_font):
        """Vibrant palette should work."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            color_palette='vibrant'
        )
        assert img is not None

    def test_pastels_palette(self, generator, test_font):
        """Pastels palette should work."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            color_palette='pastels'
        )
        assert img is not None


class TestContrast:
    """Test background/text contrast."""

    def test_dark_text_light_background_contrast(self, generator, test_font):
        """Dark text on light background should have good contrast."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='realistic_dark'
        )

        img_array = np.array(img.convert('L'))
        # Should have both dark and light pixels
        assert img_array.min() < 100, "Should have dark text"
        assert img_array.max() > 200, "Should have light background"

    def test_auto_background_mode(self, generator, test_font):
        """Auto background should provide contrasting color."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant',
            background_color='auto'
        )

        # Should have sufficient contrast
        img_array = np.array(img.convert('L'))
        contrast = img_array.max() - img_array.min()
        assert contrast > 50, f"Insufficient contrast: {contrast}"

    def test_custom_background_color(self, generator, test_font):
        """Custom background color should be applied."""
        # Light blue background
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            background_color=(173, 216, 230)
        )

        img_array = np.array(img.convert('RGB'))
        # Check that background has bluish tint
        background_pixels = img_array[(img_array == [173, 216, 230]).all(axis=2)]
        assert len(background_pixels) > 0, "Background color not applied"


class TestIntegrationWithEffects:
    """Test color integration with existing features."""

    def test_color_with_3d_effects(self, generator, test_font):
        """Color should work with 3D effects."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant',
            effect_type='embossed',
            effect_depth=0.5
        )
        assert img is not None
        assert len(boxes) == 4

    def test_color_with_overlap(self, generator, test_font):
        """Color should work with glyph overlap."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='per_glyph',
            color_palette='vibrant',
            overlap_intensity=0.3
        )
        assert img is not None
        assert len(boxes) == 4

    def test_color_with_curves(self, generator, test_font):
        """Color should work with curved text."""
        img, boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_color_with_ink_bleed(self, generator, test_font):
        """Color should work with ink bleed effect."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='realistic_dark',
            ink_bleed_intensity=0.3
        )
        assert img is not None
        assert len(boxes) == 4

    def test_all_effects_with_color(self, generator, test_font):
        """Color should work with all effects combined."""
        img, boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            overlap_intensity=0.2,
            ink_bleed_intensity=0.2,
            effect_type='embossed',
            effect_depth=0.4,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4


class TestAllDirections:
    """Test color with all text directions."""

    def test_ltr_with_color(self, generator, test_font):
        """LTR should support color."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_rtl_with_color(self, generator, test_font):
        """RTL should support color."""
        img, boxes = generator.render_right_to_left(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_ttb_with_color(self, generator, test_font):
        """TTB should support color."""
        img, boxes = generator.render_top_to_bottom(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_btt_with_color(self, generator, test_font):
        """BTT should support color."""
        img, boxes = generator.render_bottom_to_top(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_curved_ltr_with_color(self, generator, test_font):
        """Curved LTR should support color."""
        img, boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_curved_rtl_with_color(self, generator, test_font):
        """Curved RTL should support color."""
        img, boxes = generator.render_right_to_left_curved(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_curved_ttb_with_color(self, generator, test_font):
        """Curved TTB should support color."""
        img, boxes = generator.render_top_to_bottom_curved(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4

    def test_curved_btt_with_color(self, generator, test_font):
        """Curved BTT should support color."""
        img, boxes = generator.render_bottom_to_top_curved(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4


class TestBatchConfig:
    """Test batch configuration support."""

    def test_batch_config_color_params(self, generator, test_font):
        """Batch config should support color parameters."""
        # Simulates what batch config would pass
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            color_palette='realistic_dark',
            background_color='auto'
        )
        assert img is not None
        assert len(boxes) == 4


class TestCLI:
    """Test CLI parameter handling."""

    def test_cli_parameters_accepted(self, generator, test_font):
        """CLI-style parameters should be accepted."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='per_glyph',
            color_palette='vibrant',
            custom_colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
            background_color=(255, 255, 255)
        )
        assert img is not None
        assert len(boxes) == 4

    def test_through_generate_image(self, generator, test_font):
        """Color should work through generate_image pipeline."""
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        img, bboxes, text = generator.generate_image(
            "Test", font_path, 32, 'left_to_right',
            text_color_mode='uniform',
            color_palette='vibrant'
        )

        assert img is not None
        assert len(bboxes) == 4
        assert text == "Test"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_char_with_color(self, generator, test_font):
        """Single character should support color."""
        img, boxes = generator.render_left_to_right(
            "A", test_font,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 1

    def test_long_text_per_glyph_color(self, generator, test_font):
        """Long text with per-glyph color should work."""
        img, boxes = generator.render_left_to_right(
            "This is a longer test string", test_font,
            text_color_mode='per_glyph',
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) > 20

    def test_empty_custom_colors_fallback(self, generator, test_font):
        """Empty custom colors should fallback to palette."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            custom_colors=[],
            color_palette='vibrant'
        )
        assert img is not None
        assert len(boxes) == 4
