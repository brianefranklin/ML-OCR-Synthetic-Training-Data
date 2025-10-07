"""
Comprehensive test suite for RGBA transparent background pipeline.
Tests that text is rendered with transparent backgrounds throughout the entire pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from PIL import Image, ImageFont
import numpy as np
from pathlib import Path
from generator import OCRDataGenerator


@pytest.fixture
def generator():
    """Create OCR generator for testing."""
    return OCRDataGenerator([], [])


@pytest.fixture
def test_font():
    """Load test font."""
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)


class TestBasicRGBARendering:
    """Test that rendering produces RGBA images with transparency."""

    def test_left_to_right_returns_rgba(self, generator, test_font):
        """LTR rendering should return RGBA image."""
        img, boxes = generator.render_left_to_right("Test", test_font)

        assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"
        assert img.size[0] > 0 and img.size[1] > 0
        assert len(boxes) == 4

    def test_right_to_left_returns_rgba(self, generator, test_font):
        """RTL rendering should return RGBA image."""
        img, boxes = generator.render_right_to_left("Test", test_font)

        assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"

    def test_top_to_bottom_returns_rgba(self, generator, test_font):
        """TTB rendering should return RGBA image."""
        img, boxes = generator.render_top_to_bottom("Test", test_font)

        assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"

    def test_bottom_to_top_returns_rgba(self, generator, test_font):
        """BTT rendering should return RGBA image."""
        img, boxes = generator.render_bottom_to_top("Test", test_font)

        assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"

    def test_curved_text_returns_rgba(self, generator, test_font):
        """Curved text rendering should return RGBA image."""
        img, boxes = generator.render_curved_text(
            "Test", test_font, curve_type='arc', curve_intensity=0.3
        )

        assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"


class TestTransparentBackground:
    """Test that background pixels are actually transparent."""

    def test_background_has_zero_alpha(self, generator, test_font):
        """Background pixels should have alpha=0."""
        img, boxes = generator.render_left_to_right("Test", test_font)
        img_array = np.array(img)

        # Get alpha channel
        alpha = img_array[:, :, 3]

        # Should have some fully transparent pixels (background)
        transparent_pixels = np.sum(alpha == 0)
        assert transparent_pixels > 0, "Expected some transparent background pixels"

    def test_background_is_majority_transparent(self, generator, test_font):
        """Most of the image should be transparent background."""
        img, boxes = generator.render_left_to_right("Test", test_font)
        img_array = np.array(img)

        alpha = img_array[:, :, 3]
        transparent_ratio = np.sum(alpha == 0) / alpha.size

        # At least 50% should be transparent background
        assert transparent_ratio > 0.5, f"Expected >50% transparent, got {transparent_ratio*100:.1f}%"

    def test_corners_are_transparent(self, generator, test_font):
        """All four corners should be transparent (no text there)."""
        img, boxes = generator.render_left_to_right("Test", test_font)
        img_array = np.array(img)

        h, w = img_array.shape[:2]
        corners = [
            (0, 0),
            (0, w-1),
            (h-1, 0),
            (h-1, w-1)
        ]

        for y, x in corners:
            alpha_value = img_array[y, x, 3]
            assert alpha_value == 0, f"Corner ({y},{x}) should be transparent, alpha={alpha_value}"


class TestWhiteTextPreservation:
    """Test that white text is properly preserved (not made transparent)."""

    def test_white_text_is_opaque(self, generator, test_font):
        """White text should have opaque pixels."""
        # Render white text
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            custom_colors=[(255, 255, 255)]  # Pure white
        )

        img_array = np.array(img)

        # Find white pixels (RGB all 255)
        is_white = (img_array[:, :, 0] == 255) & \
                   (img_array[:, :, 1] == 255) & \
                   (img_array[:, :, 2] == 255)

        # Get alpha values of white pixels
        white_alpha = img_array[is_white, 3]

        # Some white pixels should be opaque (the text)
        opaque_white = np.sum(white_alpha == 255)
        assert opaque_white > 0, "White text pixels should be opaque (alpha=255)"

    def test_white_text_visible_on_dark_background(self, generator, test_font):
        """White text should be visible when composited on dark background."""
        from PIL import ImageDraw

        # Render white text
        text_img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            custom_colors=[(255, 255, 255)]
        )

        # Create black canvas
        canvas = Image.new('RGB', text_img.size, 'black')

        # Composite text onto canvas
        canvas.paste(text_img, (0, 0), text_img)

        # Canvas should have white pixels (the text)
        canvas_array = np.array(canvas)
        white_pixels = np.all(canvas_array == [255, 255, 255], axis=2)

        assert np.sum(white_pixels) > 0, "White text should be visible on black background"


class TestLettersWithHoles:
    """Test that letters with enclosed regions (counters) work correctly."""

    def test_letter_o_has_transparent_center(self, generator, test_font):
        """Letter 'o' should have transparent hole in the middle."""
        img, boxes = generator.render_left_to_right("o", test_font)
        img_array = np.array(img)

        # Find the bounding box of the text
        alpha = img_array[:, :, 3]
        opaque = alpha > 0

        if np.any(opaque):
            y_coords, x_coords = np.where(opaque)
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            # Check center region
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2

            # Should have some transparent pixels near center (the hole)
            center_region = alpha[center_y-5:center_y+5, center_x-5:center_x+5]
            transparent_in_center = np.sum(center_region == 0)

            assert transparent_in_center > 0, "Letter 'o' should have transparent hole in center"

    def test_letters_with_counters(self, generator, test_font):
        """Letters with enclosed regions should render correctly."""
        # Test various letters with holes
        letters_with_holes = "oadbpqeg"

        for letter in letters_with_holes:
            img, boxes = generator.render_left_to_right(letter, test_font)

            assert img.mode == 'RGBA'
            img_array = np.array(img)

            # Should have both opaque (text) and transparent (background + holes) pixels
            alpha = img_array[:, :, 3]
            has_opaque = np.any(alpha == 255)
            has_transparent = np.any(alpha == 0)

            assert has_opaque, f"Letter '{letter}' should have opaque pixels"
            assert has_transparent, f"Letter '{letter}' should have transparent pixels"


class TestAntiAliasing:
    """Test that anti-aliasing is preserved in alpha channel."""

    def test_has_semi_transparent_pixels(self, generator, test_font):
        """Should have pixels with alpha between 0 and 255 (anti-aliasing)."""
        img, boxes = generator.render_left_to_right("Test", test_font)
        img_array = np.array(img)

        alpha = img_array[:, :, 3]

        # Find semi-transparent pixels (0 < alpha < 255)
        semi_transparent = (alpha > 0) & (alpha < 255)
        count = np.sum(semi_transparent)

        assert count > 0, "Should have anti-aliased (semi-transparent) pixels"

    def test_alpha_gradient_at_edges(self, generator, test_font):
        """Alpha should have gradient values at text edges."""
        img, boxes = generator.render_left_to_right("Test", test_font)
        img_array = np.array(img)

        alpha = img_array[:, :, 3]

        # Count unique alpha values
        unique_alphas = np.unique(alpha)

        # Should have more than just 0 and 255
        assert len(unique_alphas) > 2, f"Expected gradient of alpha values, got only {unique_alphas}"


class TestEffectsWithAlpha:
    """Test that various effects preserve alpha channel."""

    def test_3d_effects_preserve_alpha(self, generator, test_font):
        """3D effects should maintain transparency."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            effect_type='embossed',
            effect_depth=0.5
        )

        assert img.mode == 'RGBA'
        img_array = np.array(img)
        alpha = img_array[:, :, 3]

        # Should still have transparent background
        assert np.any(alpha == 0), "3D effects should preserve transparent background"

    def test_ink_bleed_preserves_alpha(self, generator, test_font):
        """Ink bleed should maintain transparency."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            ink_bleed_intensity=0.5
        )

        assert img.mode == 'RGBA'
        img_array = np.array(img)
        alpha = img_array[:, :, 3]

        assert np.any(alpha == 0), "Ink bleed should preserve transparent background"

    def test_overlap_preserves_alpha(self, generator, test_font):
        """Character overlap should maintain transparency."""
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            overlap_intensity=0.5
        )

        assert img.mode == 'RGBA'
        img_array = np.array(img)
        alpha = img_array[:, :, 3]

        assert np.any(alpha == 0), "Overlap should preserve transparent background"

    def test_curves_preserve_alpha(self, generator, test_font):
        """Curved text should maintain transparency."""
        img, boxes = generator.render_curved_text(
            "Test", test_font,
            curve_type='arc',
            curve_intensity=0.3
        )

        assert img.mode == 'RGBA'
        img_array = np.array(img)
        alpha = img_array[:, :, 3]

        assert np.any(alpha == 0), "Curves should preserve transparent background"


class TestCanvasCompositing:
    """Test that RGBA text composites correctly onto canvas."""

    def test_composite_onto_white_canvas(self, generator, test_font):
        """RGBA text should composite cleanly onto white canvas."""
        from canvas_placement import place_on_canvas

        img, boxes = generator.render_left_to_right("Test", test_font)
        char_bboxes = [box.bbox for box in boxes]

        canvas_img, metadata = place_on_canvas(
            img,
            char_bboxes,
            canvas_size=(400, 300)
        )

        assert canvas_img.mode == 'RGBA'
        assert canvas_img.size == (400, 300)

    def test_no_white_halos(self, generator, test_font):
        """Should not have white halos around colored text."""
        from canvas_placement import place_on_canvas

        # Render red text
        img, boxes = generator.render_left_to_right(
            "Test", test_font,
            text_color_mode='uniform',
            custom_colors=[(255, 0, 0)]
        )
        char_bboxes = [box.bbox for box in boxes]

        canvas_img, metadata = place_on_canvas(
            img,
            char_bboxes,
            canvas_size=(400, 300)
        )

        canvas_array = np.array(canvas_img)

        # Should have red pixels (text - with high alpha)
        red_pixels = (canvas_array[:, :, 0] > 200) & \
                     (canvas_array[:, :, 1] < 50) & \
                     (canvas_array[:, :, 2] < 50) & \
                     (canvas_array[:, :, 3] > 200)

        assert np.any(red_pixels), "Should have red text pixels"

        # Should have transparent background pixels (alpha = 0)
        transparent_pixels = canvas_array[:, :, 3] == 0
        assert np.any(transparent_pixels), "Should have transparent background"

    def test_proper_alpha_blending(self, generator, test_font):
        """Semi-transparent pixels should blend correctly."""
        from canvas_placement import place_on_canvas

        img, boxes = generator.render_left_to_right("Test", test_font)
        char_bboxes = [box.bbox for box in boxes]

        # Place on canvas
        canvas_img, metadata = place_on_canvas(
            img,
            char_bboxes,
            canvas_size=(400, 300)
        )

        # Canvas should be RGBA with transparent background
        assert canvas_img.mode == 'RGBA'

        # Should have semi-transparent pixels (anti-aliasing)
        canvas_array = np.array(canvas_img)
        alpha_channel = canvas_array[:, :, 3]
        semi_transparent = (alpha_channel > 0) & (alpha_channel < 255)
        assert np.any(semi_transparent), "Should have anti-aliased semi-transparent pixels"


class TestFullPipeline:
    """Test complete generation pipeline with RGBA."""

    def test_generate_image_returns_rgba_metadata(self, generator, test_font):
        """generate_image should work with RGBA pipeline."""
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        final_img, metadata, text, augmentations_applied = generator.generate_image(
            "Test", font_path, 32, 'left_to_right'
        )

        # Final image should be RGBA (canvas now uses RGBA mode)
        assert final_img.mode == 'RGBA'

        # Should have metadata
        assert 'canvas_size' in metadata
        assert 'char_bboxes' in metadata
        assert text == "Test"

    def test_white_text_through_full_pipeline(self, generator):
        """White text should work through entire pipeline."""
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        final_img, metadata, text, augmentations_applied = generator.generate_image(
            "Test", font_path, 32, 'left_to_right',
            text_color_mode='uniform',
            custom_colors=[(255, 255, 255)]
        )

        # Should produce valid image
        assert final_img.mode == 'RGBA'
        assert final_img.size[0] > 0 and final_img.size[1] > 0

        # Should have metadata
        assert len(metadata['char_bboxes']) == 4
