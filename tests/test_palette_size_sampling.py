"""Unit tests for per-glyph palette size sampling.

These tests verify that the per_glyph color mode correctly uses the configured
palette size min/max parameters rather than using len(text).
"""

import pytest
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification


def test_per_glyph_palette_respects_fixed_size():
    """Verify that per_glyph mode uses palette_size from config, not len(text)."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=2,
        per_glyph_palette_size_max=2,
        text_color_min=(0, 0, 0),
        text_color_max=(255, 255, 255),
    )

    # Generate palette for a long text string
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 characters
    palette = generator._generate_color_palette(spec, text)

    # Palette should have exactly 2 colors, not 26
    assert isinstance(palette, list), "Palette should be a list"
    assert len(palette) == 2, f"Expected palette size 2, got {len(palette)}"


def test_per_glyph_palette_size_3():
    """Verify that palette_size=3 generates exactly 3 colors."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_3_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=3,
        per_glyph_palette_size_max=3,
        text_color_min=(0, 0, 0),
        text_color_max=(255, 255, 255),
    )

    text = "HELLO WORLD"
    palette = generator._generate_color_palette(spec, text)

    assert len(palette) == 3, f"Expected palette size 3, got {len(palette)}"


def test_per_glyph_palette_size_5():
    """Verify that palette_size=5 generates exactly 5 colors."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_5_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=5,
        per_glyph_palette_size_max=5,
        text_color_min=(0, 0, 0),
        text_color_max=(255, 255, 255),
    )

    text = "A"  # Single character
    palette = generator._generate_color_palette(spec, text)

    # Even for single character, palette should have 5 colors
    assert len(palette) == 5, f"Expected palette size 5, got {len(palette)}"


def test_per_glyph_palette_respects_min_max_range():
    """Verify that palette size is sampled from min/max range."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_range_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=2,
        per_glyph_palette_size_max=5,
        text_color_min=(0, 0, 0),
        text_color_max=(255, 255, 255),
    )

    text = "TESTING"
    palette_sizes = []

    # Generate multiple palettes to verify sampling
    for _ in range(30):
        palette = generator._generate_color_palette(spec, text)
        palette_sizes.append(len(palette))

        # Each palette size should be in range [2, 5]
        assert 2 <= len(palette) <= 5, \
            f"Palette size {len(palette)} out of range [2, 5]"

    # Verify we get some variety (not always the same size)
    unique_sizes = set(palette_sizes)
    assert len(unique_sizes) > 1, \
        "Should get variety in palette sizes across multiple generations"


def test_per_glyph_palette_colors_are_rgb_tuples():
    """Verify that each color in palette is an RGB tuple."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_format_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=3,
        per_glyph_palette_size_max=3,
        text_color_min=(50, 100, 150),
        text_color_max=(100, 150, 200),
    )

    text = "TEST"
    palette = generator._generate_color_palette(spec, text)

    assert len(palette) == 3

    for color in palette:
        assert isinstance(color, tuple), f"Color should be tuple, got {type(color)}"
        assert len(color) == 3, f"Color should be RGB (3 values), got {len(color)}"

        r, g, b = color
        assert isinstance(r, int) and 0 <= r <= 255
        assert isinstance(g, int) and 0 <= g <= 255
        assert isinstance(b, int) and 0 <= b <= 255


def test_per_glyph_palette_colors_respect_min_max():
    """Verify that colors are sampled within min/max ranges."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="palette_color_range_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="per_glyph",
        per_glyph_palette_size_min=4,
        per_glyph_palette_size_max=4,
        text_color_min=(50, 100, 150),
        text_color_max=(100, 150, 200),
    )

    text = "TEST"

    # Generate multiple palettes to verify color ranges
    for _ in range(20):
        palette = generator._generate_color_palette(spec, text)

        for color in palette:
            r, g, b = color
            assert 50 <= r <= 100, f"Red {r} out of range [50, 100]"
            assert 100 <= g <= 150, f"Green {g} out of range [100, 150]"
            assert 150 <= b <= 200, f"Blue {b} out of range [150, 200]"


def test_single_color_mode_still_works():
    """Verify that single color mode (color_mode='uniform') still works correctly."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="single_color_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="uniform",  # Default single color mode
        text_color_min=(255, 0, 0),
        text_color_max=(255, 0, 0),
    )

    text = "HELLO"
    palette = generator._generate_color_palette(spec, text)

    # Single color mode returns a single RGB tuple, not a list
    assert isinstance(palette, tuple) or palette is None, \
        "Single color mode should return tuple or None"

    if palette is not None:
        assert len(palette) == 3, "Single color should be RGB tuple"


def test_gradient_mode_still_works():
    """Verify that gradient mode still works correctly."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="gradient_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        color_mode="gradient",
        gradient_start_color_min=(255, 0, 0),
        gradient_start_color_max=(255, 0, 0),
        gradient_end_color_min=(0, 0, 255),
        gradient_end_color_max=(0, 0, 255),
    )

    text = "HELLO"
    palette = generator._generate_color_palette(spec, text)

    # Gradient mode returns list of 2 colors [start, end]
    assert isinstance(palette, list), "Gradient mode should return list"
    assert len(palette) == 2, "Gradient should have start and end colors"
    assert palette[0] == (255, 0, 0), "Start color should be red"
    assert palette[1] == (0, 0, 255), "End color should be blue"
