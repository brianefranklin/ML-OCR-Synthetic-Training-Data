"""Unit tests for shadow parameter sampling.

These tests verify that shadow options are correctly sampled from configuration
parameters and that shadows are disabled when offsets are zero.
"""

import pytest
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification


def test_drop_shadow_options_sampled_when_enabled():
    """Verify that drop shadow options are generated when offsets are non-zero."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="drop_shadow_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        drop_shadow_offset_x_min=10,
        drop_shadow_offset_x_max=10,
        drop_shadow_offset_y_min=10,
        drop_shadow_offset_y_max=10,
        drop_shadow_radius_min=2.0,
        drop_shadow_radius_max=2.0,
        drop_shadow_color_min=(0, 0, 0, 200),
        drop_shadow_color_max=(0, 0, 0, 200),
    )

    # Call the private method directly to test shadow sampling
    shadow_options = generator._generate_shadow_options(
        spec.drop_shadow_offset_x_min, spec.drop_shadow_offset_x_max, spec.drop_shadow_offset_x_distribution,
        spec.drop_shadow_offset_y_min, spec.drop_shadow_offset_y_max, spec.drop_shadow_offset_y_distribution,
        spec.drop_shadow_radius_min, spec.drop_shadow_radius_max, spec.drop_shadow_radius_distribution,
        spec.drop_shadow_color_min, spec.drop_shadow_color_max
    )

    assert shadow_options is not None, "Shadow should be enabled when offsets are non-zero"
    assert "offset" in shadow_options
    assert "radius" in shadow_options
    assert "color" in shadow_options

    # Verify values are correct
    assert shadow_options["offset"] == (10, 10)
    assert shadow_options["radius"] == 2.0
    assert shadow_options["color"] == (0, 0, 0, 200)


def test_drop_shadow_disabled_when_offsets_zero():
    """Verify that drop shadow is disabled when both offsets are 0."""
    generator = OCRDataGenerator()

    shadow_options = generator._generate_shadow_options(
        0, 0, "uniform",  # offset_x_min, max, dist
        0, 0, "uniform",  # offset_y_min, max, dist
        5.0, 5.0, "uniform",  # radius_min, max, dist
        (0, 0, 0, 255), (0, 0, 0, 255)  # color_min, max
    )

    assert shadow_options is None, "Shadow should be disabled when both offsets are 0"


def test_block_shadow_options_sampled_when_enabled():
    """Verify that block shadow options are generated when offsets are non-zero."""
    generator = OCRDataGenerator()

    spec = BatchSpecification(
        name="block_shadow_test",
        corpus_file="test.txt",
        text_direction="left_to_right",
        proportion=1.0,
        block_shadow_offset_x_min=5,
        block_shadow_offset_x_max=5,
        block_shadow_offset_y_min=5,
        block_shadow_offset_y_max=5,
        block_shadow_radius_min=1.5,
        block_shadow_radius_max=1.5,
        block_shadow_color_min=(128, 128, 128, 180),
        block_shadow_color_max=(128, 128, 128, 180),
    )

    shadow_options = generator._generate_shadow_options(
        spec.block_shadow_offset_x_min, spec.block_shadow_offset_x_max, spec.block_shadow_offset_x_distribution,
        spec.block_shadow_offset_y_min, spec.block_shadow_offset_y_max, spec.block_shadow_offset_y_distribution,
        spec.block_shadow_radius_min, spec.block_shadow_radius_max, spec.block_shadow_radius_distribution,
        spec.block_shadow_color_min, spec.block_shadow_color_max
    )

    assert shadow_options is not None
    assert shadow_options["offset"] == (5, 5)
    assert shadow_options["radius"] == 1.5
    assert shadow_options["color"] == (128, 128, 128, 180)


def test_shadow_respects_distribution_ranges():
    """Verify that shadow sampling respects min/max ranges."""
    generator = OCRDataGenerator()

    # Sample shadows multiple times and verify all values are in range
    for _ in range(20):
        shadow_options = generator._generate_shadow_options(
            -5, 5, "uniform",  # offset_x range
            -10, 10, "uniform",  # offset_y range
            1.0, 5.0, "uniform",  # radius range
            (50, 100, 150, 100), (100, 150, 200, 255)  # color range
        )

        # Shadow might be None if both offsets sampled to 0
        if shadow_options is not None:
            offset_x, offset_y = shadow_options["offset"]
            radius = shadow_options["radius"]
            r, g, b, a = shadow_options["color"]

            assert -5 <= offset_x <= 5, f"offset_x {offset_x} out of range"
            assert -10 <= offset_y <= 10, f"offset_y {offset_y} out of range"
            assert 1.0 <= radius <= 5.0, f"radius {radius} out of range"
            assert 50 <= r <= 100, f"Red {r} out of range"
            assert 100 <= g <= 150, f"Green {g} out of range"
            assert 150 <= b <= 200, f"Blue {b} out of range"
            assert 100 <= a <= 255, f"Alpha {a} out of range"


def test_shadow_with_only_x_offset_is_enabled():
    """Verify shadow is enabled when only X offset is non-zero."""
    generator = OCRDataGenerator()

    shadow_options = generator._generate_shadow_options(
        10, 10, "uniform",  # offset_x = 10
        0, 0, "uniform",   # offset_y = 0
        2.0, 2.0, "uniform",
        (0, 0, 0, 255), (0, 0, 0, 255)
    )

    assert shadow_options is not None, "Shadow should be enabled when X offset is non-zero"
    assert shadow_options["offset"] == (10, 0)


def test_shadow_with_only_y_offset_is_enabled():
    """Verify shadow is enabled when only Y offset is non-zero."""
    generator = OCRDataGenerator()

    shadow_options = generator._generate_shadow_options(
        0, 0, "uniform",   # offset_x = 0
        10, 10, "uniform",  # offset_y = 10
        2.0, 2.0, "uniform",
        (0, 0, 0, 255), (0, 0, 0, 255)
    )

    assert shadow_options is not None, "Shadow should be enabled when Y offset is non-zero"
    assert shadow_options["offset"] == (0, 10)
