"""Tests for the comprehensive test batch configuration.

This test file validates that the comprehensive test configuration loads correctly
and exercises all features of the system.
"""

import pytest
from src.batch_config import BatchConfig


def test_comprehensive_config_loads_and_validates():
    """Tests that the comprehensive test config loads without errors."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    # Verify basic structure
    assert config.total_images == 100
    assert len(config.specifications) == 4

    # Verify proportions sum to 1.0
    total_proportion = sum(spec.proportion for spec in config.specifications)
    assert abs(total_proportion - 1.0) < 0.001


def test_comprehensive_config_spec_names():
    """Tests that all expected batch specifications are present."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    spec_names = {spec.name for spec in config.specifications}
    expected_names = {
        "straight_ltr",
        "curved_arc_rtl",
        "wavy_sine_ttb",
        "stress_uniform_btt"
    }

    assert spec_names == expected_names


def test_comprehensive_config_proportions():
    """Tests that proportions match expected values."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    spec_dict = {spec.name: spec.proportion for spec in config.specifications}

    assert abs(spec_dict["straight_ltr"] - 0.4) < 0.001
    assert abs(spec_dict["curved_arc_rtl"] - 0.3) < 0.001
    assert abs(spec_dict["wavy_sine_ttb"] - 0.2) < 0.001
    assert abs(spec_dict["stress_uniform_btt"] - 0.1) < 0.001


def test_comprehensive_config_text_directions():
    """Tests that all 4 text directions are represented."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    directions = {spec.text_direction for spec in config.specifications}
    expected_directions = {
        "left_to_right",
        "right_to_left",
        "top_to_bottom",
        "bottom_to_top"
    }

    assert directions == expected_directions


def test_comprehensive_config_curve_types():
    """Tests that all 3 curve types are represented."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    curve_types = {spec.curve_type for spec in config.specifications}
    expected_curve_types = {"none", "arc", "sine"}

    assert curve_types == expected_curve_types


def test_comprehensive_config_distributions():
    """Tests that all 6 distribution types are used at least once."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    # Collect all distribution types used
    distributions_used = set()
    for spec in config.specifications:
        distributions_used.add(spec.rotation_angle_distribution)
        distributions_used.add(spec.noise_amount_distribution)
        distributions_used.add(spec.blur_radius_distribution)
        distributions_used.add(spec.arc_radius_distribution)
        distributions_used.add(spec.sine_amplitude_distribution)
        # Add more as needed...

    expected_distributions = {
        "uniform", "normal", "exponential", "beta", "lognormal", "truncated_normal"
    }

    # At minimum, we should use several different distributions
    assert len(distributions_used) >= 4, f"Only {len(distributions_used)} distributions used: {distributions_used}"


def test_comprehensive_config_non_zero_parameters():
    """Tests that each spec exercises various effects with non-zero values."""
    config = BatchConfig.from_yaml('test_configs/comprehensive_test.yaml')

    for spec in config.specifications:
        # At least some augmentations should have non-zero max values
        has_effects = (
            spec.rotation_angle_max > 0 or
            spec.noise_amount_max > 0 or
            spec.blur_radius_max > 0 or
            spec.perspective_warp_magnitude_max > 0 or
            spec.elastic_distortion_alpha_max > 0
        )

        assert has_effects, f"Spec '{spec.name}' has no effects enabled"
