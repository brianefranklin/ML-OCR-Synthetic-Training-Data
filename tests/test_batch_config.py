"""
Tests for the batch_config module.
"""

import pytest
from pathlib import Path
from src.batch_config import BatchConfig, BatchSpecification

def test_load_batch_config_from_yaml(tmp_path: Path):
    """
    Tests that a BatchConfig object can be successfully loaded from a YAML file,
    and all parameters are correctly parsed into their respective dataclasses.
    """
    yaml_content = """
total_images: 100
specifications:
  - name: "ancient_ltr_sample"
    proportion: 0.5
    text_direction: "left_to_right"
    corpus_file: "data.nosync/corpus_text/ltr/ancient_script_1.txt"
  - name: "ancient_rtl_sample"
    proportion: 0.5
    text_direction: "right_to_left"
    corpus_file: "data.nosync/corpus_text/rtl/ancient_script_2.txt"
"""
    yaml_file = tmp_path / "test_batch.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # This is the call to the code we are about to write.
    batch_config = BatchConfig.from_yaml(str(yaml_file))

    # Assertions to verify the loaded data
    assert batch_config.total_images == 100
    assert len(batch_config.specifications) == 2

    # Check the first specification
    spec1 = batch_config.specifications[0]
    assert isinstance(spec1, BatchSpecification)
    assert spec1.name == "ancient_ltr_sample"
    assert spec1.proportion == 0.5
    assert spec1.text_direction == "left_to_right"
    assert spec1.corpus_file == "data.nosync/corpus_text/ltr/ancient_script_1.txt"

    # Check the second specification
    spec2 = batch_config.specifications[1]
    assert isinstance(spec2, BatchSpecification)
    assert spec2.name == "ancient_rtl_sample"
    assert spec2.proportion == 0.5
    assert spec2.text_direction == "right_to_left"
    assert spec2.corpus_file == "data.nosync/corpus_text/rtl/ancient_script_2.txt"

def test_batch_specification_has_curve_parameters():
    """
    Tests that BatchSpecification includes all curve-related parameters with
    appropriate defaults, ensuring consistent feature vectors for ML analysis.
    """
    spec = BatchSpecification(
        name="test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt"
    )

    # Verify curve_type parameter exists with default
    assert hasattr(spec, 'curve_type')
    assert spec.curve_type == "none"

    # Verify arc parameters exist with defaults (0.0 = straight line)
    assert hasattr(spec, 'arc_radius_min')
    assert hasattr(spec, 'arc_radius_max')
    assert spec.arc_radius_min == 0.0
    assert spec.arc_radius_max == 0.0

    assert hasattr(spec, 'arc_concave')
    assert spec.arc_concave is True

    # Verify sine wave parameters exist with defaults (0.0 = straight line)
    assert hasattr(spec, 'sine_amplitude_min')
    assert hasattr(spec, 'sine_amplitude_max')
    assert spec.sine_amplitude_min == 0.0
    assert spec.sine_amplitude_max == 0.0

    assert hasattr(spec, 'sine_frequency_min')
    assert hasattr(spec, 'sine_frequency_max')
    assert spec.sine_frequency_min == 0.0
    assert spec.sine_frequency_max == 0.0

    assert hasattr(spec, 'sine_phase_min')
    assert hasattr(spec, 'sine_phase_max')
    assert spec.sine_phase_min == 0.0
    assert spec.sine_phase_max == 0.0

def test_batch_specification_accepts_curve_parameters():
    """
    Tests that BatchSpecification can be instantiated with explicit curve parameters.
    """
    spec = BatchSpecification(
        name="curved_test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        curve_type="arc",
        arc_radius_min=100.0,
        arc_radius_max=300.0,
        arc_concave=False,
        sine_amplitude_min=5.0,
        sine_amplitude_max=15.0,
        sine_frequency_min=0.01,
        sine_frequency_max=0.05,
        sine_phase_min=0.0,
        sine_phase_max=3.14
    )

    assert spec.curve_type == "arc"
    assert spec.arc_radius_min == 100.0
    assert spec.arc_radius_max == 300.0
    assert spec.arc_concave is False
    assert spec.sine_amplitude_min == 5.0
    assert spec.sine_amplitude_max == 15.0
    assert spec.sine_frequency_min == 0.01
    assert spec.sine_frequency_max == 0.05
    assert spec.sine_phase_min == 0.0
    assert spec.sine_phase_max == 3.14

def test_load_batch_config_with_curve_parameters(tmp_path: Path):
    """
    Tests that curve parameters can be loaded from YAML and parsed correctly.
    """
    yaml_content = """
total_images: 10
specifications:
  - name: "curved_arc_sample"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "arc"
    arc_radius_min: 150.0
    arc_radius_max: 250.0
    arc_concave: false
"""
    yaml_file = tmp_path / "curved_batch.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    batch_config = BatchConfig.from_yaml(str(yaml_file))

    assert batch_config.total_images == 10
    spec = batch_config.specifications[0]
    assert spec.curve_type == "arc"
    assert spec.arc_radius_min == 150.0
    assert spec.arc_radius_max == 250.0
    assert spec.arc_concave is False
    # Sine parameters should still have defaults
    assert spec.sine_amplitude_min == 0.0
    assert spec.sine_amplitude_max == 0.0


def test_batch_specification_has_distribution_fields():
    """
    Tests that BatchSpecification includes distribution fields with correct defaults.
    """
    spec = BatchSpecification(
        name="test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt"
    )

    # Test exponential defaults (degradation effects usually absent)
    assert spec.arc_radius_distribution == "exponential"
    assert spec.sine_amplitude_distribution == "exponential"
    assert spec.glyph_overlap_intensity_distribution == "exponential"
    assert spec.ink_bleed_radius_distribution == "exponential"
    assert spec.perspective_warp_magnitude_distribution == "exponential"
    assert spec.elastic_distortion_alpha_distribution == "exponential"
    assert spec.elastic_distortion_sigma_distribution == "exponential"
    assert spec.grid_distortion_limit_distribution == "exponential"
    assert spec.optical_distortion_limit_distribution == "exponential"
    assert spec.noise_amount_distribution == "exponential"
    assert spec.blur_radius_distribution == "exponential"

    # Test normal defaults (parameters with natural center)
    assert spec.rotation_angle_distribution == "normal"
    assert spec.brightness_factor_distribution == "normal"
    assert spec.contrast_factor_distribution == "normal"

    # Test uniform defaults (discrete or no natural bias)
    assert spec.grid_distortion_steps_distribution == "uniform"
    assert spec.erosion_dilation_kernel_distribution == "uniform"
    assert spec.cutout_width_distribution == "uniform"
    assert spec.cutout_height_distribution == "uniform"
    assert spec.sine_frequency_distribution == "uniform"
    assert spec.sine_phase_distribution == "uniform"


def test_load_batch_config_with_custom_distributions(tmp_path: Path):
    """
    Tests that custom distribution types can be loaded from YAML.
    """
    yaml_content = """
total_images: 50
specifications:
  - name: "custom_distributions"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "arc"
    arc_radius_min: 0.0
    arc_radius_max: 200.0
    arc_radius_distribution: "uniform"
    rotation_angle_min: -15.0
    rotation_angle_max: 15.0
    rotation_angle_distribution: "exponential"
    blur_radius_min: 0.0
    blur_radius_max: 5.0
    blur_radius_distribution: "normal"
    brightness_factor_min: 0.8
    brightness_factor_max: 1.2
    brightness_factor_distribution: "uniform"
"""
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]

    # Verify custom distributions override defaults
    assert spec.arc_radius_distribution == "uniform"
    assert spec.rotation_angle_distribution == "exponential"
    assert spec.blur_radius_distribution == "normal"
    assert spec.brightness_factor_distribution == "uniform"

    # Verify parameters were loaded correctly
    assert spec.arc_radius_min == 0.0
    assert spec.arc_radius_max == 200.0
    assert spec.rotation_angle_min == -15.0
    assert spec.rotation_angle_max == 15.0


# =============================================================================
# Tests for configuration validation
# =============================================================================

def test_invalid_distribution_type_raises_error(tmp_path: Path):
    """Tests that invalid distribution types raise ValueError."""
    yaml_content = """
total_images: 10
specifications:
  - name: "invalid_dist"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    arc_radius_min: 0.0
    arc_radius_max: 100.0
    arc_radius_distribution: "invalid_distribution_type"
"""
    yaml_file = tmp_path / "invalid_dist.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid distribution.*arc_radius_distribution"):
        BatchConfig.from_yaml(str(yaml_file))


def test_multiple_invalid_distributions_reported(tmp_path: Path):
    """Tests that multiple invalid distributions are all reported."""
    yaml_content = """
total_images: 10
specifications:
  - name: "multiple_invalid"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    arc_radius_distribution: "bad_dist1"
    blur_radius_distribution: "bad_dist2"
    rotation_angle_distribution: "bad_dist3"
"""
    yaml_file = tmp_path / "multiple_invalid.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid distribution"):
        config = BatchConfig.from_yaml(str(yaml_file))


def test_valid_distribution_types_accepted(tmp_path: Path):
    """Tests that all valid distribution types are accepted."""
    yaml_content = """
total_images: 10
specifications:
  - name: "all_valid"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    arc_radius_distribution: "uniform"
    blur_radius_distribution: "normal"
    rotation_angle_distribution: "exponential"
"""
    yaml_file = tmp_path / "all_valid.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Should not raise
    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]
    assert spec.arc_radius_distribution == "uniform"
    assert spec.blur_radius_distribution == "normal"
    assert spec.rotation_angle_distribution == "exponential"


def test_invalid_curve_type_raises_error(tmp_path: Path):
    """Tests that invalid curve_type values raise ValueError."""
    yaml_content = """
total_images: 10
specifications:
  - name: "invalid_curve"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "invalid_curve_type"
"""
    yaml_file = tmp_path / "invalid_curve.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid curve_type"):
        BatchConfig.from_yaml(str(yaml_file))


def test_valid_curve_types_accepted(tmp_path: Path):
    """Tests that all valid curve types are accepted."""
    for curve_type in ["none", "arc", "sine"]:
        yaml_content = f"""
total_images: 10
specifications:
  - name: "test_{curve_type}"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "{curve_type}"
"""
        yaml_file = tmp_path / f"curve_{curve_type}.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        # Should not raise
        config = BatchConfig.from_yaml(str(yaml_file))
        assert config.specifications[0].curve_type == curve_type


def test_curve_type_none_with_nonzero_curve_parameters_warning(tmp_path: Path):
    """Tests that curve_type='none' with non-zero curve params raises warning or error."""
    yaml_content = """
total_images: 10
specifications:
  - name: "inconsistent_curve"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "none"
    arc_radius_min: 100.0
    arc_radius_max: 200.0
"""
    yaml_file = tmp_path / "inconsistent_curve.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="curve_type is 'none' but.*arc_radius"):
        BatchConfig.from_yaml(str(yaml_file))


def test_curve_type_none_with_nonzero_sine_parameters_warning(tmp_path: Path):
    """Tests that curve_type='none' with non-zero sine params raises error."""
    yaml_content = """
total_images: 10
specifications:
  - name: "inconsistent_sine"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "none"
    sine_amplitude_min: 5.0
    sine_amplitude_max: 15.0
"""
    yaml_file = tmp_path / "inconsistent_sine.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="curve_type is 'none' but.*sine_amplitude"):
        BatchConfig.from_yaml(str(yaml_file))


def test_curve_type_arc_with_zero_arc_radius_accepted(tmp_path: Path):
    """Tests that curve_type='arc' with zero arc_radius is allowed (edge case)."""
    yaml_content = """
total_images: 10
specifications:
  - name: "arc_zero"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "arc"
    arc_radius_min: 0.0
    arc_radius_max: 0.0
"""
    yaml_file = tmp_path / "arc_zero.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Should not raise - this is valid (will produce straight text)
    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]
    assert spec.curve_type == "arc"
    assert spec.arc_radius_min == 0.0


def test_curve_type_sine_with_zero_amplitude_accepted(tmp_path: Path):
    """Tests that curve_type='sine' with zero amplitude is allowed (edge case)."""
    yaml_content = """
total_images: 10
specifications:
  - name: "sine_zero"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    curve_type: "sine"
    sine_amplitude_min: 0.0
    sine_amplitude_max: 0.0
"""
    yaml_file = tmp_path / "sine_zero.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Should not raise - this is valid (will produce straight text)
    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]
    assert spec.curve_type == "sine"
    assert spec.sine_amplitude_min == 0.0


def test_invalid_text_direction_raises_error(tmp_path: Path):
    """Tests that invalid text_direction values raise ValueError."""
    yaml_content = """
total_images: 10
specifications:
  - name: "invalid_direction"
    proportion: 1.0
    text_direction: "diagonal"
    corpus_file: "test.txt"
"""
    yaml_file = tmp_path / "invalid_direction.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid text_direction"):
        BatchConfig.from_yaml(str(yaml_file))


def test_proportions_sum_validation(tmp_path: Path):
    """Tests that proportions summing to != 1.0 raises error or warning."""
    yaml_content = """
total_images: 100
specifications:
  - name: "spec1"
    proportion: 0.3
    text_direction: "left_to_right"
    corpus_file: "test1.txt"
  - name: "spec2"
    proportion: 0.3
    text_direction: "left_to_right"
    corpus_file: "test2.txt"
"""
    yaml_file = tmp_path / "invalid_proportions.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Proportions.*do not sum to 1.0"):
        BatchConfig.from_yaml(str(yaml_file))


def test_proportions_sum_to_one_accepted(tmp_path: Path):
    """Tests that proportions summing to 1.0 are accepted."""
    yaml_content = """
total_images: 100
specifications:
  - name: "spec1"
    proportion: 0.7
    text_direction: "left_to_right"
    corpus_file: "test1.txt"
  - name: "spec2"
    proportion: 0.3
    text_direction: "left_to_right"
    corpus_file: "test2.txt"
"""
    yaml_file = tmp_path / "valid_proportions.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    # Should not raise
    config = BatchConfig.from_yaml(str(yaml_file))
    assert len(config.specifications) == 2


# =============================================================================
# Tests for color parameters
# =============================================================================

def test_batch_specification_has_default_color_parameters():
    """Tests that BatchSpecification has color parameters with correct defaults."""
    spec = BatchSpecification(
        name="test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt"
    )

    # Default should be uniform mode with black text
    assert hasattr(spec, 'color_mode')
    assert spec.color_mode == "uniform"

    # Uniform mode color range (default black)
    assert hasattr(spec, 'text_color_min')
    assert hasattr(spec, 'text_color_max')
    assert spec.text_color_min == (0, 0, 0)
    assert spec.text_color_max == (0, 0, 0)

    # Per-glyph mode parameters
    assert hasattr(spec, 'per_glyph_palette_size_min')
    assert hasattr(spec, 'per_glyph_palette_size_max')
    assert spec.per_glyph_palette_size_min == 2
    assert spec.per_glyph_palette_size_max == 5

    # Gradient mode parameters
    assert hasattr(spec, 'gradient_start_color_min')
    assert hasattr(spec, 'gradient_start_color_max')
    assert hasattr(spec, 'gradient_end_color_min')
    assert hasattr(spec, 'gradient_end_color_max')
    assert spec.gradient_start_color_min == (0, 0, 0)
    assert spec.gradient_start_color_max == (0, 0, 0)
    assert spec.gradient_end_color_min == (0, 0, 0)
    assert spec.gradient_end_color_max == (0, 0, 0)


def test_batch_specification_accepts_uniform_color_parameters():
    """Tests that uniform color mode parameters can be set."""
    spec = BatchSpecification(
        name="colored_uniform",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        color_mode="uniform",
        text_color_min=(100, 50, 0),
        text_color_max=(200, 150, 100)
    )

    assert spec.color_mode == "uniform"
    assert spec.text_color_min == (100, 50, 0)
    assert spec.text_color_max == (200, 150, 100)


def test_batch_specification_accepts_per_glyph_color_parameters():
    """Tests that per_glyph color mode parameters can be set."""
    spec = BatchSpecification(
        name="colored_per_glyph",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        color_mode="per_glyph",
        text_color_min=(0, 0, 0),
        text_color_max=(255, 255, 255),
        per_glyph_palette_size_min=3,
        per_glyph_palette_size_max=10
    )

    assert spec.color_mode == "per_glyph"
    assert spec.per_glyph_palette_size_min == 3
    assert spec.per_glyph_palette_size_max == 10


def test_batch_specification_accepts_gradient_color_parameters():
    """Tests that gradient color mode parameters can be set."""
    spec = BatchSpecification(
        name="colored_gradient",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        color_mode="gradient",
        gradient_start_color_min=(0, 0, 0),
        gradient_start_color_max=(50, 50, 50),
        gradient_end_color_min=(200, 200, 200),
        gradient_end_color_max=(255, 255, 255)
    )

    assert spec.color_mode == "gradient"
    assert spec.gradient_start_color_min == (0, 0, 0)
    assert spec.gradient_start_color_max == (50, 50, 50)
    assert spec.gradient_end_color_min == (200, 200, 200)
    assert spec.gradient_end_color_max == (255, 255, 255)


def test_load_batch_config_with_uniform_color(tmp_path: Path):
    """Tests that uniform color parameters can be loaded from YAML."""
    yaml_content = """
total_images: 10
specifications:
  - name: "uniform_colored"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    color_mode: "uniform"
    text_color_min: [50, 100, 150]
    text_color_max: [100, 150, 200]
"""
    yaml_file = tmp_path / "uniform_color.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]

    assert spec.color_mode == "uniform"
    assert spec.text_color_min == (50, 100, 150)
    assert spec.text_color_max == (100, 150, 200)


def test_load_batch_config_with_per_glyph_color(tmp_path: Path):
    """Tests that per_glyph color parameters can be loaded from YAML."""
    yaml_content = """
total_images: 10
specifications:
  - name: "per_glyph_colored"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    color_mode: "per_glyph"
    text_color_min: [0, 0, 0]
    text_color_max: [255, 255, 255]
    per_glyph_palette_size_min: 5
    per_glyph_palette_size_max: 8
"""
    yaml_file = tmp_path / "per_glyph_color.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]

    assert spec.color_mode == "per_glyph"
    assert spec.per_glyph_palette_size_min == 5
    assert spec.per_glyph_palette_size_max == 8


def test_load_batch_config_with_gradient_color(tmp_path: Path):
    """Tests that gradient color parameters can be loaded from YAML."""
    yaml_content = """
total_images: 10
specifications:
  - name: "gradient_colored"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    color_mode: "gradient"
    gradient_start_color_min: [255, 0, 0]
    gradient_start_color_max: [255, 50, 50]
    gradient_end_color_min: [0, 0, 255]
    gradient_end_color_max: [50, 50, 255]
"""
    yaml_file = tmp_path / "gradient_color.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]

    assert spec.color_mode == "gradient"
    assert spec.gradient_start_color_min == (255, 0, 0)
    assert spec.gradient_start_color_max == (255, 50, 50)
    assert spec.gradient_end_color_min == (0, 0, 255)
    assert spec.gradient_end_color_max == (50, 50, 255)


def test_invalid_color_mode_raises_error(tmp_path: Path):
    """Tests that invalid color_mode values raise ValueError."""
    yaml_content = """
total_images: 10
specifications:
  - name: "invalid_color_mode"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    color_mode: "rainbow"
"""
    yaml_file = tmp_path / "invalid_color_mode.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid color_mode"):
        BatchConfig.from_yaml(str(yaml_file))


def test_valid_color_modes_accepted(tmp_path: Path):
    """Tests that all valid color modes are accepted."""
    for color_mode in ["uniform", "per_glyph", "gradient"]:
        yaml_content = f"""
total_images: 10
specifications:
  - name: "test_{color_mode}"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    color_mode: "{color_mode}"
"""
        yaml_file = tmp_path / f"color_{color_mode}.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        # Should not raise
        config = BatchConfig.from_yaml(str(yaml_file))
        assert config.specifications[0].color_mode == color_mode


# =============================================================================
# Tests for font size parameters
# =============================================================================

def test_batch_specification_has_default_font_size_parameters():
    """Tests that BatchSpecification has font_size parameters with correct defaults."""
    spec = BatchSpecification(
        name="test",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt"
    )

    # Default should be 32 (current hardcoded value)
    assert hasattr(spec, 'font_size_min')
    assert hasattr(spec, 'font_size_max')
    assert spec.font_size_min == 32
    assert spec.font_size_max == 32


def test_batch_specification_accepts_font_size_parameters():
    """Tests that font_size parameters can be set."""
    spec = BatchSpecification(
        name="variable_font_size",
        proportion=1.0,
        text_direction="left_to_right",
        corpus_file="test.txt",
        font_size_min=24,
        font_size_max=72
    )

    assert spec.font_size_min == 24
    assert spec.font_size_max == 72


def test_load_batch_config_with_font_size(tmp_path: Path):
    """Tests that font_size parameters can be loaded from YAML."""
    yaml_content = """
total_images: 10
specifications:
  - name: "variable_fonts"
    proportion: 1.0
    text_direction: "left_to_right"
    corpus_file: "test.txt"
    font_size_min: 18
    font_size_max: 96
"""
    yaml_file = tmp_path / "font_size.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    config = BatchConfig.from_yaml(str(yaml_file))
    spec = config.specifications[0]

    assert spec.font_size_min == 18
    assert spec.font_size_max == 96
