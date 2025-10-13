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
