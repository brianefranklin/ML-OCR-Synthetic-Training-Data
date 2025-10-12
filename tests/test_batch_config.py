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
