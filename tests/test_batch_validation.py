"""
Tests for batch configuration validation.

This module tests the batch validation system that checks configurations
before generation starts, ensuring all required resources exist and
configuration values are valid.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from src.batch_validation import BatchValidator, ValidationError


def create_test_config(
    total_images: int = 10,
    specs: list = None
) -> Dict[str, Any]:
    """Create a basic test configuration."""
    if specs is None:
        specs = [{
            "name": "test_spec",
            "count": 10,
            "corpus_files": ["test.txt"],
            "font_filter": {"extensions": [".ttf"]},
            "config": {
                "direction": {"ltr": 1.0},
                "canvas": {
                    "min_padding": 10,
                    "max_padding": 50,
                    "placement": "uniform_random"
                },
                "font": {
                    "size_range": [20, 60]
                },
                "effects": {}
            }
        }]

    return {
        "total_images": total_images,
        "specifications": specs
    }


def test_valid_config_passes(tmp_path):
    """Test that a valid configuration passes validation."""
    # Create test files
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("Hello world\nTest content\n")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font data")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    (bg_dir / "bg.png").write_text("fake image")

    # Create config
    config = create_test_config()

    # Validate
    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    # Should not raise
    validator.validate()


def test_missing_corpus_file_fails(tmp_path):
    """Test that missing corpus files trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Don't create test.txt

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = create_test_config()

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "corpus" in str(exc_info.value).lower()
    assert "test.txt" in str(exc_info.value)


def test_missing_font_directory_fails(tmp_path):
    """Test that missing font directory triggers validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    # Don't create font directory

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = create_test_config()

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "font" in str(exc_info.value).lower()


def test_no_fonts_in_directory_fails(tmp_path):
    """Test that font directory with no fonts triggers validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    # Directory exists but has no .ttf files

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = create_test_config()

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "font" in str(exc_info.value).lower()
    assert "no" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()


def test_missing_background_directory_fails(tmp_path):
    """Test that missing background directory triggers validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    # Don't create background directory

    config = create_test_config()

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "background" in str(exc_info.value).lower()


def test_invalid_padding_range_fails(tmp_path):
    """Test that invalid padding ranges trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Invalid config: min > max
    config = create_test_config()
    config["specifications"][0]["config"]["canvas"]["min_padding"] = 100
    config["specifications"][0]["config"]["canvas"]["max_padding"] = 10

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "padding" in str(exc_info.value).lower()


def test_negative_padding_fails(tmp_path):
    """Test that negative padding values trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Invalid config: negative padding
    config = create_test_config()
    config["specifications"][0]["config"]["canvas"]["min_padding"] = -10

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "padding" in str(exc_info.value).lower()
    assert "negative" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()


def test_invalid_font_size_range_fails(tmp_path):
    """Test that invalid font size ranges trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Invalid config: min > max
    config = create_test_config()
    config["specifications"][0]["config"]["font"]["size_range"] = [100, 10]

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "font" in str(exc_info.value).lower() or "size" in str(exc_info.value).lower()


def test_missing_required_field_fails(tmp_path):
    """Test that missing required fields trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Missing total_images
    config = create_test_config()
    del config["total_images"]

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "total_images" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


def test_empty_specifications_fails(tmp_path):
    """Test that empty specifications list triggers validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = {
        "total_images": 10,
        "specifications": []
    }

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "specification" in str(exc_info.value).lower()
    assert "empty" in str(exc_info.value).lower() or "no" in str(exc_info.value).lower()


def test_invalid_direction_weights_fails(tmp_path):
    """Test that invalid direction weights trigger validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    # Invalid config: negative weights
    config = create_test_config()
    config["specifications"][0]["config"]["direction"] = {"ltr": -0.5, "rtl": 1.5}

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "direction" in str(exc_info.value).lower() or "weight" in str(exc_info.value).lower()


def test_zero_total_images_fails(tmp_path):
    """Test that zero total images triggers validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = create_test_config(total_images=0)

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "total_images" in str(exc_info.value).lower()


def test_validation_error_message_is_clear():
    """Test that ValidationError provides clear error messages."""
    error = ValidationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)

def test_invalid_min_max_pairs_fails(tmp_path):
    """Test that invalid min/max pairs trigger a validation error."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("content")

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "test.ttf").write_text("fake font")

    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()

    config = {
        "total_images": 1,
        "specifications": [
            {
                "name": "test_spec_invalid_min_max",
                "proportion": 1.0,
                "corpus_files": ["test.txt"],
                "min_text_length": 100,
                "max_text_length": 10,
            }
        ]
    }

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()

    assert "min_text_length" in str(exc_info.value)
    assert "cannot be greater than" in str(exc_info.value)
    assert "max_text_length" in str(exc_info.value)

    # Test another pair
    config["specifications"][0]["min_text_length"] = 10
    config["specifications"][0]["max_text_length"] = 100
    config["specifications"][0]["font_size_min"] = 50
    config["specifications"][0]["font_size_max"] = 20

    validator = BatchValidator(
        config=config,
        corpus_dir=str(corpus_dir),
        font_dir=str(font_dir),
        background_dir=str(bg_dir)
    )

    with pytest.raises(ValidationError) as exc_info:
        validator.validate()
    
    assert "font_size_min" in str(exc_info.value)
    assert "cannot be greater than" in str(exc_info.value)
    assert "font_size_max" in str(exc_info.value)