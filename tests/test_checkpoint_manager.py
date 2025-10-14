"""
Tests for checkpoint manager for resume capability.

This module tests the checkpoint manager that tracks generation progress,
enabling resume functionality after interruptions.
"""

import pytest
import json
import hashlib
from pathlib import Path
from typing import Dict, Any

from src.checkpoint_manager import CheckpointManager, CheckpointWarning


def create_test_config() -> Dict[str, Any]:
    """Create a basic test configuration."""
    return {
        "total_images": 100,
        "specifications": [{
            "name": "test_spec",
            "proportion": 1.0,
            "corpus_file": "test.txt"
        }]
    }


def test_create_new_checkpoint(tmp_path):
    """Test creating a new checkpoint file."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    manager.save_checkpoint(completed_images=0)

    # Check checkpoint file exists
    checkpoint_file = output_dir / ".generation_checkpoint.json"
    assert checkpoint_file.exists()

    # Check contents
    with open(checkpoint_file) as f:
        data = json.load(f)

    assert data["completed_images"] == 0
    assert "config_hash" in data
    assert data["total_images"] == 100


def test_update_existing_checkpoint(tmp_path):
    """Test updating an existing checkpoint."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    # Save initial checkpoint
    manager.save_checkpoint(completed_images=0)

    # Update checkpoint
    manager.save_checkpoint(completed_images=50)

    # Load and verify
    checkpoint_file = output_dir / ".generation_checkpoint.json"
    with open(checkpoint_file) as f:
        data = json.load(f)

    assert data["completed_images"] == 50


def test_load_checkpoint(tmp_path):
    """Test loading an existing checkpoint."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    # Create checkpoint
    manager1 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )
    manager1.save_checkpoint(completed_images=75)

    # Load checkpoint with new manager
    manager2 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    data = manager2.load_checkpoint()

    assert data is not None
    assert data["completed_images"] == 75
    assert data["total_images"] == 100


def test_checkpoint_does_not_exist(tmp_path):
    """Test loading when no checkpoint exists."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    data = manager.load_checkpoint()

    assert data is None


def test_config_hash_changes_warning(tmp_path):
    """Test that config changes trigger a warning."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config1 = create_test_config()

    # Create checkpoint with config1
    manager1 = CheckpointManager(
        output_dir=str(output_dir),
        config=config1
    )
    manager1.save_checkpoint(completed_images=50)

    # Load with different config
    config2 = create_test_config()
    config2["total_images"] = 200  # Change config

    manager2 = CheckpointManager(
        output_dir=str(output_dir),
        config=config2
    )

    with pytest.warns(CheckpointWarning) as warning_info:
        data = manager2.load_checkpoint()

    assert len(warning_info) > 0
    assert "config" in str(warning_info[0].message).lower()
    assert data["completed_images"] == 50  # Still loads the checkpoint


def test_same_config_no_warning(tmp_path):
    """Test that same config does not trigger warning."""
    import warnings as warn_module

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    # Create checkpoint
    manager1 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )
    manager1.save_checkpoint(completed_images=50)

    # Load with same config
    manager2 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    # Should not raise warning
    with warn_module.catch_warnings(record=True) as warning_list:
        warn_module.simplefilter("always")
        data = manager2.load_checkpoint()

    # Filter out unrelated warnings
    checkpoint_warnings = [w for w in warning_list if issubclass(w.category, CheckpointWarning)]
    assert len(checkpoint_warnings) == 0


def test_get_completed_indices(tmp_path):
    """Test retrieving list of completed image indices."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create some output files
    (output_dir / "image_00000.png").write_text("fake")
    (output_dir / "image_00001.png").write_text("fake")
    (output_dir / "image_00005.png").write_text("fake")
    (output_dir / "image_00010.png").write_text("fake")

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    completed = manager.get_completed_indices()

    assert completed == {0, 1, 5, 10}


def test_get_completed_indices_empty_directory(tmp_path):
    """Test retrieving completed indices from empty directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    completed = manager.get_completed_indices()

    assert completed == set()


def test_checkpoint_stores_timestamp(tmp_path):
    """Test that checkpoint stores timestamp."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    manager.save_checkpoint(completed_images=10)

    checkpoint_file = output_dir / ".generation_checkpoint.json"
    with open(checkpoint_file) as f:
        data = json.load(f)

    assert "timestamp" in data
    assert isinstance(data["timestamp"], str)


def test_compute_config_hash_deterministic(tmp_path):
    """Test that config hash is deterministic."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config = create_test_config()

    manager1 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    manager2 = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    hash1 = manager1._compute_config_hash()
    hash2 = manager2._compute_config_hash()

    assert hash1 == hash2


def test_compute_config_hash_changes_with_config(tmp_path):
    """Test that config hash changes when config changes."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config1 = create_test_config()
    config2 = create_test_config()
    config2["total_images"] = 200

    manager1 = CheckpointManager(
        output_dir=str(output_dir),
        config=config1
    )

    manager2 = CheckpointManager(
        output_dir=str(output_dir),
        config=config2
    )

    hash1 = manager1._compute_config_hash()
    hash2 = manager2._compute_config_hash()

    assert hash1 != hash2


def test_should_skip_index(tmp_path):
    """Test checking if an index should be skipped."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create some output files
    (output_dir / "image_00000.png").write_text("fake")
    (output_dir / "image_00001.png").write_text("fake")

    config = create_test_config()

    manager = CheckpointManager(
        output_dir=str(output_dir),
        config=config
    )

    assert manager.should_skip_index(0) is True
    assert manager.should_skip_index(1) is True
    assert manager.should_skip_index(2) is False
    assert manager.should_skip_index(100) is False
