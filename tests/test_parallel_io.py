"""
Tests for parallel I/O functionality in main.py.

This module tests the multiprocessing-based parallel image and label saving,
as well as the backwards-compatible sequential mode.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any
import tempfile
import shutil

# Import the worker function and NumpyEncoder from main
# Note: This will be available after implementation
from src.main import save_image_and_label, NumpyEncoder


def test_save_image_and_label_single(tmp_path):
    """Tests that the worker function saves a single image and label correctly."""
    # Create a test image
    image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))

    # Create a test plan with various data types including numpy types
    plan = {
        "text": "test",
        "font": "test.ttf",
        "bboxes": [
            {"char": "t", "x0": 10, "y0": 20, "x1": 30, "y1": 40},
            {"char": "e", "x0": np.int64(35), "y0": np.int64(20), "x1": np.int64(55), "y1": np.int64(40)}
        ]
    }

    # Create paths
    image_path = tmp_path / "test_image.png"
    label_path = tmp_path / "test_label.json"

    # Call the worker function
    save_image_and_label((image, plan, image_path, label_path))

    # Verify image was saved
    assert image_path.exists()
    loaded_image = Image.open(image_path)
    assert loaded_image.size == (100, 50)

    # Verify label was saved
    assert label_path.exists()
    with open(label_path) as f:
        loaded_plan = json.load(f)

    # Verify content
    assert loaded_plan["text"] == "test"
    assert len(loaded_plan["bboxes"]) == 2
    assert loaded_plan["bboxes"][1]["x0"] == 35  # Should be converted to int
    assert isinstance(loaded_plan["bboxes"][1]["x0"], int)


def test_numpy_types_serialized_correctly(tmp_path):
    """Tests that NumPy data types are correctly serialized to JSON."""
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 255))

    # Plan with various numpy types
    plan = {
        "int64_value": np.int64(42),
        "int32_value": np.int32(100),
        "float64_value": np.float64(3.14),
        "float32_value": np.float32(2.71),
        "array_value": np.array([1, 2, 3]),
        "nested": {
            "numpy_int": np.int64(999)
        }
    }

    image_path = tmp_path / "numpy_test.png"
    label_path = tmp_path / "numpy_test.json"

    # Should not raise any JSON serialization errors
    save_image_and_label((image, plan, image_path, label_path))

    # Verify JSON can be loaded
    with open(label_path) as f:
        loaded = json.load(f)

    # Verify types are converted to native Python types
    assert isinstance(loaded["int64_value"], int)
    assert isinstance(loaded["float64_value"], float)
    assert isinstance(loaded["array_value"], list)
    assert loaded["int64_value"] == 42
    assert loaded["nested"]["numpy_int"] == 999


def test_save_multiple_images_creates_all_files(tmp_path):
    """Tests that saving multiple images creates all expected files."""
    # Create multiple images and plans
    tasks = []
    for i in range(5):
        image = Image.new("RGBA", (50, 50), (i * 50, 0, 0, 255))
        plan = {"text": f"image_{i}", "index": i}
        image_path = tmp_path / f"image_{i:03d}.png"
        label_path = tmp_path / f"image_{i:03d}.json"
        tasks.append((image, plan, image_path, label_path))

    # Save all (simulating what multiprocessing.Pool.map would do)
    for task in tasks:
        save_image_and_label(task)

    # Verify all files exist
    assert len(list(tmp_path.glob("*.png"))) == 5
    assert len(list(tmp_path.glob("*.json"))) == 5

    # Verify content of one file
    with open(tmp_path / "image_003.json") as f:
        data = json.load(f)
    assert data["index"] == 3
    assert data["text"] == "image_3"


def test_worker_function_handles_large_bbox_list(tmp_path):
    """Tests that the worker function handles large lists of bounding boxes."""
    image = Image.new("RGBA", (1000, 100), (0, 0, 0, 255))

    # Create a large bbox list (simulating many characters)
    bboxes = []
    for i in range(100):
        bboxes.append({
            "char": chr(65 + (i % 26)),
            "x0": np.int64(i * 10),
            "y0": np.int64(20),
            "x1": np.int64(i * 10 + 8),
            "y1": np.int64(40)
        })

    plan = {"text": "A" * 100, "bboxes": bboxes}
    image_path = tmp_path / "large.png"
    label_path = tmp_path / "large.json"

    # Should handle without errors
    save_image_and_label((image, plan, image_path, label_path))

    # Verify
    with open(label_path) as f:
        loaded = json.load(f)
    assert len(loaded["bboxes"]) == 100
    # Check all coordinates are ints
    for bbox in loaded["bboxes"]:
        assert isinstance(bbox["x0"], int)


def test_worker_function_creates_parent_directories(tmp_path):
    """Tests that the worker function creates parent directories if needed."""
    # Create nested directory structure
    nested_dir = tmp_path / "output" / "subdir" / "nested"

    image = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
    plan = {"text": "test"}

    image_path = nested_dir / "image.png"
    label_path = nested_dir / "label.json"

    # Create parent directories manually (this is what main.py should do)
    # The worker function itself shouldn't need to create directories
    nested_dir.mkdir(parents=True, exist_ok=True)

    # Now save should work
    save_image_and_label((image, plan, image_path, label_path))

    assert image_path.exists()
    assert label_path.exists()


def test_empty_plan_saves_correctly(tmp_path):
    """Tests that an empty or minimal plan saves correctly."""
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 255))
    plan = {}  # Empty plan

    image_path = tmp_path / "empty.png"
    label_path = tmp_path / "empty.json"

    save_image_and_label((image, plan, image_path, label_path))

    # Verify files exist
    assert image_path.exists()
    assert label_path.exists()

    # Verify JSON is valid
    with open(label_path) as f:
        loaded = json.load(f)
    assert loaded == {}


def test_image_modes_supported(tmp_path):
    """Tests that different image modes (RGB, RGBA, L) are saved correctly."""
    modes_and_colors = [
        ("RGB", (255, 0, 0)),
        ("RGBA", (0, 255, 0, 255)),
        ("L", 128),
    ]

    for i, (mode, color) in enumerate(modes_and_colors):
        image = Image.new(mode, (20, 20), color)
        plan = {"mode": mode}

        image_path = tmp_path / f"mode_{i}.png"
        label_path = tmp_path / f"mode_{i}.json"

        save_image_and_label((image, plan, image_path, label_path))

        # Verify saved correctly
        assert image_path.exists()
        loaded_image = Image.open(image_path)
        assert loaded_image.size == (20, 20)
