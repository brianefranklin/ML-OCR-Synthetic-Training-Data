

import pytest
import subprocess
from pathlib import Path
import yaml
import shutil


@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for batch generation tests."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    # Create a dummy corpus file
    (text_dir / "corpus.txt").write_text("hello world test sample text")

    # Copy a font
    shutil.copy("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fonts_dir)

    return {
        "text_file": str(text_dir / "corpus.txt"),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "tmp_path": tmp_path
    }


def test_batch_count_within_10_percent_small(test_environment):
    """Test that batch generation produces expected count within 10% for small batch."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    # Small batch: 10 images
    expected_count = 10
    batch_config_data = {
        "total_images": expected_count,
        "batches": [
            {
                "name": "test_batch",
                "proportion": 1.0,
                "text_direction": "left_to_right",
                "corpus_file": test_environment["text_file"]
            }
        ]
    }

    batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
    with open(batch_config_path, "w") as f:
        yaml.dump(batch_config_data, f)

    command = [
        "python3",
        str(script_path),
        "--batch-config", str(batch_config_path),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    actual_count = len(image_files)

    # Check within 10% tolerance
    tolerance = 0.10
    min_expected = int(expected_count * (1 - tolerance))
    max_expected = int(expected_count * (1 + tolerance))

    assert min_expected <= actual_count <= max_expected, \
        f"Expected {expected_count} images (±10%: {min_expected}-{max_expected}), got {actual_count}"


def test_batch_count_within_10_percent_medium(test_environment):
    """Test that batch generation produces expected count within 10% for medium batch."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    # Medium batch: 50 images
    expected_count = 50
    batch_config_data = {

        "total_images": expected_count,

        "batches": [

            {

                "name": "test_batch",

                "proportion": 1.0,

                "text_direction": "left_to_right",

                "corpus_file": test_environment["text_file"]

            }

        ]

    }
    batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
    with open(batch_config_path, "w") as f:
        yaml.dump(batch_config_data, f)

    command = [
        "python3",
        str(script_path),
        "--batch-config", str(batch_config_path),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    actual_count = len(image_files)

    # Check within 10% tolerance
    tolerance = 0.10
    min_expected = int(expected_count * (1 - tolerance))
    max_expected = int(expected_count * (1 + tolerance))

    assert min_expected <= actual_count <= max_expected, \
        f"Expected {expected_count} images (±10%: {min_expected}-{max_expected}), got {actual_count}"


def test_batch_count_within_10_percent_large(test_environment):
    """Test that batch generation produces expected count within 10% for large batch."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    # Large batch: 100 images
    expected_count = 100
    batch_config_data = {
        "total_images": expected_count,
        "batches": [
            {
                "name": "test_batch",
                "proportion": 1.0,
                "text_direction": "left_to_right",
                "corpus_file": test_environment["text_file"]
            }
        ]
    }

    batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
    with open(batch_config_path, "w") as f:
        yaml.dump(batch_config_data, f)

    command = [
        "python3",
        str(script_path),
        "--batch-config", str(batch_config_path),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    actual_count = len(image_files)

    # Check within 10% tolerance
    tolerance = 0.10
    min_expected = int(expected_count * (1 - tolerance))
    max_expected = int(expected_count * (1 + tolerance))

    assert min_expected <= actual_count <= max_expected, \
        f"Expected {expected_count} images (±10%: {min_expected}-{max_expected}), got {actual_count}"


def test_batch_count_multiple_batches(test_environment):
    """Test that multiple batches produce expected total count within 10%."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    # Multiple batches totaling 30 images
    expected_count = 30
    batch_config_data = {
        "total_images": expected_count,
        "batches": [
            {
                "name": "batch_1",
                "proportion": 0.5,
                "text_direction": "left_to_right",
                "corpus_file": test_environment["text_file"]
            },
            {
                "name": "batch_2",
                "proportion": 0.3,
                "text_direction": "left_to_right",
                "corpus_file": test_environment["text_file"]
            },
            {
                "name": "batch_3",
                "proportion": 0.2,
                "text_direction": "left_to_right",
                "corpus_file": test_environment["text_file"]
            }
        ]
    }

    batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
    with open(batch_config_path, "w") as f:
        yaml.dump(batch_config_data, f)

    command = [
        "python3",
        str(script_path),
        "--batch-config", str(batch_config_path),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    actual_count = len(image_files)

    # Check within 10% tolerance
    tolerance = 0.10
    min_expected = int(expected_count * (1 - tolerance))
    max_expected = int(expected_count * (1 + tolerance))

    assert min_expected <= actual_count <= max_expected, \
        f"Expected {expected_count} images (±10%: {min_expected}-{max_expected}), got {actual_count}"


def test_batch_count_with_effects(test_environment):
    """Test that batch with various effects produces expected count within 10%."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    # Batch with effects: 20 images
    expected_count = 20
    batch_config_data = {
        "total_images": expected_count,
        "batches": [
            {
                "name": "effects_batch",
                "proportion": 1.0,
                "text_direction": "left_to_right",
                "curve_type": "arc",
                "curve_intensity": 0.3,
                "overlap_intensity": 0.2,
                "effect_type": "embossed",
                "effect_depth": 0.5,
                "text_color_mode": "per_glyph",
                "color_palette": "vibrant",
                "corpus_file": test_environment["text_file"]
            }
        ]
    }

    batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
    with open(batch_config_path, "w") as f:
        yaml.dump(batch_config_data, f)

    command = [
        "python3",
        str(script_path),
        "--batch-config", str(batch_config_path),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    actual_count = len(image_files)

    # Check within 10% tolerance
    tolerance = 0.10
    min_expected = int(expected_count * (1 - tolerance))
    max_expected = int(expected_count * (1 + tolerance))

    assert min_expected <= actual_count <= max_expected, \
        f"Expected {expected_count} images (±10%: {min_expected}-{max_expected}), got {actual_count}"
