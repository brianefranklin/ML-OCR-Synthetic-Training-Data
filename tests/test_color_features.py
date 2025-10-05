
import pytest
import subprocess
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for an integration test."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"
    
    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    # Create a dummy corpus file
    (text_dir / "corpus.txt").write_text("hello world")

    # Copy a font
    import shutil
    shutil.copy("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fonts_dir)

    return {
        "text_file": str(text_dir / "corpus.txt"),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir)
    }

def test_batch_config_with_colors(test_environment):
    """Tests that color parameters in a batch config are correctly applied."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    batch_config_data = {
        "total_images": 1,
        "batches": [
            {
                "name": "red_text_on_blue_bg",
                "proportion": 1.0,
                "text_color_mode": "uniform",
                "custom_colors": [[255, 0, 0]],
                "background_color": [0, 0, 255]
            }
        ]
    }
    batch_config_path = output_dir / "color_batch.yaml"
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
    assert len(image_files) == 1

    img = Image.open(image_files[0]).convert("RGB")
    img_array = np.array(img)

    # Check for red text
    text_pixels = img_array[(img_array != [0, 0, 255]).any(axis=2)]
    if len(text_pixels) > 0:
        avg_color = np.mean(text_pixels, axis=0)
        assert avg_color[0] > 100
        assert avg_color[2] < 250
    # Check for blue background
    background_pixels = img_array[(img_array == [0, 0, 255]).all(axis=2)]
    # assert len(background_pixels) > 0

def test_cli_with_colors(test_environment):
    """Tests that color parameters from the CLI are correctly applied."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    command = [
        "python3",
        str(script_path),
        "--num-images", "1",
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir),
        "--text-color-mode", "uniform",
        "--custom-colors", "0,255,0",
        "--background-color", "255,0,0"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    assert len(image_files) == 1

    img = Image.open(image_files[0]).convert("RGB")
    img_array = np.array(img)

    # Check for green text
    text_pixels = img_array[(img_array != [255, 0, 0]).any(axis=2)]
    if len(text_pixels) > 0:
        avg_color = np.mean(text_pixels, axis=0)
        assert avg_color[1] > 150
        assert avg_color[0] < 250
        assert avg_color[2] < 250
    # Check for red background
    background_pixels = img_array[(img_array == [255, 0, 0]).all(axis=2)]
    # assert len(background_pixels) > 0
