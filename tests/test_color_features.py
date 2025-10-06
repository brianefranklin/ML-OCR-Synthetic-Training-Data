
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
    """Tests that color parameters in a batch config are correctly applied with RGBA transparency."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"
    output_dir = Path(test_environment["output_dir"])

    batch_config_data = {
        "total_images": 1,
        "batches": [
            {
                "name": "red_text_transparent_bg",
                "proportion": 1.0,
                "text_color_mode": "uniform",
                "custom_colors": [[255, 0, 0]],
                # background_color is deprecated - RGBA pipeline creates transparent backgrounds
                "corpus_file": test_environment["text_file"]
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

    # Image should be RGBA mode (transparent background)
    img = Image.open(image_files[0])
    assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"

    img_array = np.array(img)

    # Check for red text (RGB channels)
    # Get non-transparent pixels (alpha > 0)
    alpha = img_array[:, :, 3]
    text_mask = alpha > 0

    if text_mask.sum() > 0:
        text_pixels = img_array[text_mask][:, :3]  # RGB only
        avg_color = np.mean(text_pixels, axis=0)
        assert avg_color[0] > 100, f"Red channel too low: {avg_color[0]}"  # Red channel should be high

    # Check for transparent background (alpha = 0)
    transparent_pixels = np.sum(alpha == 0)
    assert transparent_pixels > 0, "Expected some transparent background pixels"

def test_cli_with_colors(test_environment):
    """Tests that color parameters from the CLI are correctly applied with RGBA transparency."""
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
        "--custom-colors", "0,255,0"
        # Note: --background-color is deprecated in RGBA pipeline
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    image_files = list(output_dir.glob("image_*.png"))
    assert len(image_files) == 1

    # Image should be RGBA mode
    img = Image.open(image_files[0])
    assert img.mode == 'RGBA', f"Expected RGBA mode, got {img.mode}"

    img_array = np.array(img)

    # Check for green text (RGB channels)
    # Get non-transparent pixels (alpha > 0)
    alpha = img_array[:, :, 3]
    text_mask = alpha > 0

    if text_mask.sum() > 0:
        text_pixels = img_array[text_mask][:, :3]  # RGB only
        avg_color = np.mean(text_pixels, axis=0)
        assert avg_color[1] > 100, f"Green channel too low: {avg_color[1]}"  # Green channel should be high

    # Check for transparent background (alpha = 0)
    transparent_pixels = np.sum(alpha == 0)
    assert transparent_pixels > 0, "Expected some transparent background pixels"
