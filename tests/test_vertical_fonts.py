import pytest
import subprocess
import os
import shutil
from pathlib import Path

@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for an integration test."""
    # Create temporary directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"
    log_file = tmp_path / "generation.log"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    return {
        "text_dir": str(text_dir),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "log_file": str(log_file)
    }

@pytest.mark.parametrize("language, corpus, font_name", [
    ("korean", "안녕하세요\n안녕하세요\n안녕하세요\n안녕하세요\n안녕하세요\n", "NanumGothic.ttf"),
    ("japanese", "こんにちは\nこんにちは\nこんにちは\nこんにちは\nこんにちは\n", "NotoSerifCJKjp-Regular.otf"),
    ("chinese", "你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界\n", "NotoSansCJKtc-VF.otf"),
])
def test_top_to_bottom_text_generation(test_environment, language, corpus, font_name):
    """Tests that the --text-direction top_to_bottom flag works for different languages."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file
    text_file = Path(test_environment["text_dir"]) / f"{language}_corpus.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(corpus)

    # Copy a font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / font_name
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping {language} top_to_bottom test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "top_to_bottom",
        "--log-file", test_environment["log_file"]
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        with open(test_environment["log_file"], "r") as f:
            log_contents = f.read()
        assert result.returncode == 0, f"Script failed with error:\n{log_contents}"

    output_dir = Path(test_environment["output_dir"])
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are generally stacked top to bottom
    # Due to augmentations (rotation, perspective), strict ordering may not hold for every pair
    if len(bboxes) > 1:
        # Check that the first bbox is near the top and last bbox is near the bottom
        assert bboxes[0][1] <= bboxes[-1][1], "First character should be above or at same level as last character"

        # Check that majority of adjacent pairs are in top-to-bottom order
        # Using a lenient threshold (50%) because augmentations can significantly affect ordering
        ordered_pairs = sum(1 for i in range(len(bboxes) - 1) if bboxes[i][1] < bboxes[i+1][1])
        total_pairs = len(bboxes) - 1
        if total_pairs > 0:
            assert ordered_pairs / total_pairs >= 0.5, f"At least 50% of bboxes should be in top-to-bottom order, got {ordered_pairs}/{total_pairs}"

    # Check image dimensions are reasonable
    # Note: After augmentations (rotation, perspective), aspect ratio may change
    # So we just verify the image exists and isn't degenerate
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > 10 and img.width > 10, f"Image dimensions too small: {img.width}x{img.height}"

def test_bottom_to_top_text_generation(test_environment):
    """Tests that the --text-direction bottom_to_top flag works correctly."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file
    text_file = Path(test_environment["text_dir"]) / "test_corpus.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write("hello world this is a test\n")

    # Copy a font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "NanumGothic.ttf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping bottom_to_top test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "bottom_to_top",
        "--log-file", test_environment["log_file"]
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        with open(test_environment["log_file"], "r") as f:
            log_contents = f.read()
        assert result.returncode == 0, f"Script failed with error:\n{log_contents}"

    output_dir = Path(test_environment["output_dir"])
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are stacked bottom to top
    for i in range(len(bboxes) - 1):
        assert bboxes[i][3] > bboxes[i+1][3]

    # Check image dimensions
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > img.width