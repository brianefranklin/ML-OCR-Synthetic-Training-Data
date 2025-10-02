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
    
    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    return {
        "text_dir": str(text_dir),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir)
    }

def test_top_to_bottom_text_generation_korean(test_environment):
    """Tests that the --text-direction top_to_bottom flag works for Korean."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file with Korean text
    korean_text_file = Path(test_environment["text_dir"]) / "korean_corpus.txt"
    with open(korean_text_file, "w", encoding="utf-8") as f:
        f.write("안녕하세요\n안녕하세요\n안녕하세요\n안녕하세요\n안녕하세요\n")

    # Copy a Korean font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "NanumGothic.ttf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping Korean top_to_bottom test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(korean_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "top_to_bottom"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are stacked top to bottom
    for i in range(len(bboxes) - 1):
        assert bboxes[i][1] < bboxes[i+1][1]

    # Check image dimensions
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > img.width

def test_top_to_bottom_text_generation_japanese(test_environment):
    """Tests that the --text-direction top_to_bottom flag works for Japanese."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file with Japanese text
    japanese_text_file = Path(test_environment["text_dir"]) / "japanese_corpus.txt"
    with open(japanese_text_file, "w", encoding="utf-8") as f:
        f.write("こんにちは\nこんにちは\nこんにちは\nこんにちは\nこんにちは\n")

    # Copy a Japanese font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "NotoSerifCJKjp-Regular.otf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping Japanese top_to_bottom test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(japanese_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "top_to_bottom"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are stacked top to bottom
    for i in range(len(bboxes) - 1):
        assert bboxes[i][1] < bboxes[i+1][1]

    # Check image dimensions
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > img.width

def test_top_to_bottom_text_generation_chinese(test_environment):
    """Tests that the --text-direction top_to_bottom flag works for Chinese."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file with Chinese text
    chinese_text_file = Path(test_environment["text_dir"]) / "chinese_corpus.txt"
    with open(chinese_text_file, "w", encoding="utf-8") as f:
        f.write("你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界你好世界\n")

    # Copy a Chinese font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "NotoSansCJKtc-VF.otf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping Chinese top_to_bottom test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(chinese_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "top_to_bottom"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are stacked top to bottom
    for i in range(len(bboxes) - 1):
        assert bboxes[i][1] < bboxes[i+1][1]

    # Check image dimensions
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > img.width

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
        "--text-direction", "bottom_to_top"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

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
        assert bboxes[i][1] > bboxes[i+1][1]

    # Check image dimensions
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height > img.width