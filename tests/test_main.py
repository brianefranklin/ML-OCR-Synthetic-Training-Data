
import pytest
import subprocess
import os
import shutil
import time
import random
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

    # Create a combined corpus file
    corpus_path = text_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n")
        # Add content from other corpus files
        for corpus_file in ["arabic_corpus.txt", "japanese_corpus.txt"]:
            source_corpus_path = Path(__file__).resolve().parent.parent / corpus_file
            if source_corpus_path.exists():
                with open(source_corpus_path, "r", encoding="utf-8") as source_f:
                    f.write(source_f.read())

    # Copy a random selection of 10 font files into the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if not font_files:
        pytest.skip("No font files found, skipping integration test.")

    random_fonts = random.sample(font_files, min(10, len(font_files)))

    for font_file_to_copy in random_fonts:
        shutil.copy(font_file_to_copy, fonts_dir)

    return {
        "text_file": str(corpus_path),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir)
    }

def test_main_script_execution(test_environment):
    """Runs the main.py script as a subprocess and checks its output."""
    num_images_to_generate = 2
    
    # Get the project root directory to correctly locate src/main.py
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Construct the command
    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", str(num_images_to_generate)
    ]

    result = subprocess.run(command, text=True, check=False, capture_output=True)
    
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
    assert "INFO - Script finished." in result.stderr

    # --- Verify the output ---
    output_dir = Path(test_environment["output_dir"])

    # Check for JSON label files (one per image)
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) == num_images_to_generate, f"Expected {num_images_to_generate} JSON files, found {len(json_files)}"

    # Check for the correct number of image files
    image_files = list(output_dir.glob("image_*.png"))
    assert len(image_files) == num_images_to_generate, "Incorrect number of images were generated."

def test_main_script_bbox_output(test_environment):
    """Runs the main.py script and verifies the bounding box output."""
    num_images_to_generate = 1
    
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", str(num_images_to_generate)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])

    # Check for JSON label files
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) == num_images_to_generate, f"Expected {num_images_to_generate} JSON files"

    import json
    # Read the first JSON file
    with open(json_files[0], 'r') as f:
        label_data = json.load(f)

    assert "text" in label_data
    assert "image_file" in label_data
    assert "char_bboxes" in label_data
    assert isinstance(label_data["char_bboxes"], list)
    assert len(label_data["text"]) == len(label_data["char_bboxes"])

    # Check format of a single bounding box
    if label_data["char_bboxes"]:
        bbox = label_data["char_bboxes"][0]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(coord, (int, float)) for coord in bbox)

def test_clear_output_functionality(test_environment):
    """Tests that the --clear-output flag removes files from the output directory."""
    output_dir = Path(test_environment["output_dir"])
    
    # 1. Test with a pre-existing dummy file
    dummy_file = output_dir / "dummy.txt"
    dummy_file.touch()
    assert dummy_file.exists()

    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--output-dir", str(output_dir),
        "--clear-output",
        "--force",
        "--num-images", "0" # Don't generate images, just clear
    ]

    result = subprocess.run(command, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
    assert not dummy_file.exists(), "Dummy file was not deleted by --clear-output."

    # 2. Test that it clears script-generated files
    # First, run generation to create some files
    generation_command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", str(output_dir),
        "--num-images", "1"
    ]
    subprocess.run(generation_command, check=True)
    
    image_files = list(output_dir.glob("image_*.png"))
    json_files = list(output_dir.glob("image_*.json"))
    assert len(image_files) > 0, "Image was not generated for the second part of the test."
    assert len(json_files) > 0, "JSON label files were not generated for the second part of the test."

    # Now, run the clear command again
    clear_command = [
        "python3",
        str(script_path),
        "--output-dir", str(output_dir),
        "--clear-output",
        "--force",
        "--num-images", "0"
    ]
    result = subprocess.run(clear_command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"
    assert f"INFO - Clearing output directory: {output_dir}" in result.stderr

    # Check that the generated files are gone
    image_files_after_clear = list(output_dir.glob("image_*.png"))
    json_files_after_clear = list(output_dir.glob("image_*.json"))
    assert len(image_files_after_clear) == 0, "Generated image was not deleted."
    assert len(json_files_after_clear) == 0, "JSON label files were not deleted."

def test_top_to_bottom_text_generation(test_environment):
    """Tests that the --text-direction top_to_bottom flag works."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "top_to_bottom"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) > 0, "JSON label files were not created."

    import json
    with open(json_files[0], 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    filename = label_data["image_file"]
    bboxes = label_data["char_bboxes"]

    # Check that bboxes are generally stacked top to bottom
    # Due to augmentations (rotation, perspective), strict ordering may not hold for every pair
    if len(bboxes) > 1:
        # Check that the first bbox is near the top and last bbox is near the bottom (or equal for single char)
        assert bboxes[0][1] <= bboxes[-1][1], "First character should be above or at same level as last character"

        # Check that majority of adjacent pairs are in top-to-bottom order
        # Using a lenient threshold (50%) because perspective and rotation augmentations can significantly affect ordering
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
    assert img.height > 10 and img.width >= 10, f"Image dimensions too small: {img.width}x{img.height}"
def test_variable_text_length(test_environment):
    """Tests that the --min-text-length and --max-text-length flags work."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    min_len, max_len = 5, 15

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "5",
        "--min-text-length", str(min_len),
        "--max-text-length", str(max_len)
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) > 0, "JSON label files were not created."

    import json
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        text = label_data["text"]
        assert min_len <= len(text) <= max_len, f"Generated text \"{text}\" has length outside the specified range."

def test_invalid_font_dir(test_environment):
    """Tests that the script exits gracefully if the fonts dir is invalid."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", "non_existent_dir",
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "ERROR - Error: Fonts directory not specified or is not a valid directory." in result.stderr

def test_max_execution_time(test_environment):
    """Tests that the --max-execution-time flag stops execution."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "10",
        "--max-execution-time", "0.01"
    ]

    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    end_time = time.time()
    
    assert result.returncode == 0
    assert "INFO - \nTime limit of 0.01 seconds reached. Stopping generation." in result.stderr
    assert (end_time - start_time) < 5 # Check that it didn't run for too long

def test_empty_text_file(test_environment):
    """Tests that the script handles an empty text file gracefully."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    empty_text_file = Path(test_environment["output_dir"]) / "empty.txt"
    empty_text_file.touch()

    command = [
        "python3",
        str(script_path),
        "--text-file", str(empty_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 # The script should not crash
    assert "ERROR - Corpus must contain at least" in result.stderr or f"ERROR - No text found in {str(empty_text_file)}" in result.stderr


def test_right_to_left_text_generation(test_environment):
    """Tests that the --text-direction right_to_left flag works."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file with Arabic text
    right_to_left_text_file = Path(test_environment["output_dir"]) / "right_to_left_corpus.txt"
    with open(right_to_left_text_file, "w", encoding="utf-8") as f:
        f.write("أهلاً بالعالم\n")

    # Copy an Arabic font to the test environment
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "NotoSansArabic[wdth,wght].ttf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping right_to_left test.")

    shutil.copy(font_file_to_copy, test_environment["fonts_dir"])

    command = [
        "python3",
        str(script_path),
        "--text-file", str(right_to_left_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "right_to_left"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) > 0, "JSON label files were not created."

    import json
    with open(json_files[0], 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    filename = label_data["image_file"]
    bboxes = label_data["char_bboxes"]

    # Check that bboxes are generally ordered from right to left
    # Due to augmentations, strict ordering may not hold for every pair
    if len(bboxes) > 1:
        # Check that the first bbox is on the right and last bbox is on the left
        assert bboxes[0][0] >= bboxes[-1][0], "First character should be to the right of or at same position as last character"

        # Check that majority of adjacent pairs are in right-to-left order
        # Using a lenient threshold (50%) because perspective and rotation augmentations can significantly affect ordering
        ordered_pairs = sum(1 for i in range(len(bboxes) - 1) if bboxes[i][0] > bboxes[i+1][0])
        total_pairs = len(bboxes) - 1
        if total_pairs > 0:
            assert ordered_pairs / total_pairs > 0.5, f"At least 50% of bboxes should be in right-to-left order, got {ordered_pairs}/{total_pairs}"

def test_bottom_to_top_text_generation(test_environment):
    """Tests that the --text-direction bottom_to_top flag works."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a dummy corpus file with long text
    long_text_file = Path(test_environment["output_dir"]) / "long_corpus.txt"
    with open(long_text_file, "w", encoding="utf-8") as f:
        f.write("abcdefghijklmnopqrstuvwxyz")

    command = [
        "python3",
        str(script_path),
        "--text-file", str(long_text_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--text-direction", "bottom_to_top"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) > 0, "JSON label files were not created."

    import json
    with open(json_files[0], 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    filename = label_data["image_file"]
    bboxes = label_data["char_bboxes"]

    # Check that bboxes are generally stacked bottom to top
    # Due to augmentations (rotation, perspective), strict ordering may not hold for every pair
    if len(bboxes) > 1:
        # Check that the first bbox is near the bottom and last bbox is near the top
        assert bboxes[0][1] >= bboxes[-1][1], "First character should be below or at same level as last character"

        # Check that majority of adjacent pairs are in bottom-to-top order
        # Using a lenient threshold (50%) because perspective and rotation augmentations can significantly affect ordering
        ordered_pairs = sum(1 for i in range(len(bboxes) - 1) if bboxes[i][1] > bboxes[i+1][1])
        total_pairs = len(bboxes) - 1
        if total_pairs > 0:
            assert ordered_pairs / total_pairs > 0.5, f"At least 50% of bboxes should be in bottom-to-top order, got {ordered_pairs}/{total_pairs}"

    # Check image dimensions are reasonable
    # Note: After augmentations (rotation, perspective), aspect ratio may change
    # So we just verify the image exists and isn't degenerate
    from PIL import Image
    image_path = output_dir / filename
    img = Image.open(image_path)
    assert img.height >= 10 and img.width >= 10, f"Image dimensions too small: {img.width}x{img.height}"
