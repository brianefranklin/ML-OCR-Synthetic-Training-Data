
import pytest
import subprocess
import os
import shutil
import time
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

    # Create a dummy corpus file
    corpus_path = text_dir / "corpus.txt"
    with open(corpus_path, "w") as f:
        f.write("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n")

    # Copy a real font file into the test environment
    # This makes the test more realistic
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_file_to_copy = source_font_dir / "AlegreyaSans-BlackItalic.ttf"
    
    if not font_file_to_copy.exists():
        pytest.skip(f"Font file not found at {font_file_to_copy}, skipping integration test.")

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

    result = subprocess.run(command, text=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    assert result.returncode == 0, f"Script failed with error:\n{result.stdout}"

    # --- Verify the output ---
    output_dir = Path(test_environment["output_dir"])
    
    # Check for labels.csv
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    # Check the content of labels.csv
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    # 1 header line + num_images_to_generate data lines
    assert len(lines) == num_images_to_generate + 1, "labels.csv has incorrect number of rows."

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
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    # Check the data line (skip header)
    header, data_line = lines[0], lines[1]
    assert header.strip() == "filename,text"

    import json
    filename, json_data = data_line.strip().split(',', 1)
    label_data = json.loads(json_data)

    assert "text" in label_data
    assert "bboxes" in label_data
    assert isinstance(label_data["bboxes"], list)
    assert len(label_data["text"]) == len(label_data["bboxes"])

    # Check format of a single bounding box
    if label_data["bboxes"]:
        bbox = label_data["bboxes"][0]
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

    result = subprocess.run(command, capture_output=True, text=True, check=False)
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
    labels_file = output_dir / "labels.csv"
    assert len(image_files) > 0, "Image was not generated for the second part of the test."
    assert labels_file.exists(), "labels.csv was not generated for the second part of the test."

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

    # Check that the generated files are gone
    image_files_after_clear = list(output_dir.glob("image_*.png"))
    assert len(image_files_after_clear) == 0, "Generated image was not deleted."
    assert not labels_file.exists(), "labels.csv was not deleted."

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
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r') as f:
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
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r') as f:
        lines = f.readlines()
    
    # Check the data lines (skip header)
    for line in lines[1:]:
        import json
        filename, json_data = line.strip().split(',', 1)
        label_data = json.loads(json_data)
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
    assert "Error: Fonts directory not specified or is not a valid directory." in result.stdout

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
    assert "Time limit of 0.01 seconds reached" in result.stdout
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
    assert f"No text found in {str(empty_text_file)}" in result.stdout


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
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    
    import json
    filename, json_data = lines[1].strip().split(',', 1)
    label_data = json.loads(json_data)
    bboxes = label_data["bboxes"]

    # Check that bboxes are ordered from right to left
    for i in range(len(bboxes) - 1):
        assert bboxes[i][0] > bboxes[i+1][0]

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
    labels_file = output_dir / "labels.csv"
    assert labels_file.exists(), "labels.csv was not created."

    with open(labels_file, 'r') as f:
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
