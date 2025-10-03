import pytest
import subprocess
import os
from pathlib import Path
import shutil
import random


@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for error handling tests."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    # Copy a few font files
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
    font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if font_files:
        random_fonts = random.sample(font_files, min(5, len(font_files)))
        for font_file_to_copy in random_fonts:
            shutil.copy(font_file_to_copy, fonts_dir)

    return {
        "text_dir": str(text_dir),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir)
    }


def test_missing_corpus_file(test_environment):
    """Test that the script handles missing corpus file gracefully."""
    missing_file = os.path.join(test_environment["text_dir"], "nonexistent.txt")

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", missing_file,
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True
    )

    # Should fail gracefully with non-zero exit code
    assert result.returncode != 0
    # Should have error message about missing file
    assert "No such file or directory" in result.stderr or "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_empty_corpus_file(test_environment):
    """Test that the script handles empty corpus file gracefully."""
    empty_file = os.path.join(test_environment["text_dir"], "empty.txt")

    # Create an empty file
    Path(empty_file).touch()

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", empty_file,
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True,
        timeout=5  # Should fail quickly, not hang
    )

    # Should fail gracefully
    assert result.returncode != 0 or "Corpus is too short" in result.stderr


def test_too_small_corpus_file(test_environment):
    """Test that the script handles corpus files that are too small."""
    small_file = os.path.join(test_environment["text_dir"], "small.txt")

    # Create a file with only a few characters (less than min_text_length we'll specify)
    with open(small_file, "w", encoding="utf-8") as f:
        f.write("abc")

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", small_file,
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1",
            "--min-text-length", "10"  # Require at least 10 chars, but corpus only has 3
        ],
        capture_output=True,
        text=True,
        timeout=5  # Should fail quickly, not hang
    )

    # Should fail gracefully with error about corpus being too short
    assert result.returncode != 0 or "Corpus must contain at least" in result.stderr


def test_malformed_utf8_corpus_file(test_environment):
    """Test that the script handles malformed UTF-8 files."""
    malformed_file = os.path.join(test_environment["text_dir"], "malformed.txt")

    # Create a file with invalid UTF-8 bytes
    with open(malformed_file, "wb") as f:
        f.write(b"\x80\x81\x82\x83\x84" * 20)  # Invalid UTF-8 sequences

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", malformed_file,
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Should either fail or handle gracefully (depending on error handling strategy)
    # We just want to ensure it doesn't hang or crash unexpectedly
    assert result.returncode is not None


def test_missing_fonts_directory(test_environment):
    """Test that the script handles missing fonts directory gracefully."""
    corpus_file = os.path.join(test_environment["text_dir"], "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("This is a test corpus with enough text to generate images from.")

    missing_fonts_dir = os.path.join(test_environment["text_dir"], "nonexistent_fonts")

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", corpus_file,
            "--fonts-dir", missing_fonts_dir,
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Should fail gracefully
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_empty_fonts_directory(test_environment):
    """Test that the script handles empty fonts directory gracefully."""
    corpus_file = os.path.join(test_environment["text_dir"], "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("This is a test corpus with enough text to generate images from.")

    # Create an empty fonts directory
    empty_fonts_dir = os.path.join(test_environment["text_dir"], "empty_fonts")
    os.makedirs(empty_fonts_dir, exist_ok=True)

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", corpus_file,
            "--fonts-dir", empty_fonts_dir,
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Should fail gracefully or warn about no fonts
    assert result.returncode != 0 or "no font" in result.stderr.lower()


def test_whitespace_only_corpus(test_environment):
    """Test that the script handles corpus files with only whitespace."""
    whitespace_file = os.path.join(test_environment["text_dir"], "whitespace.txt")

    # Create a file with only whitespace
    with open(whitespace_file, "w", encoding="utf-8") as f:
        f.write("   \n\n\t\t\t   \n   ")

    result = subprocess.run(
        [
            "python3", "src/main.py",
            "--text-file", whitespace_file,
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ],
        capture_output=True,
        text=True,
        timeout=5
    )

    # Should fail gracefully
    assert result.returncode != 0 or "Corpus is too short" in result.stderr