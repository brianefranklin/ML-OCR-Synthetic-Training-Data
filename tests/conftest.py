import pytest
import shutil
import random
from pathlib import Path

# Centralized path definitions
TESTS_ROOT = Path(__file__).parent
PROJECT_ROOT = TESTS_ROOT.parent
DATA_NOSYNC_DIR = PROJECT_ROOT / "data.nosync"
FONTS_DIR = DATA_NOSYNC_DIR / "fonts"

@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for tests."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"
    log_dir = tmp_path / "logs"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()
    log_dir.mkdir()

    corpus_path = text_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 50)

    # Use the centralized FONTS_DIR constant
    source_font_dir = FONTS_DIR
    font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if font_files:
        random_fonts = random.sample(font_files, min(5, len(font_files)))
        for font_file in random_fonts:
            shutil.copy(font_file, fonts_dir)

    return {
        "text_file": str(corpus_path),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "text_dir": str(text_dir),
        "source_font_dir": source_font_dir,
        "log_dir": str(log_dir),
        "tmp_path": tmp_path
    }

@pytest.fixture
def test_font():
    """Fixture providing a test font."""
    # Use the centralized FONTS_DIR constant
    font_path = FONTS_DIR / "ABeeZee-Regular.ttf"
    if not font_path.exists():
        pytest.skip(f"Test font not found: {font_path}")
    return str(font_path)