
import pytest
import subprocess
import os
import shutil
import random
from pathlib import Path
import json
import numpy as np
from PIL import Image


@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for medium priority tests."""
    # Create temporary directories
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    # Create a basic corpus file
    corpus_path = text_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 100)

    # Copy a random selection of fonts
    source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
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
        "source_font_dir": source_font_dir
    }


def test_long_text_generation(test_environment):
    """Tests that the system can handle very long text (100+ characters)."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a corpus with very long continuous text
    long_text_file = Path(test_environment["text_dir"]) / "long_text.txt"
    long_text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 10  # 620 chars
    with open(long_text_file, "w", encoding="utf-8") as f:
        f.write(long_text)

    # Test with various long text lengths
    test_cases = [
        (100, 150),  # Medium-long
        (200, 250),  # Long
        (300, 400),  # Very long
    ]

    for min_len, max_len in test_cases:
        output_dir = Path(test_environment["output_dir"]) / f"long_{min_len}_{max_len}"
        output_dir.mkdir(exist_ok=True)

        command = [
            "python3",
            str(script_path),
            "--text-file", str(long_text_file),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", str(output_dir),
            "--num-images", "3",
            "--min-text-length", str(min_len),
            "--max-text-length", str(max_len)
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed for text length {min_len}-{max_len}: {result.stderr}"

        # Verify output
        json_files = list(output_dir.glob("image_*.json"))
        assert len(json_files) > 0, f"JSON files not created for length {min_len}-{max_len}"

        # Verify each generated image
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            text = label_data["text"]
            bboxes = label_data["char_bboxes"]
            filename = label_data["image_file"]

            # Check text length is within specified range
            assert min_len <= len(text) <= max_len, f"Text length {len(text)} outside range [{min_len}, {max_len}]"

            # Check bbox count matches text length
            assert len(bboxes) == len(text), f"Bbox count {len(bboxes)} doesn't match text length {len(text)}"

            # Check image was created and is not corrupted
            image_path = output_dir / filename
            assert image_path.exists(), f"Image {filename} not created"

            img = Image.open(image_path)
            width, height = img.size

            # Image should not be unreasonably large (max 50000 pixels in either dimension to account for canvas)
            assert width <= 50000, f"Image width {width} is unreasonably large for text length {len(text)}"
            assert height <= 50000, f"Image height {height} is unreasonably large for text length {len(text)}"

            # Image should not be too small either
            assert width >= 50, f"Image width {width} is too small"
            assert height >= 10, f"Image height {height} is too small"


def test_special_characters(test_environment):
    """Tests that the system handles special characters, numbers, and punctuation."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create test cases for different character sets
    test_cases = [
        ("numbers", "0123456789" * 10),
        ("punctuation", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" * 5),
        ("mixed", "Hello123! How are you? Testing @#$% symbols. 2024-10-02" * 3),
        ("unicode", "Café résumé naïve Zürich fiancée" * 5),
        ("spaces_newlines", "Line one\nLine two\nLine three\n\nDouble newline\n   Spaces   " * 3),
    ]

    for test_name, corpus in test_cases:
        output_dir = Path(test_environment["output_dir"]) / f"special_{test_name}"
        output_dir.mkdir(exist_ok=True)

        # Create corpus file for this test case
        corpus_file = Path(test_environment["text_dir"]) / f"{test_name}_corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(corpus)

        command = [
            "python3",
            str(script_path),
            "--text-file", str(corpus_file),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", str(output_dir),
            "--num-images", "2",
            "--min-text-length", "20",
            "--max-text-length", "50"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed for {test_name} characters: {result.stderr}"

        # Verify output
        json_files = list(output_dir.glob("image_*.json"))
        assert len(json_files) > 0, f"JSON files not created for {test_name}"

        assert len(json_files) >= 1, f"Not enough output for {test_name} test"

        # Verify each generated image
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            text = label_data["text"]
            bboxes = label_data["char_bboxes"]
            filename = label_data["image_file"]

            # Check bbox count matches text length
            assert len(bboxes) == len(text), f"Bbox mismatch for {test_name}: {len(bboxes)} bboxes vs {len(text)} chars"

            # Verify image exists and is valid
            image_path = output_dir / filename
            assert image_path.exists(), f"Image not created for {test_name}"

            img = Image.open(image_path)
            assert img.size[0] > 0 and img.size[1] > 0, f"Invalid image dimensions for {test_name}"


def test_font_compatibility(test_environment):
    """Tests that the system works with all available fonts."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    source_font_dir = test_environment["source_font_dir"]
    all_fonts = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if not all_fonts:
        pytest.skip("No fonts found in data/fonts directory")

    # Test a sample of fonts (or all if there aren't many)
    fonts_to_test = all_fonts if len(all_fonts) <= 20 else random.sample(all_fonts, 20)

    successful_fonts = []
    failed_fonts = []

    for font_path in fonts_to_test:
        font_name = font_path.name
        output_dir = Path(test_environment["output_dir"]) / f"font_{font_name.replace('.', '_')}"
        output_dir.mkdir(exist_ok=True)

        # Create a temporary fonts dir with just this font
        test_fonts_dir = output_dir / "fonts"
        test_fonts_dir.mkdir(exist_ok=True)
        shutil.copy(font_path, test_fonts_dir)

        command = [
            "python3",
            str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", str(test_fonts_dir),
            "--output-dir", str(output_dir),
            "--num-images", "1",
            "--min-text-length", "10",
            "--max-text-length", "30"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            # Verify output was created
            json_files = list(output_dir.glob("image_*.json"))
            image_files = list(output_dir.glob("image_*.png"))

            if len(json_files) > 0 and len(image_files) > 0:
                successful_fonts.append(font_name)
            else:
                failed_fonts.append((font_name, "Output files not created"))
        else:
            failed_fonts.append((font_name, result.stderr[:200]))

    # Report results
    success_rate = len(successful_fonts) / len(fonts_to_test)
    print(f"\nFont Compatibility Results:")
    print(f"  Tested: {len(fonts_to_test)} fonts")
    print(f"  Successful: {len(successful_fonts)} ({success_rate:.1%})")
    print(f"  Failed: {len(failed_fonts)}")

    if failed_fonts:
        print("\nFailed fonts:")
        for font_name, error in failed_fonts[:10]:  # Show first 10 failures
            print(f"  - {font_name}: {error[:100]}")

    # At least 70% of fonts should work
    assert success_rate >= 0.7, f"Only {success_rate:.1%} of fonts worked (expected >= 70%)"


def test_data_quality(test_environment):
    """Tests that generated images have sufficient quality for OCR training."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "10",
        "--min-text-length", "20",
        "--max-text-length", "50",
        # Note: removed --backgrounds-dir parameter (RGBA pipeline uses transparent backgrounds)
        "--text-color-mode", "per_glyph",
        "--color-palette", "vibrant",
        "--effect-type", "embossed",
        "--effect-depth", "0.5",
        "--overlap-intensity", "0.2",
        "--ink-bleed-intensity", "0.2"
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    image_files = list(output_dir.glob("image_*.png"))
    assert len(image_files) >= 10, "Not enough images generated"

    # Analyze quality metrics for each image
    quality_metrics = {
        'sufficient_contrast': 0,
        'sufficient_entropy': 0,
        'has_text_pixels': 0,
        'reasonable_text_ratio': 0,
    }

    for img_path in image_files:
        img = Image.open(img_path)

        # For RGBA images, only analyze the non-transparent regions
        if img.mode == 'RGBA':
            # Get alpha channel and non-transparent pixels
            alpha = np.array(img)[:, :, 3]
            non_transparent_mask = alpha > 0

            # Convert RGB channels to grayscale
            rgb = np.array(img)[:, :, :3]
            grayscale = np.dot(rgb, [0.2989, 0.5870, 0.1140])

            # Only analyze non-transparent pixels
            if non_transparent_mask.sum() > 0:
                img_array = grayscale[non_transparent_mask]
            else:
                continue  # Skip fully transparent images
        else:
            img_array = np.array(img.convert('L')).flatten()

        # 1. Check contrast (standard deviation)
        std_dev = np.std(img_array)
        if std_dev > 15:  # Sufficient contrast
            quality_metrics['sufficient_contrast'] += 1

        # 2. Check entropy (measure of information content)
        # Calculate histogram and entropy
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))

        if entropy > 1.0:  # Lower threshold for RGBA images with transparent backgrounds
            quality_metrics['sufficient_entropy'] += 1

        # 3. Check for text pixels (dark pixels in non-transparent regions)
        dark_pixel_ratio = np.sum(img_array < 200) / len(img_array)
        if dark_pixel_ratio > 0.01:  # At least 1% dark pixels
            quality_metrics['has_text_pixels'] += 1

        # 4. Check text-to-background ratio is reasonable
        if 0.05 <= dark_pixel_ratio <= 0.95:  # Between 5% and 95%
            quality_metrics['reasonable_text_ratio'] += 1

    # Calculate percentages
    total_images = len(image_files)
    results = {k: v / total_images for k, v in quality_metrics.items()}

    print(f"\nData Quality Results (n={total_images}):")
    for metric, percentage in results.items():
        print(f"  {metric}: {percentage:.1%}")

    # Assert quality thresholds
    # Note: Thresholds account for RGBA transparency and canvas placement
    # With transparent backgrounds, we only analyze non-transparent regions
    # Contrast: At least 50% should have good contrast (relaxed from 0.3 to 0.5 due to better RGBA handling)
    assert results['sufficient_contrast'] >= 0.5, f"Only {results['sufficient_contrast']:.1%} have sufficient contrast (expected >=50%)"
    # Entropy: At least 30% should have good information content (increased from 0.1 due to better handling)
    assert results['sufficient_entropy'] >= 0.3, f"Only {results['sufficient_entropy']:.1%} have sufficient entropy (expected >=30%)"
    # Text pixels: At least 70% should have visible text (increased from 0.5)
    assert results['has_text_pixels'] >= 0.7, f"Only {results['has_text_pixels']:.1%} have visible text pixels (expected >=70%)"
    # Text ratio: At least 30% should have reasonable text-to-background ratio
    assert results['reasonable_text_ratio'] >= 0.3, f"Only {results['reasonable_text_ratio']:.1%} have reasonable text ratio (expected >=30%)"

def test_edge_case_very_short_corpus(test_environment):
    """Tests handling of corpus shorter than min-text-length."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    # Create a very short corpus
    short_corpus_file = Path(test_environment["text_dir"]) / "short_corpus.txt"
    with open(short_corpus_file, "w", encoding="utf-8") as f:
        f.write("Hi")  # Only 2 characters

    command = [
        "python3",
        str(script_path),
        "--text-file", str(short_corpus_file),
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "1",
        "--min-text-length", "10",  # Request longer than corpus
        "--max-text-length", "20"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Should handle gracefully - either error or adapt
    # The current implementation will loop trying to get min_text_length
    # This test documents current behavior
    if result.returncode != 0:
        # Acceptable: Script recognizes impossibility and exits
        assert "text" in result.stderr.lower() or "corpus" in result.stderr.lower()
    else:
        # Acceptable: Script generates what it can (text might be shorter than min)
        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        if len(json_files) > 0:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            # Text length should be <= corpus length
            assert len(label_data["text"]) <= 2


def test_output_format_consistency(test_environment):
    """Tests that output JSON formats are consistent and parsable."""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "src" / "main.py"

    command = [
        "python3",
        str(script_path),
        "--text-file", test_environment["text_file"],
        "--fonts-dir", test_environment["fonts_dir"],
        "--output-dir", test_environment["output_dir"],
        "--num-images", "5"
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output_dir = Path(test_environment["output_dir"])
    json_files = list(output_dir.glob("image_*.json"))
    assert len(json_files) > 0, "JSON label files were not created."

    # Verify JSON format
    referenced_files = set()
    for i, json_file in enumerate(json_files, start=1):
        with open(json_file, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        filename = label_data["image_file"]
        referenced_files.add(filename)

        # Filename should follow pattern
        assert filename.startswith("image_") and filename.endswith(".png"), f"Invalid filename: {filename}"

        # JSON should have required fields
        assert "text" in label_data, f"File {i} missing 'text' field"
        assert "char_bboxes" in label_data, f"File {i} missing 'char_bboxes' field"

        # Check types
        assert isinstance(label_data["text"], str), f"File {i} 'text' is not a string"
        assert isinstance(label_data["char_bboxes"], list), f"File {i} 'char_bboxes' is not a list"

        # Check bbox format
        for j, bbox in enumerate(label_data["char_bboxes"]):
            assert isinstance(bbox, list), f"File {i} bbox {j} is not a list"
            assert len(bbox) == 4, f"File {i} bbox {j} doesn't have 4 coordinates"
            assert all(isinstance(coord, (int, float)) for coord in bbox), f"File {i} bbox {j} has non-numeric coordinates"

        # Verify corresponding image file exists
        image_path = output_dir / filename
        assert image_path.exists(), f"Referenced image {filename} doesn't exist"

    # Verify all image files have corresponding JSON
    actual_images = set(f.name for f in output_dir.glob("image_*.png"))
    unreferenced = actual_images - referenced_files
    assert len(unreferenced) == 0, f"Images without JSON files: {unreferenced}"
