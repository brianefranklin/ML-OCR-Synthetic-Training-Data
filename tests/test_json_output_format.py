"""
Comprehensive tests for JSON output format.
Tests generation_params metadata, canvas metadata, and complete schema validation.
"""

import pytest
import subprocess
import os
import shutil
import random
from pathlib import Path
import json
import yaml


@pytest.fixture
def test_environment(tmp_path):
    """Sets up a temporary directory structure for JSON format tests."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    fonts_dir = input_dir / "fonts"
    text_dir = input_dir / "text"

    fonts_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    output_dir.mkdir()

    # Create a corpus file
    corpus_path = text_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 50)

    # Copy a few fonts
    source_font_dir = Path(__file__).resolve().parent.parent / "data.nosync" / "fonts"
    font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if font_files:
        random_fonts = random.sample(font_files, min(3, len(font_files)))
        for font_file in random_fonts:
            shutil.copy(font_file, fonts_dir)

    return {
        "text_file": str(corpus_path),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "tmp_path": tmp_path
    }


class TestBasicJSONStructure:
    """Test basic JSON structure and required fields."""

    def test_json_has_required_fields(self, test_environment):
        """Test that JSON output contains all required fields."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) == 1, "Expected 1 JSON file"

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Required fields
        required_fields = ['image_file', 'text', 'canvas_size', 'text_placement', 'line_bbox', 'char_bboxes']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_json_field_types(self, test_environment):
        """Test that JSON fields have correct types."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Type checks
        assert isinstance(data['image_file'], str)
        assert isinstance(data['text'], str)
        assert isinstance(data['canvas_size'], list) and len(data['canvas_size']) == 2
        assert isinstance(data['text_placement'], list) and len(data['text_placement']) == 2
        assert isinstance(data['line_bbox'], list) and len(data['line_bbox']) == 4
        assert isinstance(data['char_bboxes'], list)

    def test_char_bboxes_count_matches_text_length(self, test_environment):
        """Test that number of char_bboxes matches text length."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            assert len(data['char_bboxes']) == len(data['text']), \
                f"Bbox count mismatch in {json_file.name}"


class TestGenerationParamsBatchMode:
    """Test generation_params field in batch mode."""

    def test_generation_params_exists_in_batch_mode(self, test_environment):
        """Test that generation_params field exists in batch mode."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        # Create a simple batch config
        batch_config_data = {
            "total_images": 2,
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
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) >= 1

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        assert 'generation_params' in data, "Missing generation_params field in batch mode"

    def test_generation_params_has_all_fields(self, test_environment):
        """Test that generation_params contains all expected fields."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "comprehensive_batch",
                    "proportion": 1.0,
                    "text_direction": "left_to_right",
                    "corpus_file": test_environment["text_file"],
                    "curve_type": "arc",
                    "curve_intensity": 0.3,
                    "overlap_intensity": 0.2,
                    "ink_bleed_intensity": 0.15,
                    "effect_type": "embossed",
                    "effect_depth": 0.5,
                    "light_azimuth": 135.0,
                    "light_elevation": 45.0,
                    "text_color_mode": "per_glyph",
                    "color_palette": "vibrant"
                }
            ]
        }

        batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        params = data['generation_params']

        # Check all expected fields
        expected_fields = [
            'text', 'font_path', 'font_size', 'text_direction',
            'curve_type', 'curve_intensity', 'overlap_intensity', 'ink_bleed_intensity',
            'effect_type', 'effect_depth', 'light_azimuth', 'light_elevation',
            'text_color_mode', 'color_palette', 'custom_colors', 'background_color',
            'augmentations'
        ]

        for field in expected_fields:
            assert field in params, f"Missing field in generation_params: {field}"

    def test_generation_params_values_match_config(self, test_environment):
        """Test that generation_params values match what was specified in config."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "test_batch",
                    "proportion": 1.0,
                    "text_direction": "right_to_left",
                    "corpus_file": test_environment["text_file"],
                    "curve_type": "sine",
                    "curve_intensity": 0.4,
                    "overlap_intensity": 0.3,
                    "effect_type": "raised",
                    "effect_depth": 0.7,
                    "text_color_mode": "uniform",
                    "color_palette": "realistic_dark"
                }
            ]
        }

        batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        params = data['generation_params']

        # Verify values match config
        assert params['text_direction'] == 'right_to_left'
        assert params['curve_type'] == 'sine'
        assert params['curve_intensity'] == 0.4
        assert params['overlap_intensity'] == 0.3
        assert params['effect_type'] == 'raised'
        assert params['effect_depth'] == 0.7
        assert params['text_color_mode'] == 'uniform'
        assert params['color_palette'] == 'realistic_dark'

    def test_augmentations_list_is_present(self, test_environment):
        """Test that augmentations list is included in generation_params."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "test_batch",
                    "proportion": 1.0,
                    "corpus_file": test_environment["text_file"]
                }
            ]
        }

        batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        params = data['generation_params']
        assert 'augmentations' in params
        assert isinstance(params['augmentations'], dict)


class TestCanvasMetadata:
    """Test canvas-related metadata fields."""

    def test_canvas_size_valid(self, test_environment):
        """Test that canvas_size is valid and reasonable."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            canvas_size = data['canvas_size']
            assert len(canvas_size) == 2
            assert canvas_size[0] > 0 and canvas_size[1] > 0
            assert canvas_size[0] <= 50000 and canvas_size[1] <= 50000  # Reasonable limits

    def test_text_placement_within_canvas(self, test_environment):
        """Test that text_placement is within canvas bounds."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            canvas_size = data['canvas_size']
            text_placement = data['text_placement']
            line_bbox = data['line_bbox']

            # Text placement should be non-negative
            assert text_placement[0] >= 0
            assert text_placement[1] >= 0

            # Line bbox should be within canvas (with small tolerance)
            assert line_bbox[2] <= canvas_size[0] + 5
            assert line_bbox[3] <= canvas_size[1] + 5

    def test_line_bbox_encloses_char_bboxes(self, test_environment):
        """Test that line_bbox encloses all char_bboxes."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            line_bbox = data['line_bbox']
            char_bboxes = data['char_bboxes']

            if not char_bboxes:
                continue

            # Calculate bounding box of all characters
            all_x_min = min(bbox[0] for bbox in char_bboxes)
            all_y_min = min(bbox[1] for bbox in char_bboxes)
            all_x_max = max(bbox[2] for bbox in char_bboxes)
            all_y_max = max(bbox[3] for bbox in char_bboxes)

            # Line bbox should enclose all characters (with tolerance for augmentations)
            tolerance = 50
            assert line_bbox[0] <= all_x_min + tolerance, \
                f"Line bbox x_min {line_bbox[0]} doesn't enclose char x_min {all_x_min}"
            assert line_bbox[1] <= all_y_min + tolerance, \
                f"Line bbox y_min {line_bbox[1]} doesn't enclose char y_min {all_y_min}"
            assert line_bbox[2] >= all_x_max - tolerance, \
                f"Line bbox x_max {line_bbox[2]} doesn't enclose char x_max {all_x_max}"
            assert line_bbox[3] >= all_y_max - tolerance, \
                f"Line bbox y_max {line_bbox[3]} doesn't enclose char y_max {all_y_max}"


class TestJSONConsistency:
    """Test JSON output consistency across multiple generations."""

    def test_all_json_files_have_same_structure(self, test_environment):
        """Test that all generated JSON files have the same structure."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 5,
            "batches": [
                {
                    "name": "batch1",
                    "proportion": 0.6,
                    "text_direction": "left_to_right",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch2",
                    "proportion": 0.4,
                    "text_direction": "right_to_left",
                    "corpus_file": test_environment["text_file"]
                }
            ]
        }

        batch_config_path = test_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) >= 3

        # Get keys from first file
        with open(json_files[0], 'r') as f:
            first_keys = set(json.load(f).keys())

        # Check all files have same top-level keys
        for json_file in json_files[1:]:
            with open(json_file, 'r') as f:
                keys = set(json.load(f).keys())
            assert keys == first_keys, f"Inconsistent keys in {json_file.name}"

    def test_json_is_valid_json(self, test_environment):
        """Test that all JSON files are valid and parseable."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {json_file.name}: {e}")

    def test_image_file_references_match_actual_files(self, test_environment):
        """Test that image_file references in JSON match actual image files."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "3"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        output_dir = Path(test_environment["output_dir"])
        json_files = list(output_dir.glob("image_*.json"))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            image_file = data['image_file']
            image_path = output_dir / image_file

            assert image_path.exists(), f"Referenced image {image_file} doesn't exist"
            assert image_path.suffix == '.png', f"Image file {image_file} is not a PNG"

    def setup_test_environment(self, tmp_path, name):
        """Helper to create isolated environments for different test runs."""
        run_path = tmp_path / name
        input_dir = run_path / "input"
        output_dir = run_path / "output"
        fonts_dir = input_dir / "fonts"
        text_dir = input_dir / "text"

        fonts_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)
        output_dir.mkdir()

        corpus_path = text_dir / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("Test corpus. " * 20)

        source_font_dir = Path(__file__).resolve().parent.parent / "data.nosync" / "fonts"
        font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))
        if font_files:
            shutil.copy(random.choice(font_files), fonts_dir)

        return {
            "text_file": str(corpus_path),
            "fonts_dir": str(fonts_dir),
            "output_dir": str(output_dir),
            "tmp_path": run_path
        }

    def test_generation_params_keys_are_consistent_across_modes(self, tmp_path):
        """Test that generation_params keys are consistent between standard and batch modes."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        # --- Setup for two separate runs ---
        standard_env = self.setup_test_environment(tmp_path, "standard")
        batch_env = self.setup_test_environment(tmp_path, "batch")

        # --- Run 1: Standard Mode ---
        standard_command = [
            "python3", str(script_path),
            "--text-file", standard_env["text_file"],
            "--fonts-dir", standard_env["fonts_dir"],
            "--output-dir", standard_env["output_dir"],
            "--num-images", "1",
            "--effect-type", "raised"
        ]
        result_standard = subprocess.run(standard_command, capture_output=True, text=True, check=False)
        assert result_standard.returncode == 0, f"Standard mode script failed: {result_standard.stderr}"

        # --- Run 2: Batch Mode ---
        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "test_batch",
                    "proportion": 1.0,
                    "corpus_file": batch_env["text_file"],
                    "effect_type": "raised"
                }
            ]
        }
        batch_config_path = batch_env["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        batch_command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", batch_env["fonts_dir"],
            "--output-dir", batch_env["output_dir"]
        ]
        result_batch = subprocess.run(batch_command, capture_output=True, text=True, check=False)
        assert result_batch.returncode == 0, f"Batch mode script failed: {result_batch.stderr}"

        # --- Compare Results ---
        standard_json_files = list(Path(standard_env["output_dir"]).glob("image_*.json"))
        assert len(standard_json_files) == 1
        with open(standard_json_files[0], 'r') as f:
            standard_data = json.load(f)
        standard_keys = set(standard_data['generation_params'].keys())

        batch_json_files = list(Path(batch_env["output_dir"]).glob("image_*.json"))
        assert len(batch_json_files) == 1
        with open(batch_json_files[0], 'r') as f:
            batch_data = json.load(f)
        batch_keys = set(batch_data['generation_params'].keys())

        assert standard_keys == batch_keys, \
            f"Key mismatch between standard and batch modes.\nStandard keys: {standard_keys}\nBatch keys: {batch_keys}"

class TestGenerationParamsStandardMode:
    """Test generation_params field in standard (non-batch) mode."""

    def test_generation_params_exists_in_standard_mode(self, test_environment):
        """Test that generation_params field exists in standard mode."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) == 1

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        assert 'generation_params' in data, "Missing generation_params field in standard mode"

    def test_generation_params_values_match_cli_args(self, test_environment):
        """Test that generation_params values match CLI args in standard mode."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        command = [
            "python3", str(script_path),
            "--text-file", test_environment["text_file"],
            "--fonts-dir", test_environment["fonts_dir"],
            "--output-dir", test_environment["output_dir"],
            "--num-images", "1",
            "--text-direction", "top_to_bottom",
            "--overlap-intensity", "0.6",
            "--ink-bleed-intensity", "0.7",
            "--effect-type", "engraved",
            "--effect-depth", "0.8",
            "--text-color-mode", "gradient",
            "--color-palette", "pastels"
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) == 1

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        params = data['generation_params']

        # Verify values match CLI args
        assert params['text_direction'] == 'top_to_bottom'
        assert params['overlap_intensity'] == 0.6
        assert params['ink_bleed_intensity'] == 0.7
        assert params['effect_type'] == 'engraved'
        assert params['effect_depth'] == 0.8
        assert params['text_color_mode'] == 'gradient'
        assert isinstance(params['augmentations'], dict)

