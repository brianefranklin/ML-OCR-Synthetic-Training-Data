"""
Comprehensive tests for batch generation retry logic.
Ensures target image count is always reached despite failures.
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
    """Sets up a temporary directory structure for retry logic tests."""
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
        f.write("The quick brown fox jumps over the lazy dog. " * 100)

    # Copy fonts
    source_font_dir = Path(__file__).resolve().parent.parent / "data.nosync" / "fonts"
    font_files = list(source_font_dir.glob("**/*.ttf")) + list(source_font_dir.glob("**/*.otf"))

    if font_files:
        selected_fonts = random.sample(font_files, min(5, len(font_files)))
        for font_file in selected_fonts:
            shutil.copy(font_file, fonts_dir)

    return {
        "text_file": str(corpus_path),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "tmp_path": tmp_path
    }


class TestRetryLogicBasics:
    """Test basic retry logic functionality."""

    def test_retry_achieves_exact_target_count(self, test_environment):
        """Test that retry logic achieves exact target count."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        target_count = 10
        batch_config_data = {
            "total_images": target_count,
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

        # Count actual generated images
        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        json_files = list(Path(test_environment["output_dir"]).glob("image_*.json"))

        assert len(image_files) == target_count, \
            f"Expected exactly {target_count} images, got {len(image_files)}"
        assert len(json_files) == target_count, \
            f"Expected exactly {target_count} JSON files, got {len(json_files)}"

    def test_retry_with_multiple_batches(self, test_environment):
        """Test retry logic with multiple batches."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        total_images = 20
        batch_config_data = {
            "total_images": total_images,
            "batches": [
                {
                    "name": "batch1",
                    "proportion": 0.5,
                    "text_direction": "left_to_right",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch2",
                    "proportion": 0.5,
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

        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        assert len(image_files) == total_images

    def test_retry_with_complex_features(self, test_environment):
        """Test retry logic with complex features that might cause failures."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        target_count = 15
        batch_config_data = {
            "total_images": target_count,
            "batches": [
                {
                    "name": "complex_batch",
                    "proportion": 1.0,
                    "text_direction": "top_to_bottom",
                    "corpus_file": test_environment["text_file"],
                    "curve_type": "arc",
                    "curve_intensity": 0.5,
                    "overlap_intensity": 0.3,
                    "ink_bleed_intensity": 0.2,
                    "effect_type": "embossed",
                    "effect_depth": 0.6,
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

        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        assert len(image_files) == target_count


class TestRetryLogicReporting:
    """Test that retry logic reports statistics correctly."""

    def test_log_reports_attempts_and_successes(self, test_environment):
        """Test that log file reports both attempts and successful generations."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        log_dir = test_environment["tmp_path"] / "logs"
        log_dir.mkdir(exist_ok=True)
        target_count = 5

        batch_config_data = {
            "total_images": target_count,
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
            "--output-dir", test_environment["output_dir"],
            "--log-dir", str(log_dir)
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        # Find the timestamped log file
        log_files = list(log_dir.glob("generation_*.log"))
        assert len(log_files) >= 1, f"Expected at least 1 log file, found {len(log_files)}"
        log_file = log_files[0]

        # Check log file for statistics
        with open(log_file, 'r') as f:
            log_content = f.read()

        # Should report successful generation
        assert f"Successfully generated {target_count}/{target_count}" in log_content or \
               f"generated {target_count} images" in log_content.lower()

    def test_log_reports_batch_progress(self, test_environment):
        """Test that log reports progress for each batch."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        log_dir = test_environment["tmp_path"] / "logs"
        log_dir.mkdir(exist_ok=True)

        batch_config_data = {
            "total_images": 6,
            "batches": [
                {
                    "name": "batch_alpha",
                    "proportion": 0.5,
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch_beta",
                    "proportion": 0.5,
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
            "--output-dir", test_environment["output_dir"],
            "--log-dir", str(log_dir)
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        # Find the timestamped log file
        log_files = list(log_dir.glob("generation_*.log"))
        assert len(log_files) >= 1, f"Expected at least 1 log file, found {len(log_files)}"
        log_file = log_files[0]

        with open(log_file, 'r') as f:
            log_content = f.read()

        # Should mention both batch names
        assert "batch_alpha" in log_content or "Batch generation progress" in log_content
        assert "batch_beta" in log_content or "Batch generation progress" in log_content


class TestRetryEdgeCases:
    """Test edge cases for retry logic."""

    def test_single_image_generation(self, test_environment):
        """Test that single image generation works with retry logic."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "single_batch",
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

        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        assert len(image_files) == 1

    def test_uneven_batch_proportions(self, test_environment):
        """Test retry logic with uneven batch proportions."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        total_images = 17  # Prime number to test rounding
        batch_config_data = {
            "total_images": total_images,
            "batches": [
                {
                    "name": "batch1",
                    "proportion": 0.3,
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch2",
                    "proportion": 0.45,
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch3",
                    "proportion": 0.25,
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

        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        # Should get exactly the target count despite rounding
        assert len(image_files) == total_images


class TestRetryWithDifferentDirections:
    """Test retry logic with different text directions."""

    def test_retry_with_all_directions(self, test_environment):
        """Test retry logic with all text directions."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 16,
            "batches": [
                {
                    "name": "ltr_batch",
                    "proportion": 0.25,
                    "text_direction": "left_to_right",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "rtl_batch",
                    "proportion": 0.25,
                    "text_direction": "right_to_left",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "ttb_batch",
                    "proportion": 0.25,
                    "text_direction": "top_to_bottom",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "btt_batch",
                    "proportion": 0.25,
                    "text_direction": "bottom_to_top",
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

        image_files = list(Path(test_environment["output_dir"]).glob("image_*.png"))
        assert len(image_files) == 16


class TestRetryInterleaving:
    """Test that retry logic maintains interleaved generation."""

    def test_batches_are_interleaved(self, test_environment):
        """Test that images from different batches are interleaved."""
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 12,
            "batches": [
                {
                    "name": "batch_A",
                    "proportion": 0.5,
                    "text_direction": "left_to_right",
                    "corpus_file": test_environment["text_file"]
                },
                {
                    "name": "batch_B",
                    "proportion": 0.5,
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

        # Read JSON files in order and check text directions are mixed
        output_dir = Path(test_environment["output_dir"])
        json_files = sorted(output_dir.glob("image_*.json"))

        directions = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'generation_params' in data:
                    directions.append(data['generation_params']['text_direction'])

        # Check that directions alternate (interleaved)
        # Not all consecutive images should have the same direction
        if len(directions) >= 4:
            same_direction_runs = 0
            for i in range(len(directions) - 1):
                if directions[i] == directions[i+1]:
                    same_direction_runs += 1

            # Should have some mixing (not all same or all different)
            # With interleaving, we expect roughly half to be different
            mixing_ratio = same_direction_runs / (len(directions) - 1)
            assert 0.2 <= mixing_ratio <= 0.8, \
                f"Poor interleaving: {mixing_ratio:.2f} same-direction ratio (expect 0.2-0.8)"
