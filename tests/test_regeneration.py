import pytest
import subprocess
import os
import shutil
import random
from pathlib import Path
import json
import yaml
import sys

@pytest.fixture
def regeneration_environment(tmp_path):
    """Sets up a temporary directory structure for the regeneration test."""
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
        # Copy at least one font
        shutil.copy(random.choice(font_files), fonts_dir)

    return {
        "text_file": str(corpus_path),
        "fonts_dir": str(fonts_dir),
        "output_dir": str(output_dir),
        "tmp_path": tmp_path
    }



@pytest.mark.parametrize("config_params", [
    pytest.param({"text_direction": "left_to_right", "effect_type": "none"}, id="ltr_simple"),
    pytest.param({"text_direction": "right_to_left", "effect_type": "raised", "overlap_intensity": 0.2}, id="rtl_raised_overlap"),
    pytest.param({"text_direction": "top_to_bottom", "curve_type": "arc", "curve_intensity": 0.4}, id="ttb_arc_curve"),
    pytest.param({"text_direction": "bottom_to_top", "curve_type": "sine", "curve_intensity": 0.3, "text_color_mode": "per_glyph"}, id="btt_sine_per_glyph"),
    pytest.param({"text_direction": "left_to_right", "ink_bleed_intensity": 0.7, "overlap_intensity": 0.6}, id="ltr_heavy_bleed_overlap"),
    pytest.param({"text_direction": "left_to_right", "effect_type": "engraved", "text_color_mode": "gradient"}, id="ltr_engraved_gradient"),
    # New edge cases
    pytest.param({"min_text_length": 100, "max_text_length": 120}, id="long_text"),
    pytest.param({"font_size": 8}, id="small_font"),
    pytest.param({"font_size": 100}, id="large_font"),
    pytest.param({"corpus_content": "!@#$%^&*()_+=-`~[]{}|;':,./<>?"}, id="special_chars"),
])
class TestRegeneration:
    """Test the ability to regenerate an image from its generation_params."""

    def test_can_regenerate_from_generation_params(self, regeneration_environment, config_params):
        """
        Test that generation_params can be used to rerun the generator.
        This test is parameterized to run with a variety of configurations.
        """
        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        # Add src to path to allow for direct import of generator
        sys.path.insert(0, str(project_root / 'src'))

        from generator import OCRDataGenerator
        from font_utils import can_font_render_text

        # --- Step 1: First Generation ---
        run_config = config_params.copy()
        if "corpus_content" in run_config:
            corpus_path = regeneration_environment["tmp_path"] / "custom_corpus.txt"
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(run_config["corpus_content"])
            run_config["corpus_file"] = str(corpus_path)
            del run_config["corpus_content"]
        else:
            run_config["corpus_file"] = regeneration_environment["text_file"]

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "regeneration_test_batch",
                    "proportion": 1.0,
                    **run_config
                }
            ]
        }

        batch_config_path = regeneration_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", regeneration_environment["fonts_dir"],
            "--output-dir", regeneration_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Initial generation failed for config {config_params}: {result.stderr}"

        # --- Step 2: Read and Prepare for Second Generation ---
        json_files = list(Path(regeneration_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) == 1, "Expected 1 JSON file from initial generation"

        with open(json_files[0], 'r') as f:
            first_pass_data = json.load(f)
        
        params = first_pass_data['generation_params']

        # --- Step 2a: Validate generation_params ---
        expected_keys = ['font_path', 'text', 'font_size', 'text_direction']
        for key in expected_keys:
            assert key in params, f"generation_params is missing expected key: {key}"

        # Check that the config params are present in the output params
        for key in config_params:
            if key != 'corpus_content': # This is a helper param for the test
                assert key in params, f"generation_params is missing key from config_params: {key}"

        assert can_font_render_text(params['font_path'], params['text'], frozenset(params['text'])), \
            "Font from first pass cannot render the generated text."

        # --- Step 3: Second Generation (Directly calling the generator) ---
        generator = OCRDataGenerator(font_files=[params['font_path']], background_images=[])

        try:
            # Call generate_image with the exact parameters from the first run.
            # This is a smoke test to ensure the params are valid and don't crash the generator.
            regen_image, regen_metadata, _, _ = generator.generate_image(
                text=params['text'],
                font_path=params['font_path'],
                font_size=params['font_size'],
                direction=params['text_direction'],
                curve_type=params.get('curve_type', 'none'),
                curve_intensity=params.get('curve_intensity', 0.0),
                overlap_intensity=params.get('overlap_intensity', 0.0),
                ink_bleed_intensity=params.get('ink_bleed_intensity', 0.0),
                effect_type=params.get('effect_type', 'none'),
                effect_depth=params.get('effect_depth', 0.5),
                light_azimuth=params.get('light_azimuth', 135.0),
                light_elevation=params.get('light_elevation', 45.0),
                text_color_mode=params.get('text_color_mode', 'uniform'),
                color_palette=params.get('color_palette', 'realistic_dark'),
                custom_colors=params.get('custom_colors'),
                background_color=params.get('background_color', 'auto')
            )
        except Exception as e:
            pytest.fail(f"Regeneration failed for config {config_params} with error: {e}")

        # --- Step 4: Assertions ---
        assert regen_image is not None, "Regenerated image is None"
        assert regen_image.width > 0 and regen_image.height > 0, "Regenerated image has invalid dimensions"
        assert 'char_bboxes' in regen_metadata, "Regenerated metadata is missing 'char_bboxes'"
        assert len(regen_metadata['char_bboxes']) == len(params['text']), \
            "Regenerated bbox count does not match text length"