"""
Comprehensive test suite for canvas placement functionality.
Tests placing text images on larger canvases with line-level and character-level bounding boxes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from PIL import Image, ImageDraw
import numpy as np
import json
from pathlib import Path


@pytest.fixture
def sample_text_image():
    """Create a sample text image with known dimensions."""
    img = Image.new('RGB', (200, 50), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 40], fill='black')
    return img


@pytest.fixture
def sample_char_bboxes():
    """Sample character bounding boxes relative to text image."""
    return [
        [10.0, 10.0, 40.0, 40.0],
        [45.0, 10.0, 75.0, 40.0],
        [80.0, 10.0, 110.0, 40.0],
        [115.0, 10.0, 145.0, 40.0],
        [150.0, 10.0, 190.0, 40.0]
    ]


class TestCanvasPlacement:
    """Test canvas placement function."""

    def test_place_on_canvas_creates_larger_image(self, sample_text_image):
        """Canvas should be larger than original text image."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=(400, 300)
        )

        assert canvas_img.size == (400, 300)
        assert canvas_img.size[0] > sample_text_image.size[0]
        assert canvas_img.size[1] > sample_text_image.size[1]

    def test_place_on_canvas_white_background(self, sample_text_image):
        """Canvas background should be transparent (RGBA mode)."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=(400, 300)
        )

        # Canvas should be RGBA mode
        assert canvas_img.mode == 'RGBA'

        # Check corners are transparent (alpha = 0)
        pixels = np.array(canvas_img)
        assert pixels[0, 0, 3] == 0, "Top-left corner should be transparent"
        assert pixels[-1, -1, 3] == 0, "Bottom-right corner should be transparent"
        assert pixels[0, -1, 3] == 0, "Top-right corner should be transparent"
        assert pixels[-1, 0, 3] == 0, "Bottom-left corner should be transparent"

    def test_place_on_canvas_metadata_structure(self, sample_text_image, sample_char_bboxes):
        """Metadata should contain all required fields."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            sample_char_bboxes,
            canvas_size=(400, 300)
        )

        assert 'canvas_size' in metadata
        assert 'text_placement' in metadata
        assert 'line_bbox' in metadata
        assert 'char_bboxes' in metadata

        assert len(metadata['canvas_size']) == 2
        assert len(metadata['text_placement']) == 2
        assert len(metadata['line_bbox']) == 4
        assert len(metadata['char_bboxes']) == len(sample_char_bboxes)

    def test_text_placement_within_bounds(self, sample_text_image):
        """Text should be placed within canvas bounds with minimum padding."""
        from canvas_placement import place_on_canvas

        min_padding = 10
        canvas_size = (400, 300)

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=canvas_size,
            min_padding=min_padding
        )

        x_offset, y_offset = metadata['text_placement']
        text_width, text_height = sample_text_image.size

        # Check minimum padding
        assert x_offset >= min_padding
        assert y_offset >= min_padding
        assert x_offset + text_width <= canvas_size[0] - min_padding
        assert y_offset + text_height <= canvas_size[1] - min_padding

    def test_char_bboxes_adjusted_for_offset(self, sample_text_image, sample_char_bboxes):
        """Character bboxes should be adjusted by text placement offset."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            sample_char_bboxes,
            canvas_size=(400, 300)
        )

        x_offset, y_offset = metadata['text_placement']

        # Each character bbox should be offset
        for i, char_bbox in enumerate(metadata['char_bboxes']):
            original_bbox = sample_char_bboxes[i]
            assert char_bbox[0] == original_bbox[0] + x_offset
            assert char_bbox[1] == original_bbox[1] + y_offset
            assert char_bbox[2] == original_bbox[2] + x_offset
            assert char_bbox[3] == original_bbox[3] + y_offset

    def test_line_bbox_encompasses_all_chars(self, sample_text_image, sample_char_bboxes):
        """Line bbox should encompass all character bboxes."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            sample_char_bboxes,
            canvas_size=(400, 300)
        )

        line_bbox = metadata['line_bbox']

        # Line bbox should contain all character bboxes
        for char_bbox in metadata['char_bboxes']:
            assert line_bbox[0] <= char_bbox[0]  # x_min
            assert line_bbox[1] <= char_bbox[1]  # y_min
            assert line_bbox[2] >= char_bbox[2]  # x_max
            assert line_bbox[3] >= char_bbox[3]  # y_max

    def test_line_bbox_matches_text_dimensions(self, sample_text_image, sample_char_bboxes):
        """Line bbox should match text image dimensions after placement."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            sample_char_bboxes,
            canvas_size=(400, 300)
        )

        line_bbox = metadata['line_bbox']
        x_offset, y_offset = metadata['text_placement']
        text_width, text_height = sample_text_image.size

        # Line bbox should match text placement + dimensions
        assert line_bbox[0] == x_offset
        assert line_bbox[1] == y_offset
        assert line_bbox[2] == x_offset + text_width
        assert line_bbox[3] == y_offset + text_height


class TestPlacementStrategies:
    """Test different text placement strategies."""

    def test_uniform_random_placement(self, sample_text_image):
        """Uniform random placement should vary across multiple calls."""
        from canvas_placement import place_on_canvas

        placements = []
        for _ in range(10):
            canvas_img, metadata = place_on_canvas(
                sample_text_image,
                [],
                canvas_size=(400, 300),
                placement='uniform_random'
            )
            placements.append(tuple(metadata['text_placement']))

        # Should have some variation (not all the same)
        unique_placements = set(placements)
        assert len(unique_placements) > 1

    def test_weighted_random_placement(self, sample_text_image):
        """Weighted random placement should vary but favor center."""
        from canvas_placement import place_on_canvas

        placements = []
        for _ in range(20):
            canvas_img, metadata = place_on_canvas(
                sample_text_image,
                [],
                canvas_size=(400, 300),
                placement='weighted_random'
            )
            placements.append(metadata['text_placement'])

        # Should have variation
        unique_placements = set([tuple(p) for p in placements])
        assert len(unique_placements) > 1

        # Calculate mean position (should be closer to center than edges)
        mean_x = np.mean([p[0] for p in placements])
        mean_y = np.mean([p[1] for p in placements])

        canvas_center_x = (400 - sample_text_image.size[0]) / 2
        canvas_center_y = (300 - sample_text_image.size[1]) / 2

        # Mean should be reasonably close to center
        assert abs(mean_x - canvas_center_x) < 50
        assert abs(mean_y - canvas_center_y) < 50

    def test_center_placement(self, sample_text_image):
        """Center placement should place text at exact center."""
        from canvas_placement import place_on_canvas

        canvas_size = (400, 300)
        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=canvas_size,
            placement='center'
        )

        x_offset, y_offset = metadata['text_placement']
        text_width, text_height = sample_text_image.size

        expected_x = (canvas_size[0] - text_width) // 2
        expected_y = (canvas_size[1] - text_height) // 2

        assert x_offset == expected_x
        assert y_offset == expected_y


class TestCanvasSizeGeneration:
    """Test random canvas size generation."""

    def test_random_canvas_size_within_constraints(self, sample_text_image):
        """Random canvas should respect text size and max megapixels."""
        from canvas_placement import generate_random_canvas_size

        min_padding = 10
        max_megapixels = 12

        for _ in range(20):
            canvas_size = generate_random_canvas_size(
                sample_text_image.size,
                min_padding=min_padding,
                max_megapixels=max_megapixels
            )

            # Should be larger than text + padding
            assert canvas_size[0] >= sample_text_image.size[0] + 2 * min_padding
            assert canvas_size[1] >= sample_text_image.size[1] + 2 * min_padding

            # Should not exceed max megapixels
            assert canvas_size[0] * canvas_size[1] <= max_megapixels * 1_000_000

    def test_random_canvas_size_variation(self, sample_text_image):
        """Random canvas sizes should vary."""
        from canvas_placement import generate_random_canvas_size

        sizes = []
        for _ in range(10):
            size = generate_random_canvas_size(sample_text_image.size)
            sizes.append(size)

        unique_sizes = set(sizes)
        assert len(unique_sizes) > 1

    def test_canvas_size_respects_minimum_padding(self, sample_text_image):
        """Canvas must have minimum padding around text."""
        from canvas_placement import generate_random_canvas_size

        min_padding = 50
        canvas_size = generate_random_canvas_size(
            sample_text_image.size,
            min_padding=min_padding
        )

        text_width, text_height = sample_text_image.size
        assert canvas_size[0] >= text_width + 2 * min_padding
        assert canvas_size[1] >= text_height + 2 * min_padding


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_text_image(self):
        """Handle very small text images."""
        from canvas_placement import place_on_canvas

        small_img = Image.new('RGB', (10, 10), color='white')
        canvas_img, metadata = place_on_canvas(
            small_img,
            [],
            canvas_size=(100, 100)
        )

        assert canvas_img.size == (100, 100)
        assert metadata['line_bbox'][2] - metadata['line_bbox'][0] == 10
        assert metadata['line_bbox'][3] - metadata['line_bbox'][1] == 10

    def test_very_large_text_image(self):
        """Handle large text images that need large canvas."""
        from canvas_placement import place_on_canvas

        large_img = Image.new('RGB', (2000, 1000), color='white')
        canvas_img, metadata = place_on_canvas(
            large_img,
            [],
            min_padding=10
        )

        # Canvas should accommodate large text + padding
        assert canvas_img.size[0] >= 2020
        assert canvas_img.size[1] >= 1020

    def test_empty_char_bboxes(self, sample_text_image):
        """Handle empty character bbox list."""
        from canvas_placement import place_on_canvas

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=(400, 300)
        )

        assert metadata['char_bboxes'] == []
        assert len(metadata['line_bbox']) == 4

    def test_single_character_bbox(self, sample_text_image):
        """Handle single character."""
        from canvas_placement import place_on_canvas

        single_bbox = [[10.0, 10.0, 40.0, 40.0]]
        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            single_bbox,
            canvas_size=(400, 300)
        )

        assert len(metadata['char_bboxes']) == 1

    def test_canvas_size_equals_text_plus_padding(self, sample_text_image):
        """Handle minimal canvas size (text + padding only)."""
        from canvas_placement import place_on_canvas

        min_padding = 10
        min_canvas_size = (
            sample_text_image.size[0] + 2 * min_padding,
            sample_text_image.size[1] + 2 * min_padding
        )

        canvas_img, metadata = place_on_canvas(
            sample_text_image,
            [],
            canvas_size=min_canvas_size,
            min_padding=min_padding
        )

        assert canvas_img.size == min_canvas_size
        # With minimal size, text should be exactly at padding offset
        assert metadata['text_placement'][0] == min_padding
        assert metadata['text_placement'][1] == min_padding


class TestJSONOutput:
    """Test JSON label file generation."""

    def test_create_label_json_structure(self, sample_text_image, sample_char_bboxes, tmp_path):
        """JSON label should have correct structure."""
        from canvas_placement import create_label_json

        metadata = {
            'canvas_size': [400, 300],
            'text_placement': [100, 50],
            'line_bbox': [100, 50, 300, 100],
            'char_bboxes': sample_char_bboxes
        }

        json_data = create_label_json(
            image_file="image_00000.png",
            text="hello",
            metadata=metadata
        )

        assert json_data['image_file'] == "image_00000.png"
        assert json_data['text'] == "hello"
        assert json_data['canvas_size'] == [400, 300]
        assert json_data['text_placement'] == [100, 50]
        assert json_data['line_bbox'] == [100, 50, 300, 100]
        assert json_data['char_bboxes'] == sample_char_bboxes

    def test_save_label_json_to_file(self, sample_char_bboxes, tmp_path):
        """Should save valid JSON file."""
        from canvas_placement import save_label_json

        metadata = {
            'canvas_size': [400, 300],
            'text_placement': [100, 50],
            'line_bbox': [100, 50, 300, 100],
            'char_bboxes': sample_char_bboxes
        }

        output_file = tmp_path / "image_00000.json"
        save_label_json(
            output_file,
            image_file="image_00000.png",
            text="hello",
            metadata=metadata
        )

        assert output_file.exists()

        # Load and validate JSON
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['image_file'] == "image_00000.png"
        assert loaded_data['text'] == "hello"
        assert loaded_data['canvas_size'] == [400, 300]

    def test_json_serializable(self, sample_char_bboxes):
        """All values should be JSON serializable."""
        from canvas_placement import create_label_json

        metadata = {
            'canvas_size': [400, 300],
            'text_placement': [100, 50],
            'line_bbox': [100, 50, 300, 100],
            'char_bboxes': sample_char_bboxes
        }

        json_data = create_label_json(
            image_file="image_00000.png",
            text="hello",
            metadata=metadata
        )

        # Should serialize without error
        json_str = json.dumps(json_data)
        assert isinstance(json_str, str)

        # Should deserialize correctly
        reloaded = json.loads(json_str)
        assert reloaded == json_data


class TestIntegration:
    """Integration tests with main pipeline."""

    def test_full_pipeline_with_canvas_placement(self, tmp_path):
        """Test complete pipeline from generation to JSON output."""
        from generator import OCRDataGenerator
        from canvas_placement import place_on_canvas, save_label_json
        from PIL import ImageFont

        # Create generator
        generator = OCRDataGenerator([], [])

        # Generate text image
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 32)
        text = "Test"

        img, char_boxes = generator.render_left_to_right(text, font)
        char_bboxes = [box.bbox for box in char_boxes]

        # Place on canvas
        canvas_img, metadata = place_on_canvas(
            img,
            char_bboxes,
            canvas_size=(400, 300)
        )

        # Save outputs
        img_file = tmp_path / "test_image.png"
        json_file = tmp_path / "test_image.json"

        canvas_img.save(img_file)
        save_label_json(json_file, "test_image.png", text, metadata)

        # Verify files exist
        assert img_file.exists()
        assert json_file.exists()

        # Verify image is correct size
        loaded_img = Image.open(img_file)
        assert loaded_img.size == (400, 300)

        # Verify JSON is valid
        with open(json_file, 'r') as f:
            label_data = json.load(f)

        assert label_data['text'] == "Test"
        assert label_data['canvas_size'] == [400, 300]
        assert len(label_data['char_bboxes']) == 4


class TestEndToEndIntegration:
    """Test canvas placement through end-to-end generation pipeline."""

    @pytest.fixture
    def integration_environment(self, tmp_path):
        """Set up environment for integration tests."""
        import shutil
        import random

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        fonts_dir = input_dir / "fonts"
        text_dir = input_dir / "text"

        fonts_dir.mkdir(parents=True)
        text_dir.mkdir(parents=True)
        output_dir.mkdir()

        # Create corpus file
        corpus_path = text_dir / "corpus.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 50)

        # Copy fonts
        source_font_dir = Path(__file__).resolve().parent.parent / "data" / "fonts"
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

    def test_generate_image_returns_canvas_metadata(self, integration_environment):
        """Test that generate_image returns canvas metadata in full pipeline."""
        import subprocess
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 1,
            "batches": [
                {
                    "name": "integration_test",
                    "proportion": 1.0,
                    "text_direction": "left_to_right",
                    "corpus_file": integration_environment["text_file"]
                }
            ]
        }

        batch_config_path = integration_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", integration_environment["fonts_dir"],
            "--output-dir", integration_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Check generated JSON has canvas metadata
        json_files = list(Path(integration_environment["output_dir"]).glob("image_*.json"))
        assert len(json_files) == 1

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        # Verify all canvas metadata fields are present
        assert 'canvas_size' in data
        assert 'text_placement' in data
        assert 'line_bbox' in data
        assert 'char_bboxes' in data

    def test_canvas_metadata_after_augmentations(self, integration_environment):
        """Test that canvas metadata is correct even after augmentations."""
        import subprocess
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 2,
            "batches": [
                {
                    "name": "augmented_test",
                    "proportion": 1.0,
                    "text_direction": "left_to_right",
                    "corpus_file": integration_environment["text_file"],
                    "curve_type": "arc",
                    "curve_intensity": 0.3,
                    "overlap_intensity": 0.2,
                    "effect_type": "embossed",
                    "effect_depth": 0.5
                }
            ]
        }

        batch_config_path = integration_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", integration_environment["fonts_dir"],
            "--output-dir", integration_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(integration_environment["output_dir"]).glob("image_*.json"))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Canvas metadata should exist
            canvas_size = data['canvas_size']
            text_placement = data['text_placement']
            line_bbox = data['line_bbox']
            char_bboxes = data['char_bboxes']

            # Basic validation
            assert len(canvas_size) == 2
            assert canvas_size[0] > 0 and canvas_size[1] > 0
            assert len(text_placement) == 2
            assert len(line_bbox) == 4
            assert len(char_bboxes) == len(data['text'])

    def test_canvas_with_all_directions(self, integration_environment):
        """Test canvas placement works with all text directions."""
        import subprocess
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        for direction in ['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top']:
            output_dir = Path(integration_environment["output_dir"]) / direction
            output_dir.mkdir(exist_ok=True)

            batch_config_data = {
                "total_images": 1,
                "batches": [
                    {
                        "name": f"{direction}_test",
                        "proportion": 1.0,
                        "text_direction": direction,
                        "corpus_file": integration_environment["text_file"]
                    }
                ]
            }

            batch_config_path = integration_environment["tmp_path"] / f"batch_{direction}.yaml"
            with open(batch_config_path, "w") as f:
                yaml.dump(batch_config_data, f)

            command = [
                "python3", str(script_path),
                "--batch-config", str(batch_config_path),
                "--fonts-dir", integration_environment["fonts_dir"],
                "--output-dir", str(output_dir)
            ]

            result = subprocess.run(command, capture_output=True, text=True, check=False)
            assert result.returncode == 0, f"Failed for {direction}: {result.stderr}"

            # Verify output
            json_files = list(output_dir.glob("image_*.json"))
            assert len(json_files) == 1

            with open(json_files[0], 'r') as f:
                data = json.load(f)

            assert 'canvas_size' in data
            assert 'line_bbox' in data
            assert 'char_bboxes' in data

    def test_line_bbox_contains_text_after_canvas_placement(self, integration_environment):
        """Test that line_bbox correctly encompasses text after canvas placement."""
        import subprocess
        import yaml

        project_root = Path(__file__).resolve().parent.parent
        script_path = project_root / "src" / "main.py"

        batch_config_data = {
            "total_images": 3,
            "batches": [
                {
                    "name": "bbox_test",
                    "proportion": 1.0,
                    "corpus_file": integration_environment["text_file"]
                }
            ]
        }

        batch_config_path = integration_environment["tmp_path"] / "batch_config.yaml"
        with open(batch_config_path, "w") as f:
            yaml.dump(batch_config_data, f)

        command = [
            "python3", str(script_path),
            "--batch-config", str(batch_config_path),
            "--fonts-dir", integration_environment["fonts_dir"],
            "--output-dir", integration_environment["output_dir"]
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0

        json_files = list(Path(integration_environment["output_dir"]).glob("image_*.json"))

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

            # Line bbox should encompass all characters (with tolerance)
            tolerance = 50
            assert line_bbox[0] <= all_x_min + tolerance
            assert line_bbox[1] <= all_y_min + tolerance
            assert line_bbox[2] >= all_x_max - tolerance
            assert line_bbox[3] >= all_y_max - tolerance
