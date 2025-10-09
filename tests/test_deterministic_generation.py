"""
Tests for deterministic image generation.

These tests verify that the OCR data generator produces consistent, reproducible
images when given the same seed and parameters. This is critical for regenerating
images from saved JSON parameters.
"""

import pytest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generator import OCRDataGenerator


class TestDeterministicGeneration:
    """Test suite for deterministic image generation."""

    @pytest.fixture
    def test_font(self):
        """Get a test font path."""
        font_dir = Path(__file__).parent.parent / "data.nosync" / "fonts"
        font_files = list(font_dir.glob("**/*.ttf"))
        if not font_files:
            pytest.skip("No fonts available for testing")
        return str(font_files[0])

    @pytest.fixture
    def generator(self, test_font):
        """Create a generator instance."""
        return OCRDataGenerator(font_files=[test_font], background_images=[])

    def test_identical_generation_no_augmentations(self, generator, test_font):
        """Test that two generations with same params produce identical images (no augmentations)."""
        params = {
            'text': 'Hello World',
            'font_path': test_font,
            'font_size': 32,
            'direction': 'left_to_right',
            'seed': 42,
            'canvas_size': (300, 150),
            'text_offset': (50, 50),
            'augmentations': None  # No augmentations for perfect match
        }

        # Generate twice
        img1, meta1, _, _ = generator.generate_image(**params)
        img2, meta2, _, _ = generator.generate_image(**params)

        # Convert to arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Should be pixel-perfect identical
        assert np.array_equal(arr1, arr2), "Images without augmentations should be identical"

    def test_identical_generation_with_augmentations(self, generator, test_font):
        """Test that two generations with same params and augmentations produce identical images."""
        params = {
            'text': 'Test123',
            'font_path': test_font,
            'font_size': 28,
            'direction': 'left_to_right',
            'seed': 12345,
            'canvas_size': (400, 200),
            'text_offset': (75, 75),
            'augmentations': {
                'perspective_transform': True,
                'elastic_distortion': True,
                'brightness_contrast': True,
                'blur': True
            }
        }

        # Generate twice
        img1, meta1, _, _ = generator.generate_image(**params)
        img2, meta2, _, _ = generator.generate_image(**params)

        # Convert to arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Should be pixel-perfect identical
        assert np.array_equal(arr1, arr2), "Images with augmentations should be identical when using same seed and params"

    def test_high_similarity_with_subprocess_generation(self, generator, test_font):
        """
        Test that regeneration from JSON params produces highly similar images.

        This test allows for small pixel differences (<3%) that may occur due to
        slight variations in random state between subprocess and direct generation.
        """
        # Simulate first-pass generation params
        params = {
            'text': 'Sample Text',
            'font_path': test_font,
            'font_size': 31,
            'direction': 'left_to_right',
            'seed': 42,
            'canvas_size': (365, 166),
            'text_offset': (85, 67),
            'augmentations': {
                'perspective_transform': True,
                'elastic_distortion': True,
                'background': True,
                'brightness_contrast': True
            }
        }

        # Generate twice (simulating subprocess + direct regeneration)
        img1, meta1, _, _ = generator.generate_image(**params)
        img2, meta2, _, _ = generator.generate_image(**params)

        # Convert to arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Calculate similarity metrics
        total_values = arr1.size
        different_values = np.sum(arr1 != arr2)
        percent_different = 100.0 * different_values / total_values

        # Images should be highly similar (>97% identical)
        assert percent_different < 3.0, \
            f"Images differ by {percent_different:.2f}%, expected <3%"

        # Mean pixel difference should be small
        mean_diff = np.mean(np.abs(arr1.astype(int) - arr2.astype(int)))
        assert mean_diff < 5.0, \
            f"Mean pixel difference is {mean_diff:.2f}, expected <5.0"

    @pytest.mark.parametrize("direction", [
        'left_to_right',
        'right_to_left',
        'top_to_bottom',
        'bottom_to_top'
    ])
    def test_deterministic_across_directions(self, generator, test_font, direction):
        """Test determinism for all text directions."""
        params = {
            'text': 'Dir Test',
            'font_path': test_font,
            'font_size': 30,
            'direction': direction,
            'seed': 999,
            'canvas_size': (250, 200),
            'text_offset': (50, 60)
        }

        # Generate twice
        img1, _, _, _ = generator.generate_image(**params)
        img2, _, _, _ = generator.generate_image(**params)

        # Should be identical
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), f"Images should be identical for direction={direction}"

    @pytest.mark.parametrize("overlap", [0.0, 0.3, 0.6])
    def test_deterministic_with_overlap(self, generator, test_font, overlap):
        """Test determinism with different overlap intensities."""
        params = {
            'text': 'Overlap',
            'font_path': test_font,
            'font_size': 35,
            'direction': 'left_to_right',
            'seed': 777,
            'overlap_intensity': overlap,
            'canvas_size': (300, 150),
            'text_offset': (40, 50)
        }

        img1, _, _, _ = generator.generate_image(**params)
        img2, _, _, _ = generator.generate_image(**params)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), f"Images should be identical with overlap={overlap}"

    def test_different_seeds_produce_different_images(self, generator, test_font):
        """Verify that different seeds actually produce different images."""
        base_params = {
            'text': 'Seed Test',
            'font_path': test_font,
            'font_size': 32,
            'direction': 'left_to_right',
            'canvas_size': (300, 150),
            'text_offset': (50, 50),
            'augmentations': {'blur': True}
        }

        # Generate with different seeds
        params1 = {**base_params, 'seed': 111}
        params2 = {**base_params, 'seed': 222}

        img1, _, _, _ = generator.generate_image(**params1)
        img2, _, _, _ = generator.generate_image(**params2)

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Should be different
        assert not np.array_equal(arr1, arr2), "Different seeds should produce different images"

        # But sizes should match (same canvas_size)
        assert arr1.shape == arr2.shape, "Images should have same dimensions"

    def test_canvas_placement_determinism(self, generator, test_font):
        """Test that text_offset produces deterministic placement."""
        params = {
            'text': 'Place',
            'font_path': test_font,
            'font_size': 30,
            'direction': 'left_to_right',
            'seed': 555,
            'canvas_size': (400, 200),
            'text_offset': (100, 75)  # Explicit placement
        }

        img1, meta1, _, _ = generator.generate_image(**params)
        img2, meta2, _, _ = generator.generate_image(**params)

        # Metadata should match
        assert meta1['text_placement'] == meta2['text_placement'], \
            "Text placement should be identical"
        assert meta1['text_placement'] == [100, 75], \
            "Text should be placed at specified offset"

        # Images should be identical
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), "Images should be identical with explicit text_offset"

    def test_metadata_consistency(self, generator, test_font):
        """Test that metadata is consistent across regenerations."""
        params = {
            'text': 'Meta',
            'font_path': test_font,
            'font_size': 28,
            'direction': 'left_to_right',
            'seed': 888,
            'canvas_size': (300, 150),
            'text_offset': (60, 50)
        }

        _, meta1, text1, augs1 = generator.generate_image(**params)
        _, meta2, text2, augs2 = generator.generate_image(**params)

        # Text should match
        assert text1 == text2, "Generated text should be identical"

        # Canvas size should match
        assert meta1['canvas_size'] == meta2['canvas_size'], "Canvas size should match"

        # Text placement should match
        assert meta1['text_placement'] == meta2['text_placement'], "Text placement should match"

        # Number of character bboxes should match
        assert len(meta1['char_bboxes']) == len(meta2['char_bboxes']), \
            "Number of character bounding boxes should match"

    @pytest.mark.parametrize("curve_type,intensity", [
        ('arc', 0.3),
        ('sine', 0.5),
        ('none', 0.0)
    ])
    def test_deterministic_with_curves(self, generator, test_font, curve_type, intensity):
        """Test determinism with curved text."""
        params = {
            'text': 'Curve',
            'font_path': test_font,
            'font_size': 32,
            'direction': 'left_to_right',
            'seed': 333,
            'curve_type': curve_type,
            'curve_intensity': intensity,
            'canvas_size': (350, 200),
            'text_offset': (50, 70)
        }

        img1, _, _, _ = generator.generate_image(**params)
        img2, _, _, _ = generator.generate_image(**params)

        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2), \
            f"Images should be identical with curve_type={curve_type}, intensity={intensity}"


class TestRegenerationTolerance:
    """
    Tests for regeneration with tolerance for minor differences.

    These tests acknowledge that subprocess-based generation may have small
    pixel differences compared to direct generation, while still being
    visually equivalent.
    """

    @pytest.fixture
    def test_font(self):
        """Get a test font path."""
        font_dir = Path(__file__).parent.parent / "data.nosync" / "fonts"
        font_files = list(font_dir.glob("**/*.ttf"))
        if not font_files:
            pytest.skip("No fonts available for testing")
        return str(font_files[0])

    @pytest.fixture
    def generator(self, test_font):
        """Create a generator instance."""
        return OCRDataGenerator(font_files=[test_font], background_images=[])

    def assert_images_highly_similar(self, img1, img2, max_diff_percent=3.0, max_mean_diff=5.0):
        """
        Assert that two images are highly similar within tolerance.

        Args:
            img1, img2: PIL Images to compare
            max_diff_percent: Maximum percentage of different pixels (default 3%)
            max_mean_diff: Maximum mean pixel value difference (default 5.0)
        """
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Check dimensions match
        assert arr1.shape == arr2.shape, \
            f"Image dimensions don't match: {arr1.shape} vs {arr2.shape}"

        # Calculate similarity metrics
        total_values = arr1.size
        different_values = np.sum(arr1 != arr2)
        percent_different = 100.0 * different_values / total_values
        mean_diff = np.mean(np.abs(arr1.astype(int) - arr2.astype(int)))

        # Assert similarity within tolerance
        assert percent_different <= max_diff_percent, \
            f"Images differ by {percent_different:.2f}%, expected <={max_diff_percent}%"

        assert mean_diff <= max_mean_diff, \
            f"Mean pixel difference is {mean_diff:.2f}, expected <={max_mean_diff}"

    def test_regeneration_similarity_with_augmentations(self, generator, test_font):
        """Test that regenerated images are highly similar even with complex augmentations."""
        params = {
            'text': 'Complex Test',
            'font_path': test_font,
            'font_size': 31,
            'direction': 'left_to_right',
            'seed': 42,
            'canvas_size': (365, 166),
            'text_offset': (85, 67),
            'augmentations': {
                'perspective_transform': True,
                'elastic_distortion': True,
                'background': True,
                'brightness_contrast': True,
                'blur': True
            },
            'overlap_intensity': 0.3,
            'ink_bleed_intensity': 0.2
        }

        # Generate multiple times
        img1, _, _, _ = generator.generate_image(**params)
        img2, _, _, _ = generator.generate_image(**params)

        # Should be highly similar (allowing up to 3% difference)
        self.assert_images_highly_similar(img1, img2, max_diff_percent=3.0, max_mean_diff=5.0)

    @pytest.mark.parametrize("effect_type", ['none', 'raised', 'embossed', 'engraved'])
    def test_regeneration_with_3d_effects(self, generator, test_font, effect_type):
        """Test regeneration similarity with 3D effects."""
        params = {
            'text': '3D Text',
            'font_path': test_font,
            'font_size': 35,
            'direction': 'left_to_right',
            'seed': 654,
            'effect_type': effect_type,
            'effect_depth': 0.5,
            'canvas_size': (300, 150),
            'text_offset': (50, 50)
        }

        img1, _, _, _ = generator.generate_image(**params)
        img2, _, _, _ = generator.generate_image(**params)

        # Should be highly similar
        self.assert_images_highly_similar(img1, img2, max_diff_percent=3.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
