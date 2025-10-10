"""
Tests for BackgroundImageManager functionality.

Tests:
- Background discovery from directories
- Background validation (size checks)
- Background cropping
- Score persistence and updates
- Weighted selection
"""

import pytest
import os
import sys
import tempfile
import json
from PIL import Image

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from background_manager import BackgroundImageManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for test backgrounds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_backgrounds(temp_dir):
    """Create sample background images of various sizes."""
    backgrounds = {}

    # Small background (100x100)
    small_path = os.path.join(temp_dir, "small.png")
    small_img = Image.new('RGB', (100, 100), color='red')
    small_img.save(small_path)
    backgrounds['small'] = small_path

    # Medium background (500x500)
    medium_path = os.path.join(temp_dir, "medium.png")
    medium_img = Image.new('RGB', (500, 500), color='green')
    medium_img.save(medium_path)
    backgrounds['medium'] = medium_path

    # Large background (2000x2000)
    large_path = os.path.join(temp_dir, "large.png")
    large_img = Image.new('RGB', (2000, 2000), color='blue')
    large_img.save(large_path)
    backgrounds['large'] = large_path

    return backgrounds


def test_background_discovery(temp_dir, sample_backgrounds):
    """Test that BackgroundImageManager discovers images correctly."""
    manager = BackgroundImageManager(
        background_dirs=[temp_dir],
        pattern="*.png"
    )

    # Should discover all 3 backgrounds
    assert len(manager.backgrounds) == 3
    assert sample_backgrounds['small'] in manager.backgrounds
    assert sample_backgrounds['medium'] in manager.backgrounds
    assert sample_backgrounds['large'] in manager.backgrounds

    # All should have default score of 1.0
    assert manager.backgrounds[sample_backgrounds['small']] == 1.0
    assert manager.backgrounds[sample_backgrounds['medium']] == 1.0
    assert manager.backgrounds[sample_backgrounds['large']] == 1.0


def test_background_validation_too_small_for_text(sample_backgrounds):
    """Test validation rejects backgrounds smaller than text bbox."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png"
    )

    # Canvas 1000x1000, text bbox (0, 0, 200, 200) - small (100x100) should fail
    canvas_size = (1000, 1000)
    text_bbox = (0, 0, 200, 200)

    is_valid, reason, penalty = manager.validate_background(
        sample_backgrounds['small'],
        canvas_size,
        text_bbox
    )

    assert not is_valid
    assert "smaller than text" in reason.lower()
    assert penalty == 1.0  # Severe penalty


def test_background_validation_too_small_for_canvas(sample_backgrounds):
    """Test validation rejects backgrounds smaller than canvas."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png"
    )

    # Canvas 1000x1000, text bbox (0, 0, 50, 50) - small (100x100) should fail (too small for canvas)
    canvas_size = (1000, 1000)
    text_bbox = (0, 0, 50, 50)

    is_valid, reason, penalty = manager.validate_background(
        sample_backgrounds['small'],
        canvas_size,
        text_bbox
    )

    assert not is_valid
    assert "smaller than canvas" in reason.lower()
    assert penalty == 0.5  # Moderate penalty


def test_background_validation_larger_than_canvas(sample_backgrounds):
    """Test validation accepts backgrounds larger than canvas."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['large'])],
        pattern="*.png"
    )

    # Canvas 500x500, text bbox (0, 0, 50, 50) - large (2000x2000) should pass
    canvas_size = (500, 500)
    text_bbox = (0, 0, 50, 50)

    is_valid, reason, penalty = manager.validate_background(
        sample_backgrounds['large'],
        canvas_size,
        text_bbox
    )

    assert is_valid
    assert "valid" in reason.lower()
    assert penalty == 0.0


def test_background_crop_to_canvas_size(sample_backgrounds):
    """Test that backgrounds are cropped to canvas size."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['large'])],
        pattern="*.png"
    )

    canvas_size = (500, 500)

    cropped_img = manager.load_and_crop_background(
        sample_backgrounds['large'],
        canvas_size
    )

    assert cropped_img is not None
    assert cropped_img.size == canvas_size


def test_background_score_persistence(temp_dir, sample_backgrounds):
    """Test that background scores are persisted to disk."""
    score_file = os.path.join(temp_dir, "test_scores.json")

    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png",
        score_file=score_file,
        enable_persistence=True  # Enable persistence for this test
    )

    # Update score (penalty reduces score from 1.0)
    manager.update_score(sample_backgrounds['small'], 0.1)
    manager.finalize()

    # Check that score file was created
    assert os.path.exists(score_file)

    # Load and verify scores
    with open(score_file, 'r') as f:
        scores = json.load(f)

    assert sample_backgrounds['small'] in scores
    assert scores[sample_backgrounds['small']] < 1.0  # Should be 0.9 (1.0 - 0.1)
    assert abs(scores[sample_backgrounds['small']] - 0.9) < 0.01


def test_background_score_loading(temp_dir, sample_backgrounds):
    """Test that scores are loaded from disk on initialization."""
    score_file = os.path.join(temp_dir, "test_scores.json")

    # Create initial manager and set scores
    manager1 = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png",
        score_file=score_file,
        enable_persistence=True  # Enable persistence
    )
    manager1.update_score(sample_backgrounds['small'], 0.5)
    manager1.finalize()

    # Create new manager - should load existing scores
    manager2 = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png",
        score_file=score_file,
        enable_persistence=True  # Enable persistence
    )

    # Should have loaded the reduced score
    assert manager2.backgrounds[sample_backgrounds['small']] < 1.0
    assert abs(manager2.backgrounds[sample_backgrounds['small']] - 0.5) < 0.01


def test_background_selection_weighted(sample_backgrounds):
    """Test that backgrounds are selected with weighting."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png"
    )

    # Penalize small background heavily
    manager.update_score(sample_backgrounds['small'], 0.99)  # Score becomes 0.01

    # Select backgrounds multiple times - should favor medium and large
    selections = []
    for _ in range(30):
        selected = manager.select_background()
        selections.append(selected)

    # Small should be selected less often
    small_count = selections.count(sample_backgrounds['small'])
    total_count = len(selections)

    # With score 0.01 vs 1.0, small should be selected ~1% of the time
    # Allow for randomness: should be less than 20% of selections
    assert small_count / total_count < 0.2


def test_background_selection_no_backgrounds():
    """Test selection when no backgrounds are available."""
    manager = BackgroundImageManager(
        background_dirs=[],
        pattern="*.png"
    )

    selected = manager.select_background()
    assert selected is None


def test_background_statistics(sample_backgrounds):
    """Test statistics calculation."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png"
    )

    # All backgrounds should have score 1.0 initially
    stats = manager.get_statistics()
    assert stats['total_backgrounds'] == 3
    assert stats['avg_score'] == 1.0
    assert stats['min_score'] == 1.0
    assert stats['max_score'] == 1.0

    # Penalize one background
    manager.update_score(sample_backgrounds['small'], 0.5)

    stats = manager.get_statistics()
    assert stats['total_backgrounds'] == 3
    assert stats['avg_score'] < 1.0  # Average should decrease
    assert stats['min_score'] < 1.0  # Min should be 0.5
    assert stats['max_score'] == 1.0  # Max still 1.0


def test_brace_expansion_pattern(temp_dir):
    """Test glob pattern with brace expansion."""
    # Create images with different extensions
    jpg_path = os.path.join(temp_dir, "test.jpg")
    png_path = os.path.join(temp_dir, "test.png")
    Image.new('RGB', (100, 100)).save(jpg_path)
    Image.new('RGB', (100, 100)).save(png_path)

    manager = BackgroundImageManager(
        background_dirs=[temp_dir],
        pattern="*.{png,jpg}"
    )

    # Should discover both
    assert len(manager.backgrounds) == 2
    assert any(jpg_path in path for path in manager.backgrounds.keys())
    assert any(png_path in path for path in manager.backgrounds.keys())


def test_directory_weights(temp_dir, sample_backgrounds):
    """Test directory-based weighting."""
    # Create subdirectories
    dir1 = os.path.join(temp_dir, "high_priority")
    dir2 = os.path.join(temp_dir, "low_priority")
    os.makedirs(dir1)
    os.makedirs(dir2)

    # Add images to each directory
    img1_path = os.path.join(dir1, "image1.png")
    img2_path = os.path.join(dir2, "image2.png")
    Image.new('RGB', (500, 500)).save(img1_path)
    Image.new('RGB', (500, 500)).save(img2_path)

    # Manager with directory weights
    manager = BackgroundImageManager(
        background_dirs=[dir1, dir2],
        pattern="*.png",
        weights={dir1: 10.0, dir2: 1.0}
    )

    # Check weights
    assert manager.get_directory_weight(img1_path) == 10.0
    assert manager.get_directory_weight(img2_path) == 1.0

    # Selection should favor high-weight directory
    selections = []
    for _ in range(30):
        selected = manager.select_background()
        selections.append(selected)

    img1_count = selections.count(img1_path)
    total_count = len(selections)

    # High weight image should be selected more often (expect >60%)
    assert img1_count / total_count > 0.6


def test_minimum_score_floor(sample_backgrounds):
    """Test that scores don't go below minimum threshold."""
    manager = BackgroundImageManager(
        background_dirs=[os.path.dirname(sample_backgrounds['small'])],
        pattern="*.png"
    )

    # Apply massive penalty
    manager.update_score(sample_backgrounds['small'], 100.0)

    # Score should be clamped to 0.01 (minimum)
    assert manager.backgrounds[sample_backgrounds['small']] == 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
