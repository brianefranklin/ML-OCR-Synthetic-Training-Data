"""
Tests for the BackgroundImageManager's score-based system.
"""

import pytest
from collections import Counter
from pathlib import Path
from src.background_manager import BackgroundImageManager

# These constants must match the values in the ResourceManager implementation
# to ensure the tests are validating against the correct thresholds and score changes.
STARTING_SCORE = 100
SUCCESS_SCORE_INCREASE = 1
FAILURE_SCORE_DECREASE = 10
HEALTH_THRESHOLD = 50

@pytest.fixture
def manager(tmp_path):
    """
    This fixture sets up a temporary directory structure with a few dummy image files.
    It then initializes a BackgroundImageManager pointing to this directory.
    This provides a consistent, isolated environment for each test function.
    """
    # Create a temporary directory for the backgrounds.
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    
    # Create dummy files to be discovered by the manager.
    (bg_dir / "healthy.jpg").touch()
    (bg_dir / "unhealthy.jpg").touch()
    (bg_dir / "less_healthy.jpg").touch()
    
    # Initialize the manager, telling it to look for images in the created directory.
    # The weight of 1.0 is arbitrary for this test.
    return BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

def test_background_starts_with_default_score(manager: BackgroundImageManager):
    """Intent: Verify that any new resource tracked by the manager starts with the default score."""
    # Get the path of the first discovered background image.
    bg_path = manager.background_paths[0]
    # Internally get or create the health record for this path.
    record = manager._get_or_create_record(bg_path)
    # Assert that its score is the default starting score.
    assert record.health_score == STARTING_SCORE

def test_score_decreases_on_failure(manager: BackgroundImageManager):
    """Intent: Verify that reporting a failure for a resource correctly decreases its score."""
    # Get the path of the first discovered background image.
    bg_path = manager.background_paths[0]
    # Report a failure for this background.
    manager.record_failure(bg_path)
    # Retrieve the updated health record.
    record = manager._get_or_create_record(bg_path)
    # Assert that the score has been reduced by the correct amount.
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE

def test_score_increases_on_success(manager: BackgroundImageManager):
    """Intent: Verify that reporting a success correctly increases a resource's score."""
    # Get the path of the first discovered background image.
    bg_path = manager.background_paths[0]
    # First, decrease the score by reporting a failure.
    manager.record_failure(bg_path)
    # Then, report a success.
    manager.record_success(bg_path)
    # Retrieve the updated record.
    record = manager._get_or_create_record(bg_path)
    # Assert that the score reflects both the decrease and the subsequent increase.
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE + SUCCESS_SCORE_INCREASE

def test_unhealthy_backgrounds_are_filtered(manager: BackgroundImageManager):
    """
    Intent: Verify that the manager correctly filters out resources whose scores have fallen
    below the health threshold.
    """
    # Find the specific paths for the healthy and unhealthy backgrounds.
    # This is a more robust way to find the files we need for the test.
    unhealthy_bg = None
    healthy_bg = None
    for path in manager.background_paths:
        if Path(path).name == 'unhealthy.jpg':
            unhealthy_bg = path
        elif Path(path).name == 'healthy.jpg':
            healthy_bg = path
    
    # Ensure we actually found the files we need for the test.
    assert unhealthy_bg is not None, "Test setup failed: Could not find unhealthy.jpg"
    assert healthy_bg is not None, "Test setup failed: Could not find healthy.jpg"

    # Report failures for the 'unhealthy' background enough times to drop its score below the threshold.
    # 6 failures * -10 points/failure = -60 points. 100 - 60 = 40, which is < 50.
    for _ in range(6):
        manager.record_failure(unhealthy_bg)
    
    # Get the set of backgrounds that the manager considers available.
    available_bgs = manager.get_available_backgrounds()
    
    # Assert that the healthy background is still available.
    assert healthy_bg in available_bgs
    # Assert that the unhealthy background has been correctly filtered out.
    assert unhealthy_bg not in available_bgs

def test_weighted_background_selection(manager: BackgroundImageManager):
    """
    Intent: Verify that the manager's random selection is weighted by the health scores,
    making healthier resources more likely to be chosen.
    """
    # Find the specific paths for the files involved in this test.
    less_healthy_bg = None
    healthy_bg = None
    unhealthy_bg = None # This one is just part of the denominator in the probability calculation.
    for path in manager.background_paths:
        if Path(path).name == 'less_healthy.jpg':
            less_healthy_bg = path
        elif Path(path).name == 'healthy.jpg':
            healthy_bg = path
        elif Path(path).name == 'unhealthy.jpg':
            unhealthy_bg = path

    # Ensure all necessary files were found.
    assert less_healthy_bg is not None
    assert healthy_bg is not None
    assert unhealthy_bg is not None

    # Report failures for the 'less_healthy' background to lower its score.
    # 5 failures * -10 points/failure = -50 points. Score becomes 50.
    for _ in range(5):
        manager.record_failure(less_healthy_bg)
    
    # Get the scores for the calculation.
    healthy_score = manager._get_or_create_record(healthy_bg).health_score
    less_healthy_score = manager._get_or_create_record(less_healthy_bg).health_score
    unhealthy_score = manager._get_or_create_record(unhealthy_bg).health_score

    # Perform a large number of selections to get a statistical sample.
    num_selections = 1000
    selections = [manager.select_background() for _ in range(num_selections)]
    counts = Counter(selections)

    # The core assertion: the healthier background should have been selected more often.
    assert counts[healthy_bg] > counts[less_healthy_bg]

    # A more detailed statistical check: does the proportion of selections match the
    # expected proportion based on the health scores? We allow for some statistical noise.
    total_score = healthy_score + less_healthy_score + unhealthy_score
    expected_proportion = healthy_score / total_score
    assert abs(counts[healthy_bg] / num_selections - expected_proportion) < 0.25