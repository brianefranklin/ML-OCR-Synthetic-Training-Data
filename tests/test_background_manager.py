"""
Tests for the BackgroundImageManager's score-based system.
"""

import pytest
from collections import Counter
from src.background_manager import BackgroundImageManager

# Constants should mirror the implementation
STARTING_SCORE = 100
SUCCESS_SCORE_INCREASE = 1
FAILURE_SCORE_DECREASE = 10
HEALTH_THRESHOLD = 50

def test_background_starts_with_default_score():
    """Tests that a new background is initialized with the starting health score."""
    manager = BackgroundImageManager()
    bg_path = "/backgrounds/new.jpg"
    record = manager._get_or_create_record(bg_path)
    assert record.health_score == STARTING_SCORE

def test_score_decreases_on_failure():
    """Tests that the health score decreases after a failure."""
    manager = BackgroundImageManager()
    bg_path = "/backgrounds/failing.jpg"
    manager.record_failure(bg_path)
    record = manager._get_or_create_record(bg_path)
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE

def test_score_increases_on_success():
    """Tests that the health score increases after a success."""
    manager = BackgroundImageManager()
    bg_path = "/backgrounds/recovering.jpg"
    manager.record_failure(bg_path)
    manager.record_success(bg_path)
    record = manager._get_or_create_record(bg_path)
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE + SUCCESS_SCORE_INCREASE

def test_unhealthy_backgrounds_are_filtered():
    """
    Tests that backgrounds with a score below the threshold are not included
    in the list of available backgrounds.
    """
    manager = BackgroundImageManager()
    healthy_bg = "/bgs/healthy.jpg"
    unhealthy_bg = "/bgs/unhealthy.jpg"
    all_bgs = [healthy_bg, unhealthy_bg]
    # 6 failures will bring the score from 100 to 40
    for _ in range(6):
        manager.record_failure(unhealthy_bg)
    
    available_bgs = manager.get_available_backgrounds(all_bgs)
    
    assert healthy_bg in available_bgs
    assert unhealthy_bg not in available_bgs

def test_weighted_background_selection():
    """
    Tests that background selection is weighted based on health score.
    """
    manager = BackgroundImageManager()
    healthy_bg = "/bgs/healthy.jpg"
    less_healthy_bg = "/bgs/less_healthy.jpg"
    bg_list = [healthy_bg, less_healthy_bg]
    # Lower the score of one background to 50
    for _ in range(5):
        manager.record_failure(less_healthy_bg)
    
    healthy_score = manager._get_or_create_record(healthy_bg).health_score
    less_healthy_score = manager._get_or_create_record(less_healthy_bg).health_score

    num_selections = 1000
    selections = [manager.select_background(bg_list) for _ in range(num_selections)]
    counts = Counter(selections)

    assert counts[healthy_bg] > counts[less_healthy_bg]
    expected_proportion = healthy_score / (healthy_score + less_healthy_score)
    assert abs(counts[healthy_bg] / num_selections - expected_proportion) < 0.25
