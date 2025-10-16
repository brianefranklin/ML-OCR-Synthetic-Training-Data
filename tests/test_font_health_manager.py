"""
Tests for the FontHealthManager's score-based system.
"""

import pytest
from collections import Counter
from src.font_health_manager import FontHealthManager

# Constants should mirror the implementation
STARTING_SCORE = 100
SUCCESS_SCORE_INCREASE = 1
FAILURE_SCORE_DECREASE = 10
HEALTH_THRESHOLD = 50

def test_font_starts_with_default_score():
    """Tests that a new font is initialized with the starting health score."""
    manager = FontHealthManager()
    font_path = "/fonts/new_font.ttf"
    record = manager._get_or_create_record(font_path)
    assert record.health_score == STARTING_SCORE

def test_score_decreases_on_failure():
    """Tests that the health score decreases after a failure."""
    manager = FontHealthManager()
    font_path = "/fonts/failing_font.ttf"
    
    manager.record_failure(font_path)
    
    record = manager._get_or_create_record(font_path)
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE

def test_score_increases_on_success():
    """Tests that the health score increases after a success."""
    manager = FontHealthManager()
    font_path = "/fonts/recovering_font.ttf"

    # Decrease the score first
    manager.record_failure(font_path)
    assert manager._get_or_create_record(font_path).health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE
    
    # Now record a success
    manager.record_success(font_path)
    assert manager._get_or_create_record(font_path).health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE + SUCCESS_SCORE_INCREASE

def test_score_is_capped_at_starting_score():
    """Tests that the health score does not exceed the starting score."""
    manager = FontHealthManager()
    font_path = "/fonts/healthy_font.ttf"
    
    manager.record_success(font_path)
    
    record = manager._get_or_create_record(font_path)
    assert record.health_score == STARTING_SCORE

def test_unhealthy_fonts_are_filtered():
    """
    Tests that fonts with a score below the threshold are not included
    in the list of available fonts.
    """
    manager = FontHealthManager()
    healthy_font = "/fonts/healthy.ttf"
    unhealthy_font = "/fonts/unhealthy.ttf"
    all_fonts = [healthy_font, unhealthy_font]

    # 6 failures will bring the score from 100 to 40, which is below the threshold of 50
    for _ in range(6):
        manager.record_failure(unhealthy_font)

    assert manager._get_or_create_record(unhealthy_font).health_score < HEALTH_THRESHOLD
    
    available_fonts = manager.get_available_fonts(all_fonts)
    
    assert healthy_font in available_fonts
    assert unhealthy_font not in available_fonts

def test_weighted_font_selection():
    """
    Tests that font selection is weighted based on health score,
    making healthier fonts more likely to be chosen.
    """
    manager = FontHealthManager()
    healthy_font = "/fonts/healthy.ttf"
    less_healthy_font = "/fonts/less_healthy.ttf"
    font_list = [healthy_font, less_healthy_font]

    # Lower the score of the less_healthy_font to 50
    for _ in range(5):
        manager.record_failure(less_healthy_font)

    healthy_score = manager._get_or_create_record(healthy_font).health_score
    less_healthy_score = manager._get_or_create_record(less_healthy_font).health_score

    assert healthy_score == 100
    assert less_healthy_score == 50

    # Perform many selections to check the distribution
    num_selections = 1000
    selections = []
    for _ in range(num_selections):
        selected = manager.select_font(font_list)
        selections.append(selected)

    counts = Counter(selections)

    assert healthy_font in counts
    assert less_healthy_font in counts

    # The healthier font should be selected more often.
    assert counts[healthy_font] > counts[less_healthy_font]

    # Check if the distribution is within a reasonable statistical range.
    expected_healthy_proportion = healthy_score / (healthy_score + less_healthy_score)
    # Allow for 25% variance for statistical fluctuations
    assert abs(counts[healthy_font] / num_selections - expected_healthy_proportion) < 0.25
