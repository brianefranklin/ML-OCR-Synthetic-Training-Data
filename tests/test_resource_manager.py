"""
Tests for the generic ResourceManager class.
"""

import pytest
from collections import Counter
from src.resource_manager import ResourceManager

# Constants from the ResourceManager implementation
STARTING_SCORE = 100
SUCCESS_SCORE_INCREASE = 1
FAILURE_SCORE_DECREASE = 10
HEALTH_THRESHOLD = 50

@pytest.fixture
def manager() -> ResourceManager:
    """Provides a fresh ResourceManager instance for each test."""
    return ResourceManager()

def test_resource_starts_with_default_score(manager: ResourceManager):
    """Tests that a new resource is initialized with the starting health score."""
    resource_id = "resource_a"
    record = manager._get_or_create_record(resource_id)
    assert record.health_score == STARTING_SCORE

def test_score_decreases_on_failure(manager: ResourceManager):
    """Tests that the health score decreases after a failure."""
    resource_id = "resource_b"
    manager.record_failure(resource_id)
    record = manager._get_or_create_record(resource_id)
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE

def test_score_increases_on_success(manager: ResourceManager):
    """Tests that the health score increases after a success."""
    resource_id = "resource_c"
    manager.record_failure(resource_id)  # Score is now 90
    manager.record_success(resource_id)  # Score is now 91
    record = manager._get_or_create_record(resource_id)
    assert record.health_score == STARTING_SCORE - FAILURE_SCORE_DECREASE + SUCCESS_SCORE_INCREASE

def test_score_is_capped_at_starting_score(manager: ResourceManager):
    """Tests that the health score does not exceed the starting score."""
    resource_id = "resource_d"
    manager.record_success(resource_id)
    record = manager._get_or_create_record(resource_id)
    assert record.health_score == STARTING_SCORE

def test_unhealthy_resources_are_filtered(manager: ResourceManager):
    """
    Tests that resources with a score below the threshold are not included
    in the list of available resources.
    """
    healthy_res = "healthy_one"
    unhealthy_res = "unhealthy_one"
    all_resources = [healthy_res, unhealthy_res]

    # 6 failures will bring the score from 100 to 40, which is below the threshold of 50
    for _ in range(6):
        manager.record_failure(unhealthy_res)

    assert manager._get_or_create_record(unhealthy_res).health_score < HEALTH_THRESHOLD
    
    available_resources = manager.get_available_resources(all_resources)
    
    assert healthy_res in available_resources
    assert unhealthy_res not in available_resources

def test_weighted_resource_selection(manager: ResourceManager):
    """
    Tests that resource selection is weighted based on health score,
    making healthier resources more likely to be chosen.
    """
    res_1 = "healthy_resource"
    res_2 = "less_healthy_resource"
    resource_list = [res_1, res_2]

    # Lower the score of the less_healthy_resource to 50
    for _ in range(5):
        manager.record_failure(res_2)

    score_1 = manager._get_or_create_record(res_1).health_score
    score_2 = manager._get_or_create_record(res_2).health_score

    assert score_1 == 100
    assert score_2 == 50

    # Perform many selections to check the distribution
    num_selections = 1000
    selections = [manager.select_resource(resource_list) for _ in range(num_selections)]
    counts = Counter(selections)

    # The healthier resource should be selected more often.
    assert counts[res_1] > counts[res_2]

    # Check if the distribution is within a reasonable statistical range.
    expected_proportion = score_1 / (score_1 + score_2)
    # Allow for variance for statistical fluctuations
    assert abs(counts[res_1] / num_selections - expected_proportion) < 0.25

def test_select_from_empty_list_raises_error(manager: ResourceManager):
    """Tests that calling select_resource with an empty list raises a ValueError."""
    with pytest.raises(ValueError):
        manager.select_resource([])
