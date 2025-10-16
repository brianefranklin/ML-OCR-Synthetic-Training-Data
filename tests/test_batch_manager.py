"""
Tests for the BatchManager class.
"""

import pytest
from collections import Counter
from src.batch_config import BatchConfig, BatchSpecification, BatchManager

def test_batch_manager_allocates_images_correctly():
    """
    Tests that the BatchManager correctly allocates the total number of images
    to the specifications based on their proportions.
    """
    spec1 = BatchSpecification(name="spec_a", proportion=0.6, text_direction="ltr", corpus_file="f1.txt")
    spec2 = BatchSpecification(name="spec_b", proportion=0.4, text_direction="rtl", corpus_file="f2.txt")
    config = BatchConfig(total_images=100, specifications=[spec1, spec2])

    manager = BatchManager(config)

    allocation = manager.get_allocation()

    assert allocation[spec1.name] == 60
    assert allocation[spec2.name] == 40

def test_uneven_allocation_is_handled():
    """
    Tests that allocation handles rounding correctly and the total sum matches
    the total number of images requested.
    """
    spec1 = BatchSpecification(name="spec_a", proportion=0.333, text_direction="ltr", corpus_file="f1.txt")
    spec2 = BatchSpecification(name="spec_b", proportion=0.333, text_direction="ltr", corpus_file="f2.txt")
    spec3 = BatchSpecification(name="spec_c", proportion=0.333, text_direction="ltr", corpus_file="f3.txt")
    config = BatchConfig(total_images=10, specifications=[spec1, spec2, spec3])

    manager = BatchManager(config)
    allocation = manager.get_allocation()

    assert sum(allocation.values()) == 10
    assert sorted(list(allocation.values())) == [3, 3, 4]

def test_task_list_is_generated_correctly():
    """
    Tests that the BatchManager can generate a full, interleaved list of
    which specification to use for each image.
    """
    spec1 = BatchSpecification(name="spec_a", proportion=0.5, text_direction="ltr", corpus_file="f1.txt")
    spec2 = BatchSpecification(name="spec_b", proportion=0.3, text_direction="rtl", corpus_file="f2.txt")
    spec3 = BatchSpecification(name="spec_c", proportion=0.2, text_direction="ltr", corpus_file="f3.txt")
    config = BatchConfig(total_images=10, specifications=[spec1, spec2, spec3])

    manager = BatchManager(config)

    # This method doesn't exist yet.
    tasks = manager.task_list()

    # 1. Check total number of tasks
    assert len(tasks) == 10

    # 2. Check the counts of each spec
    counts = Counter(spec.name for spec in tasks)
    assert counts["spec_a"] == 5
    assert counts["spec_b"] == 3
    assert counts["spec_c"] == 2

    # 3. Check for interleaved order
    assert tasks[0].name == "spec_a"
    assert tasks[1].name == "spec_b"
    assert tasks[2].name == "spec_c"
    assert tasks[3].name == "spec_a"
    assert tasks[4].name == "spec_b"
    assert tasks[5].name == "spec_c"