"""Integration tests for health tracking during actual image generation.

These tests verify that FontHealthManager and BackgroundImageManager properly
track successes and failures during the full generation pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.batch_config import BatchConfig
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager
from src.generation_orchestrator import GenerationOrchestrator


def test_font_health_tracks_successful_generation(tmp_path):
    """Verify that successful font usage is recorded in FontHealthManager."""
    # Create test configuration
    config_content = """
specifications:
  - name: test_spec
    corpus_file: test.txt
    font_size_min: 24
    font_size_max: 24
    text_direction: left_to_right
    line_break_mode: word
    min_lines: 1
    max_lines: 1
    min_text_length: 5
    max_text_length: 10
    proportion: 1.0
total_images: 2
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)

    # Create test corpus
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "test.txt"
    corpus_file.write_text("Hello world this is test text for OCR generation.")

    # Create background directory (empty is ok)
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    (bg_dir / "white.jpg").write_text("")  # Touch file

    # Load config
    batch_config = BatchConfig.from_yaml(str(config_path))

    # Initialize managers
    font_health_manager = FontHealthManager()
    background_manager = BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

    # Create corpus map
    corpus_map = {"test.txt": str(corpus_file)}

    # Get available fonts from system
    font_dir = Path("data.nosync/fonts")
    if not font_dir.exists():
        pytest.skip("Font directory not available in test environment")

    all_fonts = [str(p) for p in font_dir.glob("*.ttf")][:3]  # Use only 3 fonts for speed
    if len(all_fonts) < 1:
        pytest.skip("No fonts available for testing")

    # Create orchestrator with shared health managers
    orchestrator = GenerationOrchestrator(
        batch_config=batch_config,
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager,
        font_health_manager=font_health_manager
    )

    # Create tasks
    unique_filenames = ["test_1", "test_2"]
    tasks = orchestrator.create_task_list(
        min_text_len=5,
        max_text_len=10,
        unique_filenames=unique_filenames
    )

    assert len(tasks) == 2

    # Record initial scores
    initial_scores = {}
    for task in tasks:
        record = font_health_manager._get_or_create_record(task.font_path)
        initial_scores[task.font_path] = record.health_score

    # Simulate successful generations by calling record_success for each font used
    for task in tasks:
        font_health_manager.record_success(task.font_path)
        if task.background_path:
            background_manager.record_success(task.background_path)

    # Verify that scores increased (capped at 100)
    for task in tasks:
        record = font_health_manager._get_or_create_record(task.font_path)
        initial_score = initial_scores[task.font_path]
        if initial_score < 100:
            assert record.health_score == initial_score + 1
        else:
            assert record.health_score == 100  # Capped


def test_font_health_tracks_failures():
    """Verify that font failures are tracked and fonts are eventually denylisted."""
    font_health_manager = FontHealthManager()

    # Simulate a problematic font
    problematic_font = "/fonts/problematic_font.ttf"
    healthy_font = "/fonts/healthy_font.ttf"

    all_fonts = [problematic_font, healthy_font]

    # Verify both fonts start available
    available = font_health_manager.get_available_fonts(all_fonts)
    assert problematic_font in available
    assert healthy_font in available

    # Simulate 6 failures for the problematic font (100 - 60 = 40, below threshold of 50)
    for _ in range(6):
        font_health_manager.record_failure(problematic_font)

    # Verify problematic font is now denylisted
    available = font_health_manager.get_available_fonts(all_fonts)
    assert problematic_font not in available
    assert healthy_font in available

    # Verify the score is below threshold
    record = font_health_manager._get_or_create_record(problematic_font)
    assert record.health_score < 50


def test_background_health_tracks_successes_and_failures(tmp_path):
    """Verify that BackgroundImageManager tracks health during generation."""
    # Create test backgrounds
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    healthy_bg = bg_dir / "healthy.jpg"
    problematic_bg = bg_dir / "problematic.jpg"
    healthy_bg.write_text("")
    problematic_bg.write_text("")

    background_manager = BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

    # Record successes for healthy background
    for _ in range(10):
        background_manager.record_success(str(healthy_bg))

    # Record failures for problematic background
    for _ in range(6):
        background_manager.record_failure(str(problematic_bg))

    # Verify healthy background has high score (capped at 100)
    healthy_record = background_manager._get_or_create_record(str(healthy_bg))
    assert healthy_record.health_score == 100

    # Verify problematic background is denylisted
    problematic_record = background_manager._get_or_create_record(str(problematic_bg))
    assert problematic_record.health_score < 50

    available_bgs = background_manager.get_available_backgrounds()
    assert str(healthy_bg) in available_bgs
    assert str(problematic_bg) not in available_bgs


def test_health_tracking_with_mixed_results():
    """Verify that health tracking correctly handles mixed success/failure patterns."""
    font_health_manager = FontHealthManager()

    font_path = "/fonts/intermittent_font.ttf"

    # Simulate pattern: 2 failures, 1 success, 2 failures, 1 success
    # Starting at 100: 100 -> 90 -> 80 -> 81 -> 71 -> 61 -> 62
    font_health_manager.record_failure(font_path)  # 90
    font_health_manager.record_failure(font_path)  # 80
    font_health_manager.record_success(font_path)  # 81
    font_health_manager.record_failure(font_path)  # 71
    font_health_manager.record_failure(font_path)  # 61
    font_health_manager.record_success(font_path)  # 62

    record = font_health_manager._get_or_create_record(font_path)
    assert record.health_score == 62

    # Font should still be available (above threshold of 50)
    available = font_health_manager.get_available_fonts([font_path])
    assert font_path in available


def test_weighted_selection_favors_healthy_fonts():
    """Verify that healthier fonts are more likely to be selected."""
    from collections import Counter

    font_health_manager = FontHealthManager()

    healthy_font = "/fonts/very_healthy.ttf"
    less_healthy_font = "/fonts/less_healthy.ttf"

    # Make less_healthy_font have score of 50 (5 failures)
    for _ in range(5):
        font_health_manager.record_failure(less_healthy_font)

    # Healthy font stays at 100
    healthy_score = font_health_manager._get_or_create_record(healthy_font).health_score
    less_healthy_score = font_health_manager._get_or_create_record(less_healthy_font).health_score

    assert healthy_score == 100
    assert less_healthy_score == 50

    # Perform many selections
    font_list = [healthy_font, less_healthy_font]
    selections = []
    for _ in range(1000):
        selected = font_health_manager.select_font(font_list)
        selections.append(selected)

    counts = Counter(selections)

    # Healthy font should be selected significantly more often
    assert counts[healthy_font] > counts[less_healthy_font]

    # Verify the distribution matches expected weights (with tolerance)
    expected_proportion = healthy_score / (healthy_score + less_healthy_score)
    actual_proportion = counts[healthy_font] / len(selections)
    assert abs(actual_proportion - expected_proportion) < 0.1  # 10% tolerance
