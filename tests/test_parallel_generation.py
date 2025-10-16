"""
Tests for parallel image generation functionality.

This module tests the multiprocessing-based parallel image generation,
ensuring that parallel generation produces identical results to sequential
generation and that the worker functions are properly designed for
multiprocessing.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Tuple
import multiprocessing

# Import will be available after implementation
from src.main import generate_image_from_task
from src.generation_orchestrator import GenerationTask
from src.batch_config import BatchSpecification
from src.generator import OCRDataGenerator


def test_worker_function_is_deterministic(tmp_path):
    """Tests that the worker function produces identical output for the same input."""
    # Create a simple batch specification
    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        # All parameters at zero/default for simplicity
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        rotation_angle_min=0.0,
        rotation_angle_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        noise_amount_min=0.0,
        noise_amount_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    # Create a test task
    # Find a font file for testing
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    task = GenerationTask(
        index=0,
        source_spec=spec,
        text="Hello World",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test"
    )

    # Generate the same image twice
    result1 = generate_image_from_task((task, 0, None))
    result2 = generate_image_from_task((task, 0, None))

    idx1, image1, plan1, error1 = result1
    idx2, image2, plan2, error2 = result2

    # No errors should occur
    assert error1 is None, f"First generation should not error: {error1}"
    assert error2 is None, f"Second generation should not error: {error2}"

    # Indices should match
    assert idx1 == idx2 == 0

    # Images should be identical (pixel-by-pixel)
    assert image1.size == image2.size
    assert image1.mode == image2.mode

    # Convert to numpy for comparison
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    assert np.array_equal(img1_array, img2_array), "Images should be pixel-perfect identical"

    # Plans should be identical (including seed)
    assert plan1["seed"] == plan2["seed"]
    assert plan1["text"] == plan2["text"]


def test_worker_function_with_different_seeds_produces_different_images(tmp_path):
    """Tests that different tasks produce different images."""
    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        # Add some randomization
        rotation_angle_min=-5.0,
        rotation_angle_max=5.0,
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        noise_amount_min=0.0,
        noise_amount_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    task1 = GenerationTask(
        index=0,
        source_spec=spec,
        text="Hello",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test1"
    )

    task2 = GenerationTask(
        index=1,
        source_spec=spec,
        text="World",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test2"
    )

    # Generate different images
    result1 = generate_image_from_task((task1, 0, None))
    result2 = generate_image_from_task((task2, 1, None))

    idx1, image1, plan1, error1 = result1
    idx2, image2, plan2, error2 = result2

    # No errors should occur
    assert error1 is None, f"First generation should not error: {error1}"
    assert error2 is None, f"Second generation should not error: {error2}"

    # Different indices
    assert idx1 == 0
    assert idx2 == 1

    # Different text
    assert plan1["text"] != plan2["text"]

    # Images should be different (different text means different pixels)
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    # Not pixel-perfect identical (very likely)
    # Just check they're not all the same
    assert not np.array_equal(img1_array, img2_array), "Different tasks should produce different images"


def test_parallel_vs_sequential_produces_same_results():
    """Tests that parallel generation produces the same results as sequential generation."""
    # Create a simple spec with some randomization
    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        rotation_angle_min=-2.0,
        rotation_angle_max=2.0,
        noise_amount_min=0.0,
        noise_amount_max=0.01,
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Create a few tasks
    tasks = []
    for i, text in enumerate(["Hello", "World", "Test", "Parallel"]):
        task = GenerationTask(
            index=i,
            source_spec=spec,
            text=text,
            font_path=str(font_files[0]),
            background_path=None,
            output_filename=f"test_{i}"
        )
        tasks.append((task, i, None))

    # Generate sequentially
    sequential_results = []
    for task_args in tasks:
        result = generate_image_from_task(task_args)
        sequential_results.append(result)

    # Generate in parallel
    with multiprocessing.Pool(processes=2) as pool:
        parallel_results = pool.map(generate_image_from_task, tasks)

    # Results should be identical
    assert len(sequential_results) == len(parallel_results)

    for seq_result, par_result in zip(sequential_results, parallel_results):
        seq_idx, seq_image, seq_plan, seq_error = seq_result
        par_idx, par_image, par_plan, par_error = par_result

        # No errors should occur
        assert seq_error is None, f"Sequential generation should not error: {seq_error}"
        assert par_error is None, f"Parallel generation should not error: {par_error}"

        # Same index
        assert seq_idx == par_idx

        # Same seed (deterministic)
        assert seq_plan["seed"] == par_plan["seed"]
        assert seq_plan["text"] == par_plan["text"]

        # Same image size
        assert seq_image.size == par_image.size

        # Same pixels
        seq_array = np.array(seq_image)
        par_array = np.array(par_image)
        assert np.array_equal(seq_array, par_array), \
            f"Parallel and sequential generation should produce identical results for task {seq_idx}"


def test_worker_function_handles_numpy_types():
    """Tests that the worker function properly handles NumPy types in the plan."""
    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        # Some randomization to generate numpy types
        rotation_angle_min=0.0,
        rotation_angle_max=5.0,
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.1,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        noise_amount_min=0.0,
        noise_amount_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    task = GenerationTask(
        index=0,
        source_spec=spec,
        text="Test",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test"
    )

    idx, image, plan, error = generate_image_from_task((task, 0, None))

    # No errors should occur
    assert error is None, f"Generation should not error: {error}"

    # Verify plan contains bboxes
    assert "bboxes" in plan
    assert len(plan["bboxes"]) > 0

    # Verify bbox coordinates are proper Python ints (not numpy types)
    for bbox in plan["bboxes"]:
        assert isinstance(bbox["x0"], int), f"x0 should be int, got {type(bbox['x0'])}"
        assert isinstance(bbox["y0"], int), f"y0 should be int, got {type(bbox['y0'])}"
        assert isinstance(bbox["x1"], int), f"x1 should be int, got {type(bbox['x1'])}"
        assert isinstance(bbox["y1"], int), f"y1 should be int, got {type(bbox['y1'])}"


def test_worker_function_preserves_task_index():
    """Tests that the worker function correctly returns the task index."""
    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        rotation_angle_min=0.0,
        rotation_angle_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        noise_amount_min=0.0,
        noise_amount_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    task = GenerationTask(
        index=0, # The task's internal index can be static for this test
        source_spec=spec,
        text="Test",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test"
    )

    # Test with different indices
    for expected_idx in [0, 5, 42, 999]:
        idx, image, plan, error = generate_image_from_task((task, expected_idx, None))
        assert error is None, f"Generation should not error: {error}"
        assert idx == expected_idx, f"Worker should preserve task index {expected_idx}"


def test_worker_function_with_background_manager():
    """Tests that the worker function properly handles background manager."""
    from src.background_manager import BackgroundImageManager

    # Create a background manager (will use test backgrounds if available)
    background_dir = Path("data.nosync/backgrounds")
    if not background_dir.exists():
        pytest.skip("No background directory available for testing")

    background_manager = BackgroundImageManager(dir_weights={str(background_dir): 1.0})

    spec = BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        rotation_angle_min=0.0,
        rotation_angle_max=0.0,
        perspective_warp_magnitude_min=0.0,
        perspective_warp_magnitude_max=0.0,
        noise_amount_min=0.0,
        noise_amount_max=0.0,
        blur_radius_min=0.0,
        blur_radius_max=0.0,
        brightness_factor_min=1.0,
        brightness_factor_max=1.0,
        contrast_factor_min=1.0,
        contrast_factor_max=1.0,
        elastic_distortion_alpha_min=0.0,
        elastic_distortion_alpha_max=0.0,
        elastic_distortion_sigma_min=0.0,
        elastic_distortion_sigma_max=0.0,
        grid_distortion_steps_min=1,
        grid_distortion_steps_max=1,
        grid_distortion_limit_min=0,
        grid_distortion_limit_max=0,
        optical_distortion_limit_min=0.0,
        optical_distortion_limit_max=0.0,
        erosion_dilation_kernel_min=1,
        erosion_dilation_kernel_max=1,
        cutout_width_min=0,
        cutout_width_max=0,
        cutout_height_min=0,
        cutout_height_max=0,
        curve_type="none",
        arc_radius_min=0.0,
        arc_radius_max=0.0,
        arc_concave=True,
        sine_amplitude_min=0.0,
        sine_amplitude_max=0.0,
        sine_frequency_min=0.0,
        sine_frequency_max=0.0,
        sine_phase_min=0.0,
        sine_phase_max=0.0,
    )

    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    task = GenerationTask(
        index=0,
        source_spec=spec,
        text="Test",
        font_path=str(font_files[0]),
        background_path=None,
        output_filename="test"
    )

    # Generate with background manager
    idx, image, plan, error = generate_image_from_task((task, 0, background_manager))

    # No errors should occur
    assert error is None, f"Generation should not error: {error}"

    # Should have a background path in the plan
    assert "background_path" in plan
    # Background path might be None or a string depending on availability
    if plan["background_path"] is not None:
        assert isinstance(plan["background_path"], str)
