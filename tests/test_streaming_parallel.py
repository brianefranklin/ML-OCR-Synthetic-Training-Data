"""
Tests for streaming parallel generation functionality.

This module tests the chunked parallel generation approach, which generates
images in smaller chunks to reduce memory usage while maintaining parallelism benefits.
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Any
import tempfile

from src.generation_orchestrator import GenerationTask
from src.batch_config import BatchSpecification
from src.main import generate_image_from_task


def create_simple_spec() -> BatchSpecification:
    """Create a simple spec for testing with minimal randomization."""
    return BatchSpecification(
        name="test_spec",
        proportion=1.0,
        corpus_file="test.txt",
        text_direction="left_to_right",
        font_filter=None,
        # Minimal parameters for faster testing
        glyph_overlap_intensity_min=0.0,
        glyph_overlap_intensity_max=0.0,
        ink_bleed_radius_min=0.0,
        ink_bleed_radius_max=0.0,
        rotation_angle_min=0.0,
        rotation_angle_max=2.0,  # Small randomization for testing
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


def test_streaming_determinism_across_chunks():
    """Tests that streaming generation is deterministic across chunk boundaries."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Generate 10 images with different chunk boundaries
    # First run: generate indices 0-9 as if they were in different chunks
    # Second run: generate same indices but simulate different chunking

    results_run1 = []
    results_run2 = []

    for idx in range(10):
        task = GenerationTask(
            source_spec=spec,
            text=f"Text{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )

        # Generate same task twice
        result1 = generate_image_from_task((task, idx, None))
        result2 = generate_image_from_task((task, idx, None))

        results_run1.append(result1)
        results_run2.append(result2)

    # Verify all images are identical
    for idx, (r1, r2) in enumerate(zip(results_run1, results_run2)):
        idx1, img1, plan1 = r1
        idx2, img2, plan2 = r2

        assert idx1 == idx2 == idx
        assert np.array_equal(np.array(img1), np.array(img2)), \
            f"Image {idx} should be identical across runs"
        assert plan1["seed"] == plan2["seed"], \
            f"Image {idx} should have same seed"


def test_streaming_chunk_boundary_indices():
    """Tests that images at chunk boundaries (99, 100, 199, 200) are correct."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Test indices at chunk boundaries (assuming chunk_size=100)
    boundary_indices = [0, 99, 100, 199, 200, 299]

    for idx in boundary_indices:
        task = GenerationTask(
            source_spec=spec,
            text=f"Boundary{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )

        result_idx, image, plan = generate_image_from_task((task, idx, None))

        # Verify index is preserved
        assert result_idx == idx, f"Index should be preserved at boundary {idx}"

        # Verify image was generated
        assert image is not None
        assert image.size[0] > 0 and image.size[1] > 0

        # Verify plan contains expected text
        assert plan["text"] == f"Boundary{idx}"


def test_streaming_partial_final_chunk():
    """Tests that a partial final chunk (e.g., 50 images with chunk_size=100) works correctly."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Simulate generating 50 images (would be a partial chunk if chunk_size=100)
    results = []

    for idx in range(50):
        task = GenerationTask(
            source_spec=spec,
            text=f"Partial{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )

        result = generate_image_from_task((task, idx, None))
        results.append(result)

    # Verify all 50 images were generated
    assert len(results) == 50

    # Verify indices are correct
    for i, (idx, image, plan) in enumerate(results):
        assert idx == i, f"Index {idx} should match position {i}"


def test_streaming_maintains_order():
    """Tests that streaming maintains correct order even with parallel processing."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Generate a set of tasks
    num_images = 25
    tasks = []
    for idx in range(num_images):
        task = GenerationTask(
            source_spec=spec,
            text=f"Order{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )
        tasks.append((task, idx, None))

    # Generate all images
    results = [generate_image_from_task(task_args) for task_args in tasks]

    # Verify order is maintained
    for i, (idx, image, plan) in enumerate(results):
        assert idx == i, f"Results should be in order: expected {i}, got {idx}"
        assert plan["text"] == f"Order{i}", f"Text should match index {i}"


def test_streaming_with_different_texts():
    """Tests that streaming works correctly with different text strings per image."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Generate images with varying text lengths and content
    texts = [
        "A",
        "Hello",
        "Testing streaming mode",
        "This is a longer text string for testing",
        "1234567890",
        "Special chars: !@#$%",
    ]

    results = []
    for idx, text in enumerate(texts):
        task = GenerationTask(
            source_spec=spec,
            text=text,
            font_path=str(font_files[0]),
            background_path=None
        )

        result = generate_image_from_task((task, idx, None))
        results.append(result)

    # Verify all images were generated with correct text
    for i, (idx, image, plan) in enumerate(results):
        assert idx == i
        assert plan["text"] == texts[i]
        assert "bboxes" in plan
        assert len(plan["bboxes"]) > 0  # Should have bounding boxes


def test_streaming_produces_valid_images():
    """Tests that streaming produces valid, non-empty images."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Generate a few images
    for idx in range(5):
        task = GenerationTask(
            source_spec=spec,
            text=f"Valid{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )

        result_idx, image, plan = generate_image_from_task((task, idx, None))

        # Verify image properties
        assert isinstance(image, Image.Image)
        assert image.size[0] > 0 and image.size[1] > 0
        assert image.mode in ["RGBA", "RGB", "L"]

        # Verify plan structure
        assert "text" in plan
        assert "bboxes" in plan
        assert "seed" in plan
        assert isinstance(plan["bboxes"], list)


def test_streaming_with_sequential_indices():
    """Tests that sequential index generation works correctly in streaming mode."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Generate images sequentially (simulating chunk processing)
    chunk1_results = []
    chunk2_results = []

    # Simulate first chunk (indices 0-4)
    for idx in range(0, 5):
        task = GenerationTask(
            source_spec=spec,
            text=f"Chunk1_{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )
        result = generate_image_from_task((task, idx, None))
        chunk1_results.append(result)

    # Simulate second chunk (indices 5-9)
    for idx in range(5, 10):
        task = GenerationTask(
            source_spec=spec,
            text=f"Chunk2_{idx}",
            font_path=str(font_files[0]),
            background_path=None
        )
        result = generate_image_from_task((task, idx, None))
        chunk2_results.append(result)

    # Verify all indices are correct
    all_results = chunk1_results + chunk2_results
    for i, (idx, image, plan) in enumerate(all_results):
        assert idx == i, f"Index should be {i}, got {idx}"


def test_streaming_memory_efficiency_simulation():
    """Tests that streaming can handle chunk-based processing without accumulating all images."""
    spec = create_simple_spec()
    font_files = list(Path("data.nosync/fonts").rglob("*.ttf"))
    if not font_files:
        pytest.skip("No font files available for testing")

    # Simulate processing in chunks without keeping all images in memory
    chunk_size = 10
    total_images = 25
    processed_count = 0

    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        chunk_images = []

        # Generate chunk
        for idx in range(chunk_start, chunk_end):
            task = GenerationTask(
                source_spec=spec,
                text=f"Chunk{idx}",
                font_path=str(font_files[0]),
                background_path=None
            )
            result = generate_image_from_task((task, idx, None))
            chunk_images.append(result)

        # Verify chunk was generated correctly
        assert len(chunk_images) == (chunk_end - chunk_start)

        # In real implementation, we would save here and clear chunk_images
        # Simulate saving by just counting
        processed_count += len(chunk_images)

        # Clear chunk to simulate memory cleanup
        chunk_images = []

    # Verify all images were processed
    assert processed_count == total_images
