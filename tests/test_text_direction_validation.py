"""Tests for validating text direction correctness using truth data.

This module provides comprehensive tests to ensure that generated images have
characters arranged in the correct spatial orientation for each text direction:
- Left-to-right (LTR): Characters progress from left to right (x increases)
- Right-to-left (RTL): Characters progress from right to left (x decreases)
- Top-to-bottom (TTB): Characters progress from top to bottom (y increases)
- Bottom-to-top (BTT): Characters progress from bottom to top (y decreases)

The tests use bounding box coordinates from truth data to validate ordering.
"""

import pytest
import json
import random
from typing import List, Dict, Any
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification


def get_character_position(bbox: Dict[str, Any], direction: str) -> float:
    """Get the primary coordinate for a character based on text direction.

    Args:
        bbox: Bounding box dictionary with x0, y0, x1, y1 coordinates
        direction: Text direction ('left_to_right', 'right_to_left',
                   'top_to_bottom', 'bottom_to_top')

    Returns:
        The primary coordinate value for ordering validation
    """
    if direction == "left_to_right":
        # For LTR, use left edge (x0) - should increase
        return bbox["x0"]
    elif direction == "right_to_left":
        # For RTL, use right edge (x1) - should decrease
        return bbox["x1"]
    elif direction == "top_to_bottom":
        # For TTB, use top edge (y0) - should increase
        return bbox["y0"]
    elif direction == "bottom_to_top":
        # For BTT, use bottom edge (y1) - should decrease
        return bbox["y1"]
    else:
        raise ValueError(f"Unknown direction: {direction}")


def validate_character_ordering(
    text: str,
    bboxes: List[Dict[str, Any]],
    direction: str,
    line_index: int = None
) -> bool:
    """Validate that characters are spatially ordered correctly for the direction.

    Args:
        text: The text string
        bboxes: List of bounding boxes (must match text length)
        direction: Text direction
        line_index: Optional line index to filter by (for multi-line text)

    Returns:
        True if ordering is correct, False otherwise
    """
    # Filter by line if specified
    if line_index is not None:
        bboxes = [b for b in bboxes if b["line_index"] == line_index]

    # Must have at least 2 characters to validate ordering
    if len(bboxes) < 2:
        return True

    # Get positions for each character
    positions = [get_character_position(bbox, direction) for bbox in bboxes]

    # Check ordering based on direction
    if direction in ["left_to_right", "top_to_bottom"]:
        # Positions should increase (or stay same for overlapping glyphs)
        for i in range(len(positions) - 1):
            if positions[i+1] < positions[i]:
                return False
    elif direction in ["right_to_left", "bottom_to_top"]:
        # Positions should decrease (or stay same for overlapping glyphs)
        for i in range(len(positions) - 1):
            if positions[i+1] > positions[i]:
                return False

    return True


def validate_line_ordering(
    bboxes: List[Dict[str, Any]],
    direction: str,
    num_lines: int
) -> bool:
    """Validate that lines are spatially ordered correctly for the direction.

    Args:
        bboxes: List of all bounding boxes
        direction: Text direction
        num_lines: Number of lines in the text

    Returns:
        True if line ordering is correct, False otherwise
    """
    if num_lines <= 1:
        return True

    # Get representative position for each line (using first character)
    line_positions = []
    for line_idx in range(num_lines):
        line_bboxes = [b for b in bboxes if b["line_index"] == line_idx]
        if line_bboxes:
            # For LTR/RTL, lines progress top-to-bottom (y0 increases)
            # For TTB/BTT, lines progress left-to-right (x0 increases)
            if direction in ["left_to_right", "right_to_left"]:
                # Lines stack vertically, use top edge of first char
                line_positions.append(line_bboxes[0]["y0"])
            elif direction in ["top_to_bottom", "bottom_to_top"]:
                # Lines stack horizontally, use left edge of first char
                line_positions.append(line_bboxes[0]["x0"])

    # Validate line positions increase
    for i in range(len(line_positions) - 1):
        if line_positions[i+1] < line_positions[i]:
            return False

    return True


class TestLeftToRightValidation:
    """Test left-to-right text generation and validation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return OCRDataGenerator()

    @pytest.fixture
    def font_path(self):
        """Path to test font."""
        return "data.nosync/fonts/NotoSansAnatolianHieroglyphs-Regular.ttf"

    def test_ltr_single_line_character_order(self, generator, font_path):
        """Test that LTR single-line text has characters progressing left to right."""
        random.seed(42)

        # Create a simple single-line spec
        spec = BatchSpecification(
            name="test_ltr_single",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="ltr/gutenberg_11.txt",
            min_text_length=5,
            max_text_length=50,
            font_size_min=48,
            font_size_max=48,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        text = "Hello World"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        # Validate character ordering
        assert validate_character_ordering(
            text,
            bboxes,
            "left_to_right"
        ), "LTR single-line characters should progress left to right"

    def test_ltr_multi_line_character_order(self, generator, font_path):
        """Test that LTR multi-line text has characters progressing left to right on each line."""
        random.seed(42)

        # Create a multi-line spec
        spec = BatchSpecification(
            name="test_ltr_multi",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="ltr/gutenberg_11.txt",
            min_text_length=5,
            max_text_length=100,
            min_lines=3,
            max_lines=3,
            line_break_mode="word",
            line_spacing_min=1.5,
            line_spacing_max=1.5,
            text_alignment="left",
            font_size_min=36,
            font_size_max=36,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        text = "The quick brown fox jumps over the lazy dog"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        num_lines = plan["num_lines"]
        assert num_lines == 3, "Should have 3 lines"

        # Validate each line's character ordering
        for line_idx in range(num_lines):
            assert validate_character_ordering(
                text,
                bboxes,
                "left_to_right",
                line_index=line_idx
            ), f"LTR line {line_idx} characters should progress left to right"

        # Validate line ordering (lines stack top to bottom)
        assert validate_line_ordering(
            bboxes,
            "left_to_right",
            num_lines
        ), "LTR lines should stack top to bottom"


class TestRightToLeftValidation:
    """Test right-to-left text generation and validation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return OCRDataGenerator()

    @pytest.fixture
    def font_path(self):
        """Path to test font."""
        return "data.nosync/fonts/IBMPlexSansArabic-Bold.ttf"

    def test_rtl_single_line_character_order(self, generator, font_path):
        """Test that RTL single-line text has characters progressing right to left."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_rtl_single",
            proportion=1.0,
            text_direction="right_to_left",
            corpus_file="rtl/arabic.txt",
            min_text_length=5,
            max_text_length=50,
            font_size_min=48,
            font_size_max=48,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        # Arabic text
        text = "مرحبا بك"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        # Validate character ordering
        assert validate_character_ordering(
            text,
            bboxes,
            "right_to_left"
        ), "RTL single-line characters should progress right to left"

    def test_rtl_multi_line_character_order(self, generator, font_path):
        """Test that RTL multi-line text has characters progressing right to left on each line."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_rtl_multi",
            proportion=1.0,
            text_direction="right_to_left",
            corpus_file="rtl/arabic.txt",
            min_text_length=5,
            max_text_length=100,
            min_lines=3,
            max_lines=3,
            line_break_mode="character",
            line_spacing_min=1.5,
            line_spacing_max=1.5,
            text_alignment="right",
            font_size_min=36,
            font_size_max=36,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        # Arabic text
        text = "الحياة جميلة والعالم واسع والفرص كثيرة"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        num_lines = plan["num_lines"]
        assert num_lines == 3, "Should have 3 lines"

        # Validate each line's character ordering
        for line_idx in range(num_lines):
            assert validate_character_ordering(
                text,
                bboxes,
                "right_to_left",
                line_index=line_idx
            ), f"RTL line {line_idx} characters should progress right to left"

        # Validate line ordering (lines stack top to bottom)
        assert validate_line_ordering(
            bboxes,
            "right_to_left",
            num_lines
        ), "RTL lines should stack top to bottom"


class TestTopToBottomValidation:
    """Test top-to-bottom text generation and validation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return OCRDataGenerator()

    @pytest.fixture
    def font_path(self):
        """Path to test font."""
        return "data.nosync/fonts/IBMPlexSansJP-Bold.ttf"

    def test_ttb_single_line_character_order(self, generator, font_path):
        """Test that TTB single-line text has characters progressing top to bottom."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_ttb_single",
            proportion=1.0,
            text_direction="top_to_bottom",
            corpus_file="ttb/japanese.txt",
            min_text_length=5,
            max_text_length=50,
            font_size_min=48,
            font_size_max=48,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        # Japanese text
        text = "こんにちは世界"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        # Validate character ordering
        assert validate_character_ordering(
            text,
            bboxes,
            "top_to_bottom"
        ), "TTB single-line characters should progress top to bottom"

    def test_ttb_multi_line_character_order(self, generator, font_path):
        """Test that TTB multi-line text has characters progressing top to bottom on each line."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_ttb_multi",
            proportion=1.0,
            text_direction="top_to_bottom",
            corpus_file="ttb/japanese.txt",
            min_text_length=5,
            max_text_length=100,
            min_lines=3,
            max_lines=3,
            line_break_mode="character",
            line_spacing_min=1.5,
            line_spacing_max=1.5,
            text_alignment="top",
            font_size_min=36,
            font_size_max=36,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        text = "日本語の縦書きテストです"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        num_lines = plan["num_lines"]
        assert num_lines == 3, "Should have 3 lines"

        # Validate each line's character ordering
        for line_idx in range(num_lines):
            assert validate_character_ordering(
                text,
                bboxes,
                "top_to_bottom",
                line_index=line_idx
            ), f"TTB line {line_idx} characters should progress top to bottom"

        # Validate line ordering (lines stack left to right)
        assert validate_line_ordering(
            bboxes,
            "top_to_bottom",
            num_lines
        ), "TTB lines should stack left to right"


class TestBottomToTopValidation:
    """Test bottom-to-top text generation and validation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return OCRDataGenerator()

    @pytest.fixture
    def font_path(self):
        """Path to test font."""
        return "data.nosync/fonts/NotoSansAnatolianHieroglyphs-Regular.ttf"

    def test_btt_single_line_character_order(self, generator, font_path):
        """Test that BTT single-line text has characters progressing bottom to top."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_btt_single",
            proportion=1.0,
            text_direction="bottom_to_top",
            corpus_file="btt/english.txt",
            min_text_length=5,
            max_text_length=50,
            font_size_min=48,
            font_size_max=48,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        text = "Bottom to Top"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        # Validate character ordering
        assert validate_character_ordering(
            text,
            bboxes,
            "bottom_to_top"
        ), "BTT single-line characters should progress bottom to top"

    def test_btt_multi_line_character_order(self, generator, font_path):
        """Test that BTT multi-line text has characters progressing bottom to top on each line."""
        random.seed(42)

        spec = BatchSpecification(
            name="test_btt_multi",
            proportion=1.0,
            text_direction="bottom_to_top",
            corpus_file="btt/english.txt",
            min_text_length=5,
            max_text_length=100,
            min_lines=3,
            max_lines=3,
            line_break_mode="word",
            line_spacing_min=1.5,
            line_spacing_max=1.5,
            text_alignment="bottom",
            font_size_min=36,
            font_size_max=36,
            rotation_angle_min=0.0,
            rotation_angle_max=0.0,
            curve_type="none",
        )

        text = "Testing bottom-to-top text with multiple lines"
        plan = generator.plan_generation(spec, text, font_path)
        image, bboxes = generator.generate_from_plan(plan)

        num_lines = plan["num_lines"]
        assert num_lines == 3, "Should have 3 lines"

        # Validate each line's character ordering
        for line_idx in range(num_lines):
            assert validate_character_ordering(
                text,
                bboxes,
                "bottom_to_top",
                line_index=line_idx
            ), f"BTT line {line_idx} characters should progress bottom to top"

        # Validate line ordering (lines stack left to right)
        assert validate_line_ordering(
            bboxes,
            "bottom_to_top",
            num_lines
        ), "BTT lines should stack left to right"


class TestMixedDirectionValidation:
    """Test validation logic with edge cases and mixed scenarios."""

    def test_validate_character_ordering_function(self):
        """Test the validate_character_ordering function with synthetic data."""
        # LTR - should pass (x0 increases)
        ltr_bboxes = [
            {"char": "a", "line_index": 0, "x0": 10, "y0": 10, "x1": 20, "y1": 30},
            {"char": "b", "line_index": 0, "x0": 20, "y0": 10, "x1": 30, "y1": 30},
            {"char": "c", "line_index": 0, "x0": 30, "y0": 10, "x1": 40, "y1": 30},
        ]
        assert validate_character_ordering("abc", ltr_bboxes, "left_to_right")

        # LTR - should fail (x0 decreases)
        ltr_bboxes_bad = [
            {"char": "a", "line_index": 0, "x0": 30, "y0": 10, "x1": 40, "y1": 30},
            {"char": "b", "line_index": 0, "x0": 20, "y0": 10, "x1": 30, "y1": 30},
            {"char": "c", "line_index": 0, "x0": 10, "y0": 10, "x1": 20, "y1": 30},
        ]
        assert not validate_character_ordering("abc", ltr_bboxes_bad, "left_to_right")

        # RTL - should pass (x1 decreases)
        rtl_bboxes = [
            {"char": "a", "line_index": 0, "x0": 100, "y0": 10, "x1": 110, "y1": 30},
            {"char": "b", "line_index": 0, "x0": 90, "y0": 10, "x1": 100, "y1": 30},
            {"char": "c", "line_index": 0, "x0": 80, "y0": 10, "x1": 90, "y1": 30},
        ]
        assert validate_character_ordering("abc", rtl_bboxes, "right_to_left")

        # TTB - should pass (y0 increases)
        ttb_bboxes = [
            {"char": "a", "line_index": 0, "x0": 10, "y0": 10, "x1": 30, "y1": 20},
            {"char": "b", "line_index": 0, "x0": 10, "y0": 20, "x1": 30, "y1": 30},
            {"char": "c", "line_index": 0, "x0": 10, "y0": 30, "x1": 30, "y1": 40},
        ]
        assert validate_character_ordering("abc", ttb_bboxes, "top_to_bottom")

        # BTT - should pass (y1 decreases)
        btt_bboxes = [
            {"char": "a", "line_index": 0, "x0": 10, "y0": 90, "x1": 30, "y1": 100},
            {"char": "b", "line_index": 0, "x0": 10, "y0": 80, "x1": 30, "y1": 90},
            {"char": "c", "line_index": 0, "x0": 10, "y0": 70, "x1": 30, "y1": 80},
        ]
        assert validate_character_ordering("abc", btt_bboxes, "bottom_to_top")

    def test_validate_line_ordering_function(self):
        """Test the validate_line_ordering function with synthetic data."""
        # LTR multi-line - lines should stack top to bottom (y0 increases)
        ltr_bboxes = [
            {"char": "a", "line_index": 0, "x0": 10, "y0": 10, "x1": 20, "y1": 30},
            {"char": "b", "line_index": 0, "x0": 20, "y0": 10, "x1": 30, "y1": 30},
            {"char": "c", "line_index": 1, "x0": 10, "y0": 40, "x1": 20, "y1": 60},
            {"char": "d", "line_index": 1, "x0": 20, "y0": 40, "x1": 30, "y1": 60},
        ]
        assert validate_line_ordering(ltr_bboxes, "left_to_right", 2)

        # TTB multi-line - lines should stack left to right (x0 increases)
        ttb_bboxes = [
            {"char": "a", "line_index": 0, "x0": 10, "y0": 10, "x1": 30, "y1": 20},
            {"char": "b", "line_index": 0, "x0": 10, "y0": 20, "x1": 30, "y1": 30},
            {"char": "c", "line_index": 1, "x0": 40, "y0": 10, "x1": 60, "y1": 20},
            {"char": "d", "line_index": 1, "x0": 40, "y0": 20, "x1": 60, "y1": 30},
        ]
        assert validate_line_ordering(ttb_bboxes, "top_to_bottom", 2)
