"""Tests for multi-line text generation in the OCRDataGenerator."""

import pytest
import random
from PIL import Image
from src.generator import OCRDataGenerator
from src.batch_config import BatchSpecification


class TestMultilineGeneration:
    """Tests for multi-line text generation."""

    @pytest.fixture
    def generator(self):
        """Create a generator instance."""
        return OCRDataGenerator()

    @pytest.fixture
    def single_line_spec(self):
        """Create a single-line batch specification."""
        return BatchSpecification(
            name="test_single",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=10,
            max_text_length=20,
            min_lines=1,
            max_lines=1,
            font_size_min=32,
            font_size_max=32
        )

    @pytest.fixture
    def multiline_spec(self):
        """Create a multi-line batch specification."""
        return BatchSpecification(
            name="test_multi",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=50,
            max_text_length=100,
            min_lines=2,
            max_lines=5,
            line_break_mode="word",
            line_spacing_min=1.0,
            line_spacing_max=1.5,
            text_alignment="left",
            font_size_min=32,
            font_size_max=32
        )

    @pytest.fixture
    def font_path(self):
        """Get a standard font path."""
        try:
            # Try to use a common system font
            path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            with open(path, 'rb'):
                return path
        except (OSError, FileNotFoundError):
            pytest.skip("Required font not available")

    def test_single_line_plan_backward_compatible(self, generator, single_line_spec, font_path):
        """Test that single-line mode generates backward-compatible plans."""
        random.seed(42)
        text = "Hello World Test"
        plan = generator.plan_generation(single_line_spec, text, font_path)

        assert plan["text"] == text
        assert plan["num_lines"] == 1
        assert plan["lines"] == [text]
        assert "line_spacing" in plan
        assert "line_break_mode" in plan
        assert "text_alignment" in plan

    def test_multiline_plan_creates_lines(self, generator, multiline_spec, font_path):
        """Test that multi-line spec generates plan with multiple lines."""
        random.seed(42)
        text = "This is a longer text that should be broken into multiple lines for testing"
        plan = generator.plan_generation(multiline_spec, text, font_path)

        assert plan["text"] == text
        assert plan["num_lines"] >= 2
        assert plan["num_lines"] <= 5
        assert len(plan["lines"]) == plan["num_lines"]
        assert "".join(plan["lines"]).replace(" ", "") == text.replace(" ", "")  # Same chars
        assert plan["line_break_mode"] == "word"
        assert plan["text_alignment"] == "left"
        assert 1.0 <= plan["line_spacing"] <= 1.5

    def test_single_line_generate_from_plan(self, generator, single_line_spec, font_path):
        """Test that single-line generation works correctly."""
        random.seed(42)
        text = "Hello World"
        plan = generator.plan_generation(single_line_spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        assert isinstance(image, Image.Image)
        assert len(bboxes) == len(text)
        # Single-line mode should include line_index: 0 for all characters
        assert all("line_index" in bbox for bbox in bboxes)
        assert all(bbox["line_index"] == 0 for bbox in bboxes)
        # All bboxes should have required fields
        for bbox in bboxes:
            assert "char" in bbox
            assert "x0" in bbox
            assert "y0" in bbox
            assert "x1" in bbox
            assert "y1" in bbox

    def test_multiline_generate_from_plan(self, generator, multiline_spec, font_path):
        """Test that multi-line generation works correctly."""
        random.seed(42)
        text = "Hello World This Is A Test Of Multi Line Generation"
        plan = generator.plan_generation(multiline_spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        assert isinstance(image, Image.Image)
        assert len(bboxes) > 0
        # Multi-line mode should include line_index
        assert all("line_index" in bbox for bbox in bboxes)
        # Check that line indices are valid
        line_indices = set(bbox["line_index"] for bbox in bboxes)
        assert len(line_indices) == plan["num_lines"]
        assert min(line_indices) == 0
        assert max(line_indices) == plan["num_lines"] - 1

        # All bboxes should have required fields
        for bbox in bboxes:
            assert "char" in bbox
            assert "x0" in bbox
            assert "y0" in bbox
            assert "x1" in bbox
            assert "y1" in bbox
            assert isinstance(bbox["line_index"], int)

    def test_multiline_character_break_mode(self, generator, font_path):
        """Test multi-line with character break mode."""
        spec = BatchSpecification(
            name="test_char_break",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=20,
            max_text_length=40,
            min_lines=3,
            max_lines=3,
            line_break_mode="character",
            line_spacing_min=1.0,
            line_spacing_max=1.0,
            text_alignment="center",
            font_size_min=32,
            font_size_max=32
        )

        random.seed(42)
        text = "HelloWorldTest"
        plan = generator.plan_generation(spec, text, font_path)

        assert plan["num_lines"] == 3
        assert plan["line_break_mode"] == "character"
        # Character mode should break anywhere
        assert "".join(plan["lines"]) == text

        image, bboxes = generator.generate_from_plan(plan)
        assert len(bboxes) == len(text)

    def test_multiline_vertical_text(self, generator, font_path):
        """Test multi-line with vertical text direction."""
        spec = BatchSpecification(
            name="test_vertical",
            proportion=1.0,
            text_direction="top_to_bottom",
            corpus_file="test.txt",
            min_text_length=20,
            max_text_length=40,
            min_lines=2,
            max_lines=3,
            line_break_mode="character",
            line_spacing_min=1.2,
            line_spacing_max=1.2,
            text_alignment="top",
            font_size_min=32,
            font_size_max=32
        )

        random.seed(42)
        text = "VerticalTest"
        plan = generator.plan_generation(spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        assert isinstance(image, Image.Image)
        assert len(bboxes) > 0
        assert all("line_index" in bbox for bbox in bboxes)

    def test_multiline_with_effects(self, generator, font_path):
        """Test that multi-line works with effects applied uniformly."""
        spec = BatchSpecification(
            name="test_effects",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=30,
            max_text_length=60,
            min_lines=2,
            max_lines=3,
            line_break_mode="word",
            blur_radius_min=0.5,
            blur_radius_max=0.5,
            noise_amount_min=0.1,
            noise_amount_max=0.1,
            font_size_min=32,
            font_size_max=32
        )

        random.seed(42)
        text = "Testing effects with multiple lines of text"
        plan = generator.plan_generation(spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        # Verify image was generated successfully with effects
        assert isinstance(image, Image.Image)
        assert image.width > 0
        assert image.height > 0
        assert len(bboxes) > 0

    def test_multiline_with_curves(self, generator, font_path):
        """Test that multi-line works with curved text (each line curved)."""
        spec = BatchSpecification(
            name="test_curves",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=30,
            max_text_length=60,
            min_lines=2,
            max_lines=2,
            line_break_mode="word",
            curve_type="arc",
            arc_radius_min=100.0,
            arc_radius_max=100.0,
            font_size_min=32,
            font_size_max=32
        )

        random.seed(42)
        text = "Testing curved text with multiple lines"
        plan = generator.plan_generation(spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        # Verify curved multi-line rendering works
        assert isinstance(image, Image.Image)
        assert len(bboxes) > 0
        assert all("line_index" in bbox for bbox in bboxes)

    def test_multiline_alignment_variations(self, generator, font_path):
        """Test different text alignments for multi-line text."""
        for alignment in ["left", "center", "right"]:
            spec = BatchSpecification(
                name=f"test_{alignment}",
                proportion=1.0,
                text_direction="left_to_right",
                corpus_file="test.txt",
                min_text_length=30,
                max_text_length=60,
                min_lines=3,
                max_lines=3,
                line_break_mode="word",
                text_alignment=alignment,
                font_size_min=32,
                font_size_max=32
            )

            random.seed(42)
            text = "Testing alignment with multiple lines of varying lengths"
            plan = generator.plan_generation(spec, text, font_path)

            assert plan["text_alignment"] == alignment

            image, bboxes = generator.generate_from_plan(plan)

            assert isinstance(image, Image.Image)
            assert len(bboxes) > 0

    def test_multiline_bboxes_ordering(self, generator, multiline_spec, font_path):
        """Test that bounding boxes maintain proper character order."""
        random.seed(42)
        text = "ABCDEFGHIJKLMNOP"
        plan = generator.plan_generation(multiline_spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        # Reconstruct text from bboxes
        reconstructed = "".join(bbox["char"] for bbox in bboxes)
        assert reconstructed == text

    def test_multiline_empty_lines_handled(self, generator, font_path):
        """Test that empty lines are handled gracefully."""
        spec = BatchSpecification(
            name="test_empty",
            proportion=1.0,
            text_direction="left_to_right",
            corpus_file="test.txt",
            min_text_length=5,
            max_text_length=10,
            min_lines=5,
            max_lines=5,
            line_break_mode="character",
            font_size_min=32,
            font_size_max=32
        )

        random.seed(42)
        text = "Short"  # Will create empty lines
        plan = generator.plan_generation(spec, text, font_path)

        image, bboxes = generator.generate_from_plan(plan)

        # Should handle empty lines without crashing
        assert isinstance(image, Image.Image)
        assert len(bboxes) == len(text)
