"""Tests for the text_layout module."""

import pytest
from PIL import ImageFont
from src.text_layout import (
    break_into_lines,
    calculate_multiline_dimensions,
    _break_by_words,
    _break_by_characters
)


class TestBreakIntoLines:
    """Tests for the break_into_lines function."""

    def test_single_line_returns_text_as_is(self):
        """Test that single line mode returns text unchanged."""
        text = "Hello world"
        result = break_into_lines(text, 50, 1, "word")
        assert result == [text]

    def test_character_mode_splits_evenly(self):
        """Test that character mode splits text evenly."""
        text = "HelloWorld"
        result = break_into_lines(text, 5, 2, "character")
        assert len(result) == 2
        assert "".join(result) == text

    def test_word_mode_respects_boundaries(self):
        """Test that word mode keeps words together."""
        text = "Hello world testing"
        result = break_into_lines(text, 10, 2, "word")
        assert len(result) == 2
        # Each line should contain whole words
        for line in result:
            if line:  # Skip empty lines
                words = line.split()
                reconstructed = " ".join(words)
                assert reconstructed == line

    def test_empty_text_returns_empty_line(self):
        """Test that empty text returns appropriate result."""
        result = break_into_lines("", 10, 2, "word")
        assert result == [""]

    def test_short_text_pads_with_empty_lines(self):
        """Test that text shorter than num_lines is padded."""
        text = "Hi"
        result = break_into_lines(text, 10, 5, "character")
        assert len(result) == 5
        assert result[:2] == ["H", "i"]
        assert all(line == "" for line in result[2:])

    def test_invalid_mode_raises_error(self):
        """Test that invalid break mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown break_mode"):
            break_into_lines("test", 10, 2, "invalid_mode")


class TestBreakByWords:
    """Tests for the _break_by_words function."""

    def test_single_word_per_line(self):
        """Test breaking with one word per line."""
        text = "one two three four"
        result = _break_by_words(text, 5, 4)
        assert len(result) == 4
        assert all(len(line.split()) <= 2 for line in result)

    def test_no_words_returns_text(self):
        """Test that text without spaces returns as single line."""
        text = "nospaceshere"
        result = _break_by_words(text, 5, 2)
        assert text in "".join(result)


class TestBreakByCharacters:
    """Tests for the _break_by_characters function."""

    def test_even_distribution(self):
        """Test that characters are distributed evenly."""
        text = "12345678"
        result = _break_by_characters(text, 2)
        assert len(result) == 2
        assert len(result[0]) == 4
        assert len(result[1]) == 4
        assert "".join(result) == text

    def test_uneven_distribution(self):
        """Test that remainder characters are distributed to first lines."""
        text = "1234567"  # 7 chars, 3 lines
        result = _break_by_characters(text, 3)
        assert len(result) == 3
        # First lines get extra characters
        assert len(result[0]) >= len(result[2])
        assert "".join(result) == text

    def test_single_line_returns_text(self):
        """Test single line mode returns text unchanged."""
        text = "test"
        result = _break_by_characters(text, 1)
        assert result == [text]


class TestCalculateMultilineDimensions:
    """Tests for the calculate_multiline_dimensions function."""

    @pytest.fixture
    def font(self):
        """Create a test font."""
        # Use a standard font available on most systems
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        except OSError:
            # Fallback to default font if DejaVu not available
            pytest.skip("Required font not available")

    def test_horizontal_text_dimensions(self, font):
        """Test dimension calculation for horizontal text."""
        lines = ["Hello", "World"]
        width, height = calculate_multiline_dimensions(
            lines, font, 1.0, "left_to_right", 0.0
        )
        assert width > 0
        assert height > 0
        # Height should be proportional to number of lines
        single_width, single_height = calculate_multiline_dimensions(
            ["Hello"], font, 1.0, "left_to_right", 0.0
        )
        assert height > single_height

    def test_vertical_text_dimensions(self, font):
        """Test dimension calculation for vertical text."""
        lines = ["Hello", "World"]
        width, height = calculate_multiline_dimensions(
            lines, font, 1.0, "top_to_bottom", 0.0
        )
        assert width > 0
        assert height > 0
        # Width should be proportional to number of lines for vertical text
        single_width, single_height = calculate_multiline_dimensions(
            ["Hello"], font, 1.0, "top_to_bottom", 0.0
        )
        assert width > single_width

    def test_empty_lines_return_zero(self, font):
        """Test that empty lines return zero dimensions."""
        width, height = calculate_multiline_dimensions(
            [], font, 1.0, "left_to_right", 0.0
        )
        assert width == 0
        assert height == 0

        width, height = calculate_multiline_dimensions(
            ["", ""], font, 1.0, "left_to_right", 0.0
        )
        assert width == 0
        assert height == 0

    def test_line_spacing_affects_height(self, font):
        """Test that line spacing affects dimensions."""
        lines = ["Hello", "World"]
        width1, height1 = calculate_multiline_dimensions(
            lines, font, 1.0, "left_to_right", 0.0
        )
        width2, height2 = calculate_multiline_dimensions(
            lines, font, 1.5, "left_to_right", 0.0
        )
        # With larger spacing, height should increase
        assert height2 > height1
        # Width should remain the same
        assert width1 == width2

    def test_glyph_overlap_affects_dimensions(self, font):
        """Test that glyph overlap affects dimensions."""
        lines = ["Hello"]
        width1, height1 = calculate_multiline_dimensions(
            lines, font, 1.0, "left_to_right", 0.0
        )
        width2, height2 = calculate_multiline_dimensions(
            lines, font, 1.0, "left_to_right", 0.5
        )
        # With overlap, width should decrease
        assert width2 < width1
