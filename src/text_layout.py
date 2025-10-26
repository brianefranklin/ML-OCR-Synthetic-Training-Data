"""Text layout utilities for multi-line text generation.

This module provides functions for breaking text into lines and calculating
line positions for multi-line text rendering. All behaviors are configurable
and make no assumptions about text direction or writing system.
"""

from typing import List, Tuple
from PIL import ImageFont


def break_into_lines(
    text: str,
    max_chars_per_line: int,
    num_lines: int,
    break_mode: str = "word"
) -> List[str]:
    """Breaks text into multiple lines.

    Args:
        text: The text to break into lines.
        max_chars_per_line: Maximum characters per line (soft limit for word mode).
        num_lines: Desired number of lines to create.
        break_mode: Line breaking strategy - "word" (respects word boundaries)
                    or "character" (breaks at any character position).
                    This is configurable for all text directions.

    Returns:
        List of text strings, one per line.

    Examples:
        >>> break_into_lines("Hello world", 5, 2, "word")
        ['Hello', 'world']
        >>> break_into_lines("Hello", 2, 2, "character")
        ['He', 'llo']
    """
    if num_lines == 1:
        return [text]

    if len(text) == 0:
        return [""]

    # If text is shorter than desired lines, return character per line
    if len(text) <= num_lines:
        return list(text) + [""] * (num_lines - len(text))

    if break_mode == "word":
        return _break_by_words(text, max_chars_per_line, num_lines)
    elif break_mode == "character":
        return _break_by_characters(text, num_lines)
    else:
        raise ValueError(f"Unknown break_mode: {break_mode}. Must be 'word' or 'character'.")


def _break_by_words(text: str, max_chars_per_line: int, num_lines: int) -> List[str]:
    """Breaks text into lines respecting word boundaries.

    Works for any text direction. Word boundaries are determined by whitespace.

    Args:
        text: The text to break.
        max_chars_per_line: Maximum characters per line (soft limit).
        num_lines: Desired number of lines.

    Returns:
        List of text strings.
    """
    words = text.split()
    if not words:
        return [text]

    # Calculate target characters per line
    total_chars = len(text)
    target_chars_per_line = total_chars // num_lines

    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)
        # Space before word (except for first word in line)
        space_length = 1 if current_line else 0

        # Check if adding this word would exceed target (with some flexibility)
        would_exceed = (current_length + space_length + word_length > target_chars_per_line)
        not_last_line = len(lines) < num_lines - 1

        if would_exceed and current_line and not_last_line:
            # Start a new line
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            # Add word to current line
            current_line.append(word)
            current_length += space_length + word_length

    # Add remaining words to last line
    if current_line:
        lines.append(" ".join(current_line))

    # Pad with empty lines if necessary
    while len(lines) < num_lines:
        lines.append("")

    return lines[:num_lines]


def _break_by_characters(text: str, num_lines: int) -> List[str]:
    """Breaks text into lines by character count.

    Works for any text direction and writing system.

    Args:
        text: The text to break.
        num_lines: Number of lines to create.

    Returns:
        List of text strings.
    """
    if num_lines == 1:
        return [text]

    chars_per_line = len(text) // num_lines
    remainder = len(text) % num_lines

    lines = []
    start = 0

    for i in range(num_lines):
        # Distribute remainder characters across first lines
        line_length = chars_per_line + (1 if i < remainder else 0)
        end = start + line_length
        lines.append(text[start:end])
        start = end

    return lines


def calculate_line_positions(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
    alignment: str,
    direction: str
) -> List[Tuple[int, int]]:
    """Calculates the position for each line of text.

    Args:
        lines: List of text strings (one per line).
        font: The PIL font to use for measurement.
        line_spacing: Line spacing multiplier (e.g., 1.2 for 1.2x line height).
        alignment: Text alignment - "left", "center", "right" for horizontal;
                   "top", "center", "bottom" for vertical.
        direction: Text direction - "left_to_right", "right_to_left",
                   "top_to_bottom", "bottom_to_top".

    Returns:
        List of (x, y) tuples representing the position of each line.

    Note:
        For horizontal text (LTR/RTL), returns relative x positions.
        For vertical text (TTB/BTT), returns relative y positions.
    """
    if direction in ["left_to_right", "right_to_left"]:
        return _calculate_horizontal_positions(lines, font, line_spacing, alignment)
    elif direction in ["top_to_bottom", "bottom_to_top"]:
        return _calculate_vertical_positions(lines, font, line_spacing, alignment)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _calculate_horizontal_positions(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
    alignment: str
) -> List[Tuple[int, int]]:
    """Calculates positions for horizontal text (LTR/RTL).

    Returns list of (x_offset, y_offset) for each line.
    """
    ascent, descent = font.getmetrics()
    line_height = int((ascent + descent) * line_spacing)

    # Measure all lines to find max width
    line_widths = []
    for line in lines:
        if line:
            try:
                bbox = font.getbbox(line)
                width = bbox[2] - bbox[0]
            except (AttributeError, OSError):
                # Fallback: estimate width
                width = len(line) * int(font.size * 0.6)
        else:
            width = 0
        line_widths.append(width)

    max_width = max(line_widths) if line_widths else 0

    # Calculate positions
    positions = []
    for i, width in enumerate(line_widths):
        y_pos = i * line_height

        if alignment == "left":
            x_pos = 0
        elif alignment == "center":
            x_pos = (max_width - width) // 2
        elif alignment == "right":
            x_pos = max_width - width
        else:
            # Default to left
            x_pos = 0

        positions.append((x_pos, y_pos))

    return positions


def _calculate_vertical_positions(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
    alignment: str
) -> List[Tuple[int, int]]:
    """Calculates positions for vertical text (TTB/BTT).

    Returns list of (x_offset, y_offset) for each line.
    For vertical text, lines are arranged horizontally (side by side).
    """
    ascent, descent = font.getmetrics()
    char_width = int(font.size * 0.6)  # Estimate character width
    line_offset = int(char_width * line_spacing * 2)  # Space between vertical lines

    # Measure all lines to find max height
    line_heights = []
    for line in lines:
        if line:
            line_height = len(line) * (ascent + descent)
        else:
            line_height = 0
        line_heights.append(line_height)

    max_height = max(line_heights) if line_heights else 0

    # Calculate positions
    positions = []
    for i, height in enumerate(line_heights):
        x_pos = i * line_offset

        if alignment == "top":
            y_pos = 0
        elif alignment == "center":
            y_pos = (max_height - height) // 2
        elif alignment == "bottom":
            y_pos = max_height - height
        else:
            # Default to top
            y_pos = 0

        positions.append((x_pos, y_pos))

    return positions


def calculate_multiline_dimensions(
    lines: List[str],
    font: ImageFont.FreeTypeFont,
    line_spacing: float,
    direction: str,
    glyph_overlap_intensity: float = 0.0
) -> Tuple[int, int]:
    """Calculates the total dimensions needed for multi-line text.

    Args:
        lines: List of text strings (one per line).
        font: The PIL font to use.
        line_spacing: Line spacing multiplier.
        direction: Text direction.
        glyph_overlap_intensity: Character overlap intensity (0.0-1.0).

    Returns:
        Tuple of (width, height) in pixels.
    """
    if not lines or all(not line for line in lines):
        return (0, 0)

    ascent, descent = font.getmetrics()
    char_height = ascent + descent

    if direction in ["left_to_right", "right_to_left"]:
        # Horizontal text: height grows with lines, width is max line width
        line_height = int(char_height * line_spacing)
        total_height = line_height * len(lines)

        max_width = 0
        for line in lines:
            if line:
                line_width = 0
                for char in line:
                    try:
                        bbox = font.getbbox(char)
                        char_width = bbox[2] - bbox[0]
                    except (AttributeError, OSError):
                        char_width = int(font.size * 0.6)
                    line_width += char_width * (1 - glyph_overlap_intensity)
                max_width = max(max_width, int(line_width))

        return (max_width, total_height)

    else:  # vertical text
        # Vertical text: width grows with lines, height is max line height
        char_width = int(font.size * 0.6)
        line_offset = int(char_width * line_spacing * 2)
        total_width = line_offset * len(lines)

        max_height = 0
        for line in lines:
            if line:
                line_height = len(line) * int(char_height * (1 - glyph_overlap_intensity))
                max_height = max(max_height, line_height)

        return (total_width, max_height)
