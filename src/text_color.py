"""
Text Color Module for OCR Synthetic Data Generation

Provides glyph-level color support with multiple modes and palettes.
Supports uniform color (all chars same) and per-glyph color (each char different).
"""

import random
from typing import List, Tuple, Optional, Union


# Color Palettes
REALISTIC_DARK = [
    (0, 0, 0),          # Black
    (25, 25, 112),      # Midnight Blue
    (139, 69, 19),      # Saddle Brown
    (47, 79, 79),       # Dark Slate Gray
    (0, 0, 128),        # Navy
    (85, 107, 47),      # Dark Olive Green
]

REALISTIC_LIGHT = [
    (255, 255, 255),    # White
    (245, 245, 220),    # Beige
    (240, 248, 255),    # Alice Blue
    (255, 250, 240),    # Floral White
    (250, 250, 210),    # Light Goldenrod Yellow
]

VIBRANT = [
    (255, 0, 0),        # Red
    (0, 255, 0),        # Green
    (0, 0, 255),        # Blue
    (255, 165, 0),      # Orange
    (255, 0, 255),      # Magenta
    (0, 255, 255),      # Cyan
    (255, 255, 0),      # Yellow
    (128, 0, 128),      # Purple
]

PASTELS = [
    (255, 182, 193),    # Light Pink
    (173, 216, 230),    # Light Blue
    (221, 160, 221),    # Plum
    (255, 218, 185),    # Peach Puff
    (216, 191, 216),    # Thistle
    (152, 251, 152),    # Pale Green
    (255, 255, 224),    # Light Yellow
]

# Palette mapping
PALETTES = {
    'realistic_dark': REALISTIC_DARK,
    'realistic_light': REALISTIC_LIGHT,
    'vibrant': VIBRANT,
    'pastels': PASTELS,
}


class ColorRenderer:
    """
    Handles text color generation and management.

    Supports multiple color modes and palettes for realistic
    and artistic OCR training data.
    """

    @staticmethod
    def get_palette(palette_name: str) -> List[Tuple[int, int, int]]:
        """
        Get color palette by name.

        Args:
            palette_name: Name of palette ('realistic_dark', 'vibrant', etc.)

        Returns:
            List of RGB tuples
        """
        return PALETTES.get(palette_name, REALISTIC_DARK)

    @staticmethod
    def generate_line_color(color_mode: str = 'uniform',
                           color_palette: str = 'realistic_dark',
                           custom_colors: Optional[List[Tuple[int, int, int]]] = None) -> Tuple[int, int, int]:
        """
        Generate a single color for uniform mode (all chars same color).

        Args:
            color_mode: Color mode (ignored, for API consistency)
            color_palette: Palette name
            custom_colors: Optional custom RGB colors

        Returns:
            RGB tuple for text color
        """
        if custom_colors and len(custom_colors) > 0:
            return random.choice(custom_colors)

        palette = ColorRenderer.get_palette(color_palette)
        return random.choice(palette)

    @staticmethod
    def generate_line_colors(text: str,
                            color_mode: str = 'uniform',
                            color_palette: str = 'realistic_dark',
                            custom_colors: Optional[List[Tuple[int, int, int]]] = None) -> List[Tuple[int, int, int]]:
        """
        Generate colors for each character in text.

        Args:
            text: Text string
            color_mode: 'uniform', 'per_glyph', 'gradient', or 'random'
            color_palette: Palette name
            custom_colors: Optional custom RGB colors

        Returns:
            List of RGB tuples (one per character)
        """
        # Use custom colors if provided
        if custom_colors and len(custom_colors) > 0:
            palette = custom_colors
        else:
            palette = ColorRenderer.get_palette(color_palette)

        text_length = len(text)

        if color_mode == 'uniform':
            # All characters same color
            color = random.choice(palette)
            return [color] * text_length

        elif color_mode == 'per_glyph':
            # Each character different color
            return [random.choice(palette) for _ in range(text_length)]

        elif color_mode == 'gradient':
            # Gradient from first to second color in palette
            if len(palette) < 2:
                # Not enough colors for gradient, use uniform
                color = palette[0]
                return [color] * text_length

            start_color = palette[0]
            end_color = palette[1]

            colors = []
            for i in range(text_length):
                t = i / max(text_length - 1, 1)  # Avoid division by zero
                r = int(start_color[0] + t * (end_color[0] - start_color[0]))
                g = int(start_color[1] + t * (end_color[1] - start_color[1]))
                b = int(start_color[2] + t * (end_color[2] - start_color[2]))
                colors.append((r, g, b))

            return colors

        elif color_mode == 'random':
            # Random mode: randomly choose uniform or per_glyph
            mode = random.choice(['uniform', 'per_glyph'])
            return ColorRenderer.generate_line_colors(text, mode, color_palette, custom_colors)

        else:
            # Invalid mode, fallback to uniform
            color = random.choice(palette)
            return [color] * text_length

    @staticmethod
    def get_contrasting_color(text_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Get contrasting background color for given text color.

        Args:
            text_color: RGB tuple for text

        Returns:
            RGB tuple for contrasting background
        """
        # Calculate luminance
        r, g, b = text_color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0

        # If text is dark (luminance < 0.5), use light background
        if luminance < 0.5:
            return (255, 255, 255)  # White
        else:
            return (0, 0, 0)  # Black

    @staticmethod
    def parse_color_string(color_str: str) -> Optional[Tuple[int, int, int]]:
        """
        Parse color string like "255,0,0" to RGB tuple.

        Args:
            color_str: String in format "R,G,B"

        Returns:
            RGB tuple or None if invalid
        """
        try:
            parts = color_str.split(',')
            if len(parts) == 3:
                r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                # Validate range
                if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                    return (r, g, b)
        except (ValueError, AttributeError):
            pass
        return None


# Utility functions for easy integration
def get_text_colors(text: str,
                   text_color_mode: str = 'uniform',
                   color_palette: str = 'realistic_dark',
                   custom_colors: Optional[List[Tuple[int, int, int]]] = None) -> List[Tuple[int, int, int]]:
    """
    Convenience function to get text colors.

    Args:
        text: Text string
        text_color_mode: Color mode
        color_palette: Palette name
        custom_colors: Optional custom colors

    Returns:
        List of RGB tuples (one per character)
    """
    return ColorRenderer.generate_line_colors(text, text_color_mode, color_palette, custom_colors)


def get_background_color(text_color: Union[Tuple[int, int, int], str],
                         background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[int, int, int]:
    """
    Get background color (auto-contrast or custom).

    Args:
        text_color: RGB tuple or 'auto'
        background_color: RGB tuple or 'auto'

    Returns:
        RGB tuple for background
    """
    # If background is auto, calculate contrasting color
    if background_color == 'auto' or background_color is None:
        if isinstance(text_color, tuple):
            return ColorRenderer.get_contrasting_color(text_color)
        else:
            return (255, 255, 255)  # Default white

    # If background is tuple, return it
    if isinstance(background_color, tuple):
        return background_color

    # Try to parse as string
    parsed = ColorRenderer.parse_color_string(str(background_color))
    if parsed:
        return parsed

    # Default to white
    return (255, 255, 255)
