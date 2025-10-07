"""
Font utility functions for OCR data generation.

Provides font validation, loading, and character extraction utilities.
"""

import os
import logging
import functools
from PIL import ImageFont


# Global font health manager (set by main.py)
_font_health_manager = None


def set_font_health_manager(manager):
    """Set the global font health manager instance."""
    global _font_health_manager
    _font_health_manager = manager


def extract_sample_characters(text: str, max_samples: int = 100) -> str:
    """
    Extract a sample of unique characters from the text corpus.

    Args:
        text: Text corpus to sample from
        max_samples: Maximum number of unique characters to extract

    Returns:
        String containing unique sample characters
    """
    # Handle None or empty input
    if not text:
        return ""

    # Get unique characters, preserving order
    seen = set()
    unique_chars = []
    for char in text:
        if char not in seen:
            seen.add(char)
            unique_chars.append(char)
            if len(unique_chars) >= max_samples:
                break

    return ''.join(unique_chars)


@functools.lru_cache(maxsize=None)
def can_font_render_text(font_path, text, character_set):
    """
    Check if a font can render the given text.

    This function also integrates with the font health manager to:
    - Skip fonts in cooldown
    - Skip fonts with health below threshold
    - Track character coverage
    - Record validation failures

    Args:
        font_path: Path to font file
        text: Text to validate
        character_set: Set of characters to check (must be frozenset for caching)

    Returns:
        True if font can render text, False otherwise
    """
    # Handle None or empty text
    if not text or not character_set:
        return False

    font_name = os.path.basename(font_path)

    # Check font health first
    if _font_health_manager:
        _font_health_manager.register_font(font_path)
        font_health = _font_health_manager.fonts[font_path]

        if font_health.is_in_cooldown():
            logging.debug(f"Font {font_name} is in cooldown, skipping")
            return False

        if font_health.health_score < _font_health_manager.min_health_threshold:
            logging.debug(f"Font {font_name} health too low ({font_health.health_score:.1f}), skipping")
            return False

    try:
        font = ImageFont.truetype(font_path, size=24)
        for char in text:
            if char not in character_set:
                return False

        # Track character coverage on success
        if _font_health_manager:
            _font_health_manager.fonts[font_path].character_coverage.update(character_set)

        return True
    except Exception as e:
        # Record failure in health manager
        if _font_health_manager:
            _font_health_manager.record_failure(font_path, reason="validation_error")
        logging.warning(f"Skipping font {font_name} due to error: {e}")
        return False
