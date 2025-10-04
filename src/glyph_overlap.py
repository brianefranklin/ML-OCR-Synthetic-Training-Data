"""
Language-Agnostic Glyph Overlap Module for OCR Synthetic Data Generator

Implements realistic text rendering with configurable character overlap
without relying on language-specific kerning tables or ligature rules.
Works with any script: Latin, CJK, Arabic, ancient languages, etc.
"""

import random
from typing import Tuple
from PIL import Image, ImageFilter
import numpy as np


class OverlapRenderer:
    """
    Handles language-agnostic glyph overlap rendering.

    Uses spacing reduction approach that works for any character set
    without requiring predefined kerning pairs or ligature tables.
    """

    @staticmethod
    def calculate_overlap_spacing(char_width: float,
                                  overlap_intensity: float,
                                  enable_variation: bool = True) -> float:
        """
        Calculate spacing reduction based on overlap intensity.

        Args:
            char_width: Original character width
            overlap_intensity: Overlap amount (0.0 to 1.0)
                0.0 = no overlap (standard spacing)
                0.5 = 50% spacing reduction
                1.0 = maximum overlap (~80% reduction to prevent complete overlap)
            enable_variation: Add natural random variation

        Returns:
            Adjusted spacing value
        """
        # Clamp intensity to valid range
        intensity = max(0.0, min(1.0, overlap_intensity))

        # Apply maximum overlap cap (80%) to prevent characters merging completely
        # This creates realistic overlap similar to tight kerning
        max_reduction = 0.80
        reduction_factor = intensity * max_reduction

        # Calculate base spacing
        base_spacing = char_width * (1.0 - reduction_factor)

        # Add natural variation (±5-10% of intensity) for realism
        if enable_variation and intensity > 0:
            variation = random.uniform(-0.1, 0.1) * intensity * char_width
            base_spacing += variation

        # Ensure spacing is never negative
        return max(0.0, base_spacing)

    @staticmethod
    def apply_ink_bleed(image: Image.Image,
                       ink_bleed_intensity: float) -> Image.Image:
        """
        Apply ink bleeding effect to simulate real-world document scanning.

        Args:
            image: Input image
            ink_bleed_intensity: Bleed strength (0.0 to 1.0)
                0.0 = no effect
                0.5 = moderate blur and edge softening
                1.0 = strong bleeding effect

        Returns:
            Image with ink bleed effect applied
        """
        if ink_bleed_intensity <= 0.0:
            return image

        # Clamp intensity
        intensity = max(0.0, min(1.0, ink_bleed_intensity))

        # Convert to grayscale for processing
        if image.mode == 'RGB':
            gray = image.convert('L')
        else:
            gray = image

        # Apply Gaussian blur to simulate ink spread
        # Blur radius scales with intensity
        blur_radius = random.uniform(0.3, 1.2) * intensity
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Apply threshold to maintain text darkness
        # This prevents text from becoming too light
        np_img = np.array(blurred)
        threshold = 200
        np_img = np.where(np_img < threshold, np_img * (1.0 - intensity * 0.1), np_img)

        # Optional: Apply slight morphological dilation for stronger bleed
        if intensity > 0.4:
            try:
                from scipy import ndimage
                # Small structural element for slight expansion
                struct = np.ones((2, 2))
                np_img = ndimage.grey_dilation(np_img, footprint=struct)
            except ImportError:
                # scipy not available, skip dilation
                pass

        # Convert back to PIL Image
        result = Image.fromarray(np_img.astype(np.uint8))

        # Convert back to RGB if original was RGB
        if image.mode == 'RGB':
            return result.convert('RGB')
        return result

    @staticmethod
    def calculate_vertical_overlap_spacing(char_height: float,
                                          overlap_intensity: float,
                                          enable_variation: bool = True) -> float:
        """
        Calculate vertical spacing reduction for vertical text directions.

        Args:
            char_height: Original character height
            overlap_intensity: Overlap amount (0.0 to 1.0)
            enable_variation: Add natural random variation

        Returns:
            Adjusted spacing value
        """
        # Use same logic as horizontal but for vertical dimension
        intensity = max(0.0, min(1.0, overlap_intensity))
        max_reduction = 0.80
        reduction_factor = intensity * max_reduction

        base_spacing = char_height * (1.0 - reduction_factor)

        if enable_variation and intensity > 0:
            variation = random.uniform(-0.1, 0.1) * intensity * char_height
            base_spacing += variation

        return max(0.0, base_spacing)

    @staticmethod
    def should_apply_ink_bleed(ink_bleed_intensity: float) -> bool:
        """
        Determine if ink bleed should be applied (probabilistic).

        Args:
            ink_bleed_intensity: Configured intensity

        Returns:
            True if should apply effect
        """
        if ink_bleed_intensity <= 0.0:
            return False

        # Apply with probability proportional to intensity
        # intensity 0.5 = 50% chance, 1.0 = 100% chance
        return random.random() < ink_bleed_intensity


# Utility functions for integration
def add_overlap_to_spacing(original_spacing: float,
                           overlap_intensity: float) -> float:
    """
    Simple helper to reduce spacing by overlap intensity.

    Args:
        original_spacing: Current spacing between characters
        overlap_intensity: Overlap amount (0.0 to 1.0)

    Returns:
        Adjusted spacing
    """
    intensity = max(0.0, min(1.0, overlap_intensity))
    max_reduction = 0.80
    reduction = intensity * max_reduction
    return max(0.0, original_spacing * (1.0 - reduction))
