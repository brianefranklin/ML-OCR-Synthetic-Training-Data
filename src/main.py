"""
OCR Synthetic Data Generator - Refactored
Generates synthetic text images with character-level bounding boxes for OCR training.
Supports multiple text directions: left-to-right, right-to-left, top-to-bottom, bottom-to-top.
"""

import argparse
import os
import json
import random
import sys
import time
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import bidi.algorithm

from augmentations import apply_augmentations


@dataclass
class CharacterBox:
    """Represents a character with its bounding box."""
    char: str
    bbox: List[float]  # [x_min, y_min, x_max, y_max]


class OCRDataGenerator:
    """
    OCR synthetic data generator with character-level bounding boxes.

    Features:
    - Character-level bounding boxes for precise OCR training
    - Multi-directional text support (LTR, RTL, TTB, BTT)
    - Proper BiDi text rendering for RTL languages
    - Corpus-based text generation
    - Configurable augmentation pipeline
    """

    DIRECTION_NAMES = {
        'left_to_right': 'LTR',
        'right_to_left': 'RTL',
        'top_to_bottom': 'TTB',
        'bottom_to_top': 'BTT'
    }

    def __init__(self,
                 font_files: List[str],
                 background_images: Optional[List[str]] = None):
        """
        Initialize the OCR data generator.

        Args:
            font_files: List of paths to font files (.ttf, .otf)
            background_images: Optional list of background image paths
        """
        self.font_files = font_files
        self.background_images = background_images or []

    def render_curved_text(self,
                          text: str,
                          font: ImageFont.FreeTypeFont,
                          curve_type: str = 'arc',
                          curve_intensity: float = 0.3,
                          overlap_intensity: float = 0.0,
                          ink_bleed_intensity: float = 0.0,
                          effect_type: str = 'none',
                          effect_depth: float = 0.5,
                          light_azimuth: float = 135.0,
                          light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                                color_palette: str = 'realistic_dark',
                                                                custom_colors: List[Tuple[int, int, int]] = None,
                                                                background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render text along a curve (arc or sine wave).

        Args:
            text: Text to render
            font: Font to use
            curve_type: 'arc' for circular arc, 'sine' for wave
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        import math
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Handle zero or negative intensity - fall back to straight rendering
        if curve_intensity <= 0.0:
            return self.render_left_to_right(text, font, overlap_intensity, ink_bleed_intensity)

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_info = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            char_info.append({'char': char, 'width': width, 'height': height})

        # Calculate total width with overlap
        total_width = 0
        for i, c in enumerate(char_info):
            if i == 0:
                total_width += c['width']
            else:
                spacing = OverlapRenderer.calculate_overlap_spacing(
                    c['width'], overlap_intensity, enable_variation=False
                )
                total_width += spacing

        max_height = max(c['height'] for c in char_info) if char_info else 20

        # Prevent division by zero
        if total_width == 0:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Clamp intensity to reasonable range
        curve_intensity = max(0.01, min(curve_intensity, 1.0))

        # Calculate curve parameters
        if curve_type == 'arc':
            # Circular arc
            base_radius = total_width / (2 * curve_intensity)
            radius = max(base_radius, total_width)
            arc_height = (total_width ** 2) / (8 * radius)
            curve_height = int(arc_height * 2 + max_height + 80)
        else:  # sine wave
            wavelength = total_width
            amplitude = max_height * curve_intensity * 1.5
            curve_height = int(max_height + amplitude * 2 + 80)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create oversized canvas with transparent background
        img_width = int(total_width + 100)
        img_height = curve_height
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters along curve
        char_boxes = []
        x_pos = 50

        for i, info in enumerate(char_info):
            char = info['char']
            char_width = info['width']
            char_height = info['height']
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)

            # Calculate position and rotation
            if curve_type == 'arc':
                # Position on arc
                theta = (x_pos - total_width/2 - 50) / radius
                y_offset = radius * (1 - math.cos(theta)) if radius > 0 else 0
                rotation_angle = math.degrees(theta)
                x_draw = int(x_pos)
                y_draw = int(img_height / 2 - y_offset)
            else:  # sine
                # Sine wave
                phase = (x_pos / total_width) * 2 * math.pi * (1 + curve_intensity)
                y_offset = amplitude * math.sin(phase)
                # Tangent angle for rotation
                rotation_angle = math.degrees(math.atan(
                    amplitude * 2 * math.pi * (1 + curve_intensity) / total_width * math.cos(phase)
                ))
                x_draw = int(x_pos)
                y_draw = int(img_height / 2 + y_offset)

            # Render character
            if abs(rotation_angle) > 0.5:
                # Get original character bbox before rotation
                temp_char_img = Image.new('RGBA', (char_width + 20, char_height + 20), (255, 255, 255, 0))
                temp_char_draw = ImageDraw.Draw(temp_char_img)
                temp_char_draw.text((10, 10), char, font=font, fill=char_color)
                original_bbox = temp_char_draw.textbbox((10, 10), char, font=font)

                # Create rotated character image
                char_img = Image.new('RGBA', (char_width + 60, char_height + 60), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_center_x = 30
                char_center_y = 30
                char_draw.text((char_center_x, char_center_y), char, font=font, fill=char_color)
                rotated = char_img.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

                # Paste with transparency
                paste_x = int(x_draw - rotated.width / 2)
                paste_y = int(y_draw - rotated.height / 2)
                image.paste(rotated, (paste_x, paste_y), rotated)

                # Calculate accurate bbox using rotation matrix
                # Original bbox corners relative to character center
                orig_x0, orig_y0, orig_x1, orig_y1 = original_bbox
                cx, cy = 10 + (orig_x1 - orig_x0) / 2, 10 + (orig_y1 - orig_y0) / 2

                corners = [
                    (orig_x0 - 10, orig_y0 - 10),  # top-left
                    (orig_x1 - 10, orig_y0 - 10),  # top-right
                    (orig_x1 - 10, orig_y1 - 10),  # bottom-right
                    (orig_x0 - 10, orig_y1 - 10),  # bottom-left
                ]

                # Apply rotation matrix to each corner
                angle_rad = math.radians(-rotation_angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                rotated_corners = []
                for x, y in corners:
                    # Rotate around origin
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    # Translate to actual position
                    rotated_corners.append((new_x + x_draw, new_y + y_draw))

                # Find bounding box of rotated corners
                xs = [corner[0] for corner in rotated_corners]
                ys = [corner[1] for corner in rotated_corners]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                # Straight character
                draw.text((x_draw, y_draw - char_height/2), char, font=font, fill=char_color)
                bbox = [x_draw, y_draw - char_height/2,
                       x_draw + char_width, y_draw + char_height/2]

            char_boxes.append(CharacterBox(char, bbox))

            # Apply overlap to spacing
            spacing = OverlapRenderer.calculate_overlap_spacing(
                char_width, overlap_intensity, enable_variation=True
            )
            x_pos += spacing

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_right_to_left_curved(self,
                                    text: str,
                                    font: ImageFont.FreeTypeFont,
                                    curve_type: str = 'arc',
                                    curve_intensity: float = 0.3,
                                    overlap_intensity: float = 0.0,
                                    ink_bleed_intensity: float = 0.0,
                                    effect_type: str = 'none',
                                    effect_depth: float = 0.5,
                                    light_azimuth: float = 135.0,
                                    light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                                color_palette: str = 'realistic_dark',
                                                                custom_colors: List[Tuple[int, int, int]] = None,
                                                                background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render RTL text along a curve (arc or sine wave).

        Args:
            text: Text to render (RTL)
            font: Font to use
            curve_type: 'arc' for circular arc, 'sine' for wave
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        import math
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Handle zero or negative intensity - fall back to straight rendering
        if curve_intensity <= 0.0:
            return self.render_right_to_left(text, font, overlap_intensity, ink_bleed_intensity)

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_info = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            char_info.append({'char': char, 'width': width, 'height': height})

        # Calculate total width with overlap
        total_width = 0
        for i, c in enumerate(char_info):
            if i == 0:
                total_width += c['width']
            else:
                spacing = OverlapRenderer.calculate_overlap_spacing(
                    c['width'], overlap_intensity, enable_variation=False
                )
                total_width += spacing

        max_height = max(c['height'] for c in char_info) if char_info else 20

        # Prevent division by zero
        if total_width == 0:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Clamp intensity to reasonable range
        curve_intensity = max(0.01, min(curve_intensity, 1.0))

        # Calculate curve parameters
        if curve_type == 'arc':
            # Circular arc
            base_radius = total_width / (2 * curve_intensity)
            radius = max(base_radius, total_width)
            arc_height = (total_width ** 2) / (8 * radius)
            curve_height = int(arc_height * 2 + max_height + 80)
        else:  # sine wave
            wavelength = total_width
            amplitude = max_height * curve_intensity * 1.5
            curve_height = int(max_height + amplitude * 2 + 80)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create oversized canvas with transparent background
        img_width = int(total_width + 100)
        img_height = curve_height
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters along curve (RTL: start from right)
        char_boxes = []
        x_pos = img_width - 50  # Start from right edge

        for i, info in enumerate(char_info):
            char = info['char']
            char_width = info['width']
            char_height = info['height']
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)

            # Calculate position and rotation (RTL: mirror curve horizontally)
            if curve_type == 'arc':
                # Position on arc (mirrored)
                theta = -((x_pos - total_width/2 - 50) / radius)  # Negative for RTL
                y_offset = radius * (1 - math.cos(theta)) if radius > 0 else 0
                rotation_angle = math.degrees(theta)
                x_draw = int(x_pos - char_width)  # Draw from right
                y_draw = int(img_height / 2 - y_offset)
            else:  # sine
                # Sine wave (mirrored phase)
                phase = ((img_width - x_pos) / total_width) * 2 * math.pi * (1 + curve_intensity)
                y_offset = amplitude * math.sin(phase)
                # Tangent angle for rotation
                rotation_angle = -math.degrees(math.atan(
                    amplitude * 2 * math.pi * (1 + curve_intensity) / total_width * math.cos(phase)
                ))
                x_draw = int(x_pos - char_width)
                y_draw = int(img_height / 2 + y_offset)

            # Render character
            if abs(rotation_angle) > 0.5:
                # Get original character bbox before rotation
                temp_char_img = Image.new('RGBA', (char_width + 20, char_height + 20), (255, 255, 255, 0))
                temp_char_draw = ImageDraw.Draw(temp_char_img)
                temp_char_draw.text((10, 10), char, font=font, fill=char_color)
                original_bbox = temp_char_draw.textbbox((10, 10), char, font=font)

                # Create rotated character image
                char_img = Image.new('RGBA', (char_width + 60, char_height + 60), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_center_x = 30
                char_center_y = 30
                char_draw.text((char_center_x, char_center_y), char, font=font, fill=char_color)
                rotated = char_img.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

                # Paste with transparency
                paste_x = int(x_draw + char_width/2 - rotated.width / 2)
                paste_y = int(y_draw - rotated.height / 2)
                image.paste(rotated, (paste_x, paste_y), rotated)

                # Calculate accurate bbox using rotation matrix
                orig_x0, orig_y0, orig_x1, orig_y1 = original_bbox
                cx, cy = 10 + (orig_x1 - orig_x0) / 2, 10 + (orig_y1 - orig_y0) / 2

                corners = [
                    (orig_x0 - 10, orig_y0 - 10),
                    (orig_x1 - 10, orig_y0 - 10),
                    (orig_x1 - 10, orig_y1 - 10),
                    (orig_x0 - 10, orig_y1 - 10),
                ]

                # Apply rotation matrix
                angle_rad = math.radians(-rotation_angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                rotated_corners = []
                for x, y in corners:
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_corners.append((new_x + x_draw + char_width/2, new_y + y_draw))

                # Find bounding box
                xs = [corner[0] for corner in rotated_corners]
                ys = [corner[1] for corner in rotated_corners]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                # Straight character
                draw.text((x_draw, y_draw - char_height/2), char, font=font, fill=char_color)
                bbox = [x_draw, y_draw - char_height/2,
                       x_draw + char_width, y_draw + char_height/2]

            char_boxes.append(CharacterBox(char, bbox))

            # Apply overlap to spacing (move left for RTL)
            spacing = OverlapRenderer.calculate_overlap_spacing(
                char_width, overlap_intensity, enable_variation=True
            )
            x_pos -= spacing

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def load_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """
        Load a TrueType/OpenType font.

        Args:
            font_path: Path to font file
            size: Font size in points

        Returns:
            Loaded font object
        """
        return ImageFont.truetype(font_path, size=size)

    def extract_text_segment(self,
                           corpus: str,
                           min_length: int,
                           max_length: int,
                           max_attempts: int = 100) -> Optional[str]:
        """
        Extract a random text segment from corpus.

        Args:
            corpus: Source text corpus
            min_length: Minimum text length
            max_length: Maximum text length
            max_attempts: Maximum attempts to find valid segment

        Returns:
            Text segment or None if failed
        """
        text_line = ""
        attempts = 0

        while len(text_line) < min_length and attempts < max_attempts:
            text_length = random.randint(min_length, max_length)
            start_index = random.randint(0, len(corpus) - text_length)
            text_line = corpus[start_index:start_index + text_length].replace('\n', ' ').strip()
            attempts += 1

        return text_line if len(text_line) >= min_length else None

    def render_left_to_right(self,
                            text: str,
                            font: ImageFont.FreeTypeFont,
                            overlap_intensity: float = 0.0,
                            ink_bleed_intensity: float = 0.0,
                            effect_type: str = 'none',
                            effect_depth: float = 0.5,
                            light_azimuth: float = 135.0,
                            light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                            color_palette: str = 'realistic_dark',
                            custom_colors: List[Tuple[int, int, int]] = None,
                            background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render left-to-right horizontal text with character bboxes.

        Args:
            text: Text to render
            font: Font to use
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)
            text_color_mode: Color mode ('uniform', 'per_glyph', 'gradient', 'random')
            color_palette: Color palette name ('realistic_dark', 'vibrant', 'pastels', etc.)
            custom_colors: Optional list of custom RGB tuples
            background_color: DEPRECATED - Ignored. Text rendered with transparent background.

        Returns:
            Tuple of (image, character_boxes)

        Note:
            All render functions now return RGBA images with transparent backgrounds.
            The background_color parameter is deprecated and ignored.
        """
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Measure characters and calculate width with overlap
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        total_width = 40  # Margins
        for i, char in enumerate(text):
            char_width = temp_draw.textlength(char, font=font)
            if i == 0:
                total_width += char_width
            else:
                spacing = OverlapRenderer.calculate_overlap_spacing(
                    char_width, overlap_intensity, enable_variation=False  # No variation for measurement
                )
                total_width += spacing

        total_text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create actual image with transparent background
        image = Image.new('RGBA', (int(total_width), img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters and collect bboxes
        char_boxes = []
        x_offset = 20
        y_offset = 15

        for i, char in enumerate(text):
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)
            char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
            draw.text((x_offset, y_offset), char, font=font, fill=char_color)
            char_boxes.append(CharacterBox(char, list(char_bbox)))

            # Apply overlap to spacing
            char_width = draw.textlength(char, font=font)
            spacing = OverlapRenderer.calculate_overlap_spacing(
                char_width, overlap_intensity, enable_variation=True
            )
            x_offset += spacing

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_right_to_left(self,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           overlap_intensity: float = 0.0,
                           ink_bleed_intensity: float = 0.0,
                           effect_type: str = 'none',
                           effect_depth: float = 0.5,
                           light_azimuth: float = 135.0,
                           light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                        color_palette: str = 'realistic_dark',
                                                        custom_colors: List[Tuple[int, int, int]] = None,
                                                        background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render right-to-left horizontal text with proper BiDi handling.

        Args:
            text: Text to render
            font: Font to use
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Use BiDi algorithm for proper RTL display
        display_text = bidi.algorithm.get_display(text)

        # Measure characters and calculate width with overlap
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        total_width = 40  # Margins
        for i, char in enumerate(display_text):
            char_width = temp_draw.textlength(char, font=font)
            if i == 0:
                total_width += char_width
            else:
                base_spacing = 1
                spacing = OverlapRenderer.calculate_overlap_spacing(
                    char_width, overlap_intensity, enable_variation=False
                )
                total_width += max(base_spacing, spacing * 0.1)

        total_text_bbox = temp_draw.textbbox((0, 0), display_text, font=font)
        img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            display_text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create actual image with transparent background
        image = Image.new('RGBA', (int(total_width), img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters from right to left
        char_boxes = []
        x_offset = int(total_width) - 20
        y_offset = 15

        for i, char in enumerate(display_text):
            char_width = draw.textlength(char, font=font)
            x_offset -= char_width

            # Add full opacity to RGB color
            char_color = text_colors[i]
            if len(char_color) == 3:
                char_color = (*char_color, 255)
            draw.text((x_offset, y_offset), char, font=font, fill=char_color)
            char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))

            # Apply overlap to spacing (small base spacing + overlap reduction)
            base_spacing = 1
            spacing = OverlapRenderer.calculate_overlap_spacing(
                char_width, overlap_intensity, enable_variation=True
            )
            x_offset -= max(base_spacing, spacing * 0.1)  # Ensure some spacing

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_top_to_bottom(self,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           overlap_intensity: float = 0.0,
                           ink_bleed_intensity: float = 0.0,
                           effect_type: str = 'none',
                           effect_depth: float = 0.5,
                           light_azimuth: float = 135.0,
                           light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                        color_palette: str = 'realistic_dark',
                                                        custom_colors: List[Tuple[int, int, int]] = None,
                                                        background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render top-to-bottom vertical text (traditional CJK style).

        Args:
            text: Text to render
            font: Font to use
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_widths = []
        char_heights = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_widths.append(bbox[2] - bbox[0])
            char_heights.append(bbox[3] - bbox[1])

        max_char_width = max(char_widths) if char_widths else 0

        # Calculate height with overlap
        total_height = 30  # Margins
        base_spacing = 5
        for i, char_height in enumerate(char_heights):
            total_height += char_height
            if i < len(char_heights) - 1:  # Not last character
                reduced_spacing = max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))
                total_height += reduced_spacing

        img_width = max_char_width + 40
        img_height = int(total_height)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create actual image with transparent background
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters top to bottom
        char_boxes = []
        y_cursor = 15

        for i, char in enumerate(text):
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)
            char_bbox_temp = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox_temp[2] - char_bbox_temp[0]
            char_height = char_bbox_temp[3] - char_bbox_temp[1]
            x_cursor = (img_width - char_width) / 2

            draw.text((x_cursor, y_cursor), char, font=font, fill=char_color)
            char_bbox = draw.textbbox((x_cursor, y_cursor), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))
            logging.debug(f"char: {char}, bbox: {char_bbox}")

            # Apply vertical overlap to spacing
            y_cursor += char_height + max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_bottom_to_top(self,
                           text: str,
                           font: ImageFont.FreeTypeFont,
                           overlap_intensity: float = 0.0,
                           ink_bleed_intensity: float = 0.0,
                           effect_type: str = 'none',
                           effect_depth: float = 0.5,
                           light_azimuth: float = 135.0,
                           light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                        color_palette: str = 'realistic_dark',
                                                        custom_colors: List[Tuple[int, int, int]] = None,
                                                        background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render bottom-to-top vertical text.

        Args:
            text: Text to render
            font: Font to use
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_widths = []
        char_heights = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_widths.append(bbox[2] - bbox[0])
            char_heights.append(bbox[3] - bbox[1])

        max_char_width = max(char_widths) if char_widths else 0

        # Calculate height with overlap
        total_height = 30  # Margins
        base_spacing = 5
        for i, char_height in enumerate(char_heights):
            total_height += char_height
            if i < len(char_heights) - 1:  # Not last character
                reduced_spacing = max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))
                total_height += reduced_spacing

        img_width = max_char_width + 40
        img_height = int(total_height)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create actual image with transparent background
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters bottom to top
        char_boxes = []
        y_cursor = img_height - 15

        for i, char in enumerate(text):
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)
            char_bbox_temp = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox_temp[2] - char_bbox_temp[0]
            char_height = char_bbox_temp[3] - char_bbox_temp[1]
            x_cursor = (img_width - char_width) / 2
            y_cursor -= char_height

            draw.text((x_cursor, y_cursor), char, font=font, fill=char_color)
            char_bbox = draw.textbbox((x_cursor, y_cursor), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))
            logging.debug(f"char: {char}, bbox: {char_bbox}")

            # Apply vertical overlap to spacing
            y_cursor -= max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_top_to_bottom_curved(self,
                                    text: str,
                                    font: ImageFont.FreeTypeFont,
                                    curve_type: str = 'arc',
                                    curve_intensity: float = 0.3,
                                    overlap_intensity: float = 0.0,
                                    ink_bleed_intensity: float = 0.0,
                                    effect_type: str = 'none',
                                    effect_depth: float = 0.5,
                                    light_azimuth: float = 135.0,
                                    light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                                color_palette: str = 'realistic_dark',
                                                                custom_colors: List[Tuple[int, int, int]] = None,
                                                                background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render text along a curved vertical baseline from top to bottom.

        Args:
            text: Text to render
            font: Font to use
            curve_type: 'arc' for circular arc, 'sine' for wave
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)

        Returns:
            Tuple of (image, character_boxes)
        """
        import math
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Handle zero or negative intensity - fall back to straight rendering
        if curve_intensity <= 0.0:
            return self.render_top_to_bottom(text, font, overlap_intensity, ink_bleed_intensity)

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_info = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            char_info.append({'char': char, 'width': width, 'height': height})

        # Calculate total height with overlap
        base_spacing = 5
        total_height = 0
        for i, c in enumerate(char_info):
            total_height += c['height']
            if i < len(char_info) - 1:  # Not last character
                reduced_spacing = max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))
                total_height += reduced_spacing

        max_width = max(c['width'] for c in char_info) if char_info else 20

        # Prevent division by zero
        if total_height == 0:
            empty_img = Image.new('RGB', (10, 10), color='white')
            return empty_img, []

        # Clamp intensity to reasonable range
        curve_intensity = max(0.01, min(curve_intensity, 1.0))

        # Calculate curve parameters
        if curve_type == 'arc':
            # Circular arc for vertical text
            base_radius = total_height / (2 * curve_intensity)
            radius = max(base_radius, total_height)
            arc_width = (total_height ** 2) / (8 * radius)
            curve_width = int(arc_width * 2 + max_width + 80)
        else:  # sine wave
            wavelength = total_height
            amplitude = max_width * curve_intensity * 1.5
            curve_width = int(max_width + amplitude * 2 + 80)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create oversized canvas with transparent background
        img_height = int(total_height + 100)
        img_width = curve_width
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters along vertical curve
        char_boxes = []
        y_pos = 50

        for i, info in enumerate(char_info):
            char = info['char']
            char_width = info['width']
            char_height = info['height']
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)

            # Calculate position and rotation for vertical curve
            if curve_type == 'arc':
                # Position on vertical arc
                theta = (y_pos - total_height/2 - 50) / radius
                x_offset = radius * (1 - math.cos(theta)) if radius > 0 else 0
                rotation_angle = math.degrees(theta)
                y_draw = int(y_pos)
                x_draw = int(img_width / 2 - x_offset)
            else:  # sine
                # Sine wave for vertical text
                phase = (y_pos / total_height) * 2 * math.pi * (1 + curve_intensity)
                x_offset = amplitude * math.sin(phase)
                # Tangent angle for rotation
                rotation_angle = math.degrees(math.atan(
                    amplitude * 2 * math.pi * (1 + curve_intensity) / total_height * math.cos(phase)
                ))
                y_draw = int(y_pos)
                x_draw = int(img_width / 2 + x_offset)

            # Render character
            if abs(rotation_angle) > 0.5:
                # Get original character bbox before rotation
                temp_char_img = Image.new('RGBA', (char_width + 20, char_height + 20), (255, 255, 255, 0))
                temp_char_draw = ImageDraw.Draw(temp_char_img)
                temp_char_draw.text((10, 10), char, font=font, fill=char_color)
                original_bbox = temp_char_draw.textbbox((10, 10), char, font=font)

                # Create rotated character image
                char_img = Image.new('RGBA', (char_width + 60, char_height + 60), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_center_x = 30
                char_center_y = 30
                char_draw.text((char_center_x, char_center_y), char, font=font, fill=char_color)
                rotated = char_img.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

                # Paste with transparency
                paste_x = int(x_draw - rotated.width / 2)
                paste_y = int(y_draw - rotated.height / 2)
                image.paste(rotated, (paste_x, paste_y), rotated)

                # Calculate accurate bbox using rotation matrix
                orig_x0, orig_y0, orig_x1, orig_y1 = original_bbox

                corners = [
                    (orig_x0 - 10, orig_y0 - 10),  # top-left
                    (orig_x1 - 10, orig_y0 - 10),  # top-right
                    (orig_x1 - 10, orig_y1 - 10),  # bottom-right
                    (orig_x0 - 10, orig_y1 - 10),  # bottom-left
                ]

                # Apply rotation matrix to each corner
                angle_rad = math.radians(-rotation_angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                rotated_corners = []
                for x, y in corners:
                    # Rotate around origin
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    # Translate to actual position
                    rotated_corners.append((new_x + x_draw, new_y + y_draw))

                # Find bounding box of rotated corners
                xs = [corner[0] for corner in rotated_corners]
                ys = [corner[1] for corner in rotated_corners]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                # Straight character
                draw.text((x_draw - char_width/2, y_draw), char, font=font, fill=char_color)
                bbox = [x_draw - char_width/2, y_draw,
                       x_draw + char_width/2, y_draw + char_height]

            char_boxes.append(CharacterBox(char, bbox))

            # Apply vertical overlap to spacing
            base_spacing = 5
            reduced_spacing = max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))
            y_pos += char_height + reduced_spacing

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_bottom_to_top_curved(self,
                                    text: str,
                                    font: ImageFont.FreeTypeFont,
                                    curve_type: str = 'arc',
                                    curve_intensity: float = 0.3,
                                    overlap_intensity: float = 0.0,
                                    ink_bleed_intensity: float = 0.0,
                                    effect_type: str = 'none',
                                    effect_depth: float = 0.5,
                                    light_azimuth: float = 135.0,
                                    light_elevation: float = 45.0,
                            text_color_mode: str = 'uniform',
                                                                color_palette: str = 'realistic_dark',
                                                                custom_colors: List[Tuple[int, int, int]] = None,
                                                                background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render text along a curved vertical baseline from bottom to top.

        Args:
            text: Text to render
            font: Font to use
            curve_type: 'arc' for circular arc, 'sine' for wave
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Character overlap amount (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect strength (0.0-1.0)

        Returns:
            Tuple of (image, character_boxes)
        """
        import math
        from glyph_overlap import OverlapRenderer
        from text_color import ColorRenderer

        # Handle empty text
        if not text:
            empty_img = Image.new('RGBA', (10, 10), color=(255, 255, 255, 0))
            return empty_img, []

        # Handle zero or negative intensity - fall back to straight rendering
        if curve_intensity <= 0.0:
            return self.render_bottom_to_top(text, font, overlap_intensity, ink_bleed_intensity)

        # Measure characters
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_info = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            char_info.append({'char': char, 'width': width, 'height': height})

        # Calculate total height with overlap
        base_spacing = 5
        total_height = 0
        for i, c in enumerate(char_info):
            total_height += c['height']
            if i < len(char_info) - 1:  # Not last character
                reduced_spacing = max(0, base_spacing - (base_spacing * overlap_intensity * 0.8))
                total_height += reduced_spacing

        max_width = max(c['width'] for c in char_info) if char_info else 20

        # Prevent division by zero
        if total_height == 0:
            empty_img = Image.new('RGB', (10, 10), color='white')
            return empty_img, []

        # Clamp intensity to reasonable range
        curve_intensity = max(0.01, min(curve_intensity, 1.0))

        # Calculate curve parameters
        if curve_type == 'arc':
            # Circular arc for vertical text
            base_radius = total_height / (2 * curve_intensity)
            radius = max(base_radius, total_height)
            arc_width = (total_height ** 2) / (8 * radius)
            curve_width = int(arc_width * 2 + max_width + 80)
        else:  # sine wave
            wavelength = total_height
            amplitude = max_width * curve_intensity * 1.5
            curve_width = int(max_width + amplitude * 2 + 80)

        # Generate text colors
        text_colors = ColorRenderer.generate_line_colors(
            text, text_color_mode, color_palette, custom_colors
        )

        # Determine background color
        bg_color = background_color
        if bg_color == 'auto' or bg_color is None:
            bg_color = ColorRenderer.get_contrasting_color(text_colors[0])
        elif isinstance(bg_color, str):
            parsed = ColorRenderer.parse_color_string(bg_color)
            bg_color = parsed if parsed else (255, 255, 255)

        # Create oversized canvas with transparent background
        img_height = int(total_height + 100)
        img_width = curve_width
        image = Image.new('RGBA', (img_width, img_height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        # Render characters along vertical curve (bottom to top)
        char_boxes = []
        y_pos = img_height - 50  # Start from bottom

        for i, info in enumerate(char_info):
            char = info['char']
            char_width = info['width']
            char_height = info['height']
            char_color = text_colors[i]
            # Add full opacity to RGB color
            if len(char_color) == 3:
                char_color = (*char_color, 255)

            # Move up by character height first (bottom-to-top)
            y_pos -= char_height

            # Calculate position and rotation for vertical curve
            # Invert the curve direction for bottom-to-top
            if curve_type == 'arc':
                # Position on vertical arc (inverted)
                theta = (y_pos - total_height/2 - 50) / radius
                x_offset = radius * (1 - math.cos(theta)) if radius > 0 else 0
                # Invert the curve by negating x_offset
                rotation_angle = math.degrees(theta)
                y_draw = int(y_pos)
                x_draw = int(img_width / 2 + x_offset)  # Note: + instead of -
            else:  # sine
                # Sine wave for vertical text (inverted phase)
                phase = (y_pos / total_height) * 2 * math.pi * (1 + curve_intensity)
                x_offset = amplitude * math.sin(phase)
                # Tangent angle for rotation
                rotation_angle = math.degrees(math.atan(
                    amplitude * 2 * math.pi * (1 + curve_intensity) / total_height * math.cos(phase)
                ))
                y_draw = int(y_pos)
                x_draw = int(img_width / 2 - x_offset)  # Note: - instead of +

            # Render character
            if abs(rotation_angle) > 0.5:
                # Get original character bbox before rotation
                temp_char_img = Image.new('RGBA', (char_width + 20, char_height + 20), (255, 255, 255, 0))
                temp_char_draw = ImageDraw.Draw(temp_char_img)
                temp_char_draw.text((10, 10), char, font=font, fill=char_color)
                original_bbox = temp_char_draw.textbbox((10, 10), char, font=font)

                # Create rotated character image
                char_img = Image.new('RGBA', (char_width + 60, char_height + 60), (255, 255, 255, 0))
                char_draw = ImageDraw.Draw(char_img)
                char_center_x = 30
                char_center_y = 30
                char_draw.text((char_center_x, char_center_y), char, font=font, fill=char_color)
                rotated = char_img.rotate(-rotation_angle, expand=True, fillcolor=(255, 255, 255, 0))

                # Paste with transparency
                paste_x = int(x_draw - rotated.width / 2)
                paste_y = int(y_draw - rotated.height / 2)
                image.paste(rotated, (paste_x, paste_y), rotated)

                # Calculate accurate bbox using rotation matrix
                orig_x0, orig_y0, orig_x1, orig_y1 = original_bbox

                corners = [
                    (orig_x0 - 10, orig_y0 - 10),  # top-left
                    (orig_x1 - 10, orig_y0 - 10),  # top-right
                    (orig_x1 - 10, orig_y1 - 10),  # bottom-right
                    (orig_x0 - 10, orig_y1 - 10),  # bottom-left
                ]

                # Apply rotation matrix to each corner
                angle_rad = math.radians(-rotation_angle)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                rotated_corners = []
                for x, y in corners:
                    # Rotate around origin
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    # Translate to actual position
                    rotated_corners.append((new_x + x_draw, new_y + y_draw))

                # Find bounding box of rotated corners
                xs = [corner[0] for corner in rotated_corners]
                ys = [corner[1] for corner in rotated_corners]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            else:
                # Straight character
                draw.text((x_draw - char_width/2, y_draw), char, font=font, fill=char_color)
                bbox = [x_draw - char_width/2, y_draw,
                       x_draw + char_width/2, y_draw + char_height]

            char_boxes.append(CharacterBox(char, bbox))

        # Apply ink bleed effect if enabled
        if OverlapRenderer.should_apply_ink_bleed(ink_bleed_intensity):
            image = OverlapRenderer.apply_ink_bleed(image, ink_bleed_intensity)

        # Apply 3D effect if enabled
        from text_3d_effects import Text3DEffects
        if Text3DEffects.should_apply_effect(effect_type, effect_depth):
            image = Text3DEffects.apply_effect(image, effect_type, effect_depth,
                                              light_azimuth, light_elevation)

        return image, char_boxes

    def render_text(self,
                   text: str,
                   font: ImageFont.FreeTypeFont,
                   direction: str,
                   overlap_intensity: float = 0.0,
                   ink_bleed_intensity: float = 0.0,
                   effect_type: str = 'none',
                   effect_depth: float = 0.5,
                   light_azimuth: float = 135.0,
                   light_elevation: float = 45.0,
                   text_color_mode: str = 'uniform',
                   color_palette: str = 'realistic_dark',
                   custom_colors: List[Tuple[int, int, int]] = None,
                   background_color: Union[Tuple[int, int, int], str] = 'auto') -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render text in specified direction with character-level bboxes.

        Args:
            text: Text to render
            font: Font to use
            direction: Text direction ('left_to_right', 'right_to_left', etc.)
            overlap_intensity: Glyph overlap intensity (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect intensity (0.0-1.0)
            effect_type: 3D effect type ('none', 'raised', 'embossed', 'engraved')
            effect_depth: 3D effect depth (0.0-1.0)
            light_azimuth: Light direction angle (0-360 degrees)
            light_elevation: Light elevation angle (0-90 degrees)

        Returns:
            Tuple of (image, character_boxes)
        """
        if direction == 'left_to_right':
            return self.render_left_to_right(text, font, overlap_intensity, ink_bleed_intensity,
                                            effect_type, effect_depth, light_azimuth, light_elevation,
                                            text_color_mode, color_palette, custom_colors, background_color)
        elif direction == 'right_to_left':
            return self.render_right_to_left(text, font, overlap_intensity, ink_bleed_intensity,
                                            effect_type, effect_depth, light_azimuth, light_elevation,
                                            text_color_mode, color_palette, custom_colors, background_color)
        elif direction == 'top_to_bottom':
            return self.render_top_to_bottom(text, font, overlap_intensity, ink_bleed_intensity,
                                            effect_type, effect_depth, light_azimuth, light_elevation,
                                            text_color_mode, color_palette, custom_colors, background_color)
        elif direction == 'bottom_to_top':
            return self.render_bottom_to_top(text, font, overlap_intensity, ink_bleed_intensity,
                                            effect_type, effect_depth, light_azimuth, light_elevation,
                                            text_color_mode, color_palette, custom_colors, background_color)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def generate_image(self,
                      text: str,
                      font_path: str,
                      font_size: int,
                      direction: str,
                      curve_type: str = 'none',
                      curve_intensity: float = 0.0,
                      overlap_intensity: float = 0.0,
                      ink_bleed_intensity: float = 0.0,
                      effect_type: str = 'none',
                      effect_depth: float = 0.5,
                      light_azimuth: float = 135.0,
                      light_elevation: float = 45.0,
                      text_color_mode: str = 'uniform',
                      color_palette: str = 'realistic_dark',
                      custom_colors: List[Tuple[int, int, int]] = None,
                      background_color: Union[Tuple[int, int, int], str] = 'auto',
                      canvas_enabled: bool = True,
                      canvas_size: Tuple[int, int] = None,
                      canvas_min_padding: int = 10,
                      canvas_placement: str = 'weighted_random',
                      canvas_max_megapixels: float = 12.0) -> Tuple[Image.Image, Dict, str]:
        """
        Generate a single synthetic OCR image with augmentations.

        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size in points
            direction: Text direction
            curve_type: Type of text curvature ('none', 'arc', 'sine')
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Glyph overlap intensity (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect intensity (0.0-1.0)
            canvas_enabled: Whether to place text on larger canvas
            canvas_size: Fixed canvas size (if None, generates random size)
            canvas_min_padding: Minimum padding around text
            canvas_placement: Placement strategy ('weighted_random', 'uniform_random', 'center')
            canvas_max_megapixels: Maximum canvas size in megapixels

        Returns:
            Tuple of (final_image, metadata_dict, text)
            If canvas_enabled=False: metadata_dict = {'char_bboxes': [...]}
            If canvas_enabled=True: metadata_dict = {'canvas_size': [...], 'text_placement': [...], 'line_bbox': [...], 'char_bboxes': [...]}
        """
        # Load font
        font = self.load_font(font_path, font_size)

        # Render text with character bboxes
        if curve_type != 'none' and curve_intensity > 0:
            # Use curved rendering based on direction
            if direction == 'left_to_right':
                image, char_boxes = self.render_curved_text(text, font, curve_type, curve_intensity,
                                                            overlap_intensity, ink_bleed_intensity,
                                                            effect_type, effect_depth, light_azimuth, light_elevation,
                                                            text_color_mode, color_palette, custom_colors, background_color)
            elif direction == 'top_to_bottom':
                image, char_boxes = self.render_top_to_bottom_curved(text, font, curve_type, curve_intensity,
                                                                     overlap_intensity, ink_bleed_intensity,
                                                                     effect_type, effect_depth, light_azimuth, light_elevation,
                                                                     text_color_mode, color_palette, custom_colors, background_color)
            elif direction == 'bottom_to_top':
                image, char_boxes = self.render_bottom_to_top_curved(text, font, curve_type, curve_intensity,
                                                                     overlap_intensity, ink_bleed_intensity,
                                                                     effect_type, effect_depth, light_azimuth, light_elevation,
                                                                     text_color_mode, color_palette, custom_colors, background_color)
            elif direction == 'right_to_left':
                image, char_boxes = self.render_right_to_left_curved(text, font, curve_type, curve_intensity,
                                                                     overlap_intensity, ink_bleed_intensity,
                                                                     effect_type, effect_depth, light_azimuth, light_elevation,
                                                                     text_color_mode, color_palette, custom_colors, background_color)
            else:
                image, char_boxes = self.render_text(text, font, direction,
                                                    overlap_intensity, ink_bleed_intensity,
                                                    effect_type, effect_depth, light_azimuth, light_elevation,
                                                    text_color_mode, color_palette, custom_colors, background_color)
        else:
            # Use standard rendering
            image, char_boxes = self.render_text(text, font, direction,
                                                overlap_intensity, ink_bleed_intensity,
                                                effect_type, effect_depth, light_azimuth, light_elevation)

        # Extract just the bbox coordinates
        char_bboxes = [box.bbox for box in char_boxes]

        # Apply augmentations
        augmented_image, augmented_bboxes = apply_augmentations(
            image, char_bboxes, self.background_images
        )

        # Apply canvas placement if enabled
        if canvas_enabled:
            from canvas_placement import place_on_canvas, generate_random_canvas_size

            # Generate canvas size if not provided
            if canvas_size is None:
                canvas_size = generate_random_canvas_size(
                    augmented_image.size,
                    min_padding=canvas_min_padding,
                    max_megapixels=canvas_max_megapixels
                )

            # Place on canvas
            final_image, metadata = place_on_canvas(
                augmented_image,
                augmented_bboxes,
                canvas_size=canvas_size,
                min_padding=canvas_min_padding,
                placement=canvas_placement,
                background_color=(255, 255, 255)
            )

            return final_image, metadata, text
        else:
            # Return without canvas placement (legacy format)
            metadata = {'char_bboxes': augmented_bboxes}
            return augmented_image, metadata, text


def setup_logging(log_level: str, log_file: str) -> None:
    """Configure logging with both file and console output."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )

    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def clear_output_directory(output_dir: str, force: bool = False) -> bool:
    """
    Clear the output directory after user confirmation.

    Args:
        output_dir: Directory to clear
        force: If True, skip confirmation prompt

    Returns:
        True if cleared or doesn't exist, False if user cancelled
    """
    if not os.path.exists(output_dir):
        logging.info(f"Output directory {output_dir} does not exist. Nothing to clear.")
        return True

    if not force:
        response = input(f"Are you sure you want to clear the output directory at {output_dir}? [y/N] ")
        if response.lower() != 'y':
            logging.info("Aborting.")
            return False

    logging.info(f"Clearing output directory: {output_dir}")
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')

    return True


def extract_sample_characters(text: str, max_samples: int = 100) -> str:
    """
    Extract a sample of unique characters from the text corpus.

    Args:
        text: Text corpus to sample from
        max_samples: Maximum number of unique characters to extract

    Returns:
        String containing unique sample characters
    """
    # Get unique characters, preserving order
    seen = set()
    unique_chars = []
    for char in text:
        if char not in seen and not char.isspace():
            seen.add(char)
            unique_chars.append(char)
            if len(unique_chars) >= max_samples:
                break

    return ''.join(unique_chars)


import functools

@functools.lru_cache(maxsize=None)
def can_font_render_text(font_path, text, character_set):
    try:
        font = ImageFont.truetype(font_path, size=24)
        for char in text:
            if char not in character_set:
                return False
        return True
    except Exception as e:
        logging.warning(f"Skipping font {os.path.basename(font_path)} due to error: {e}")
        return False


def generate_with_batches(batch_config, font_files, background_images, args):
    """
    Generate images using batch configuration.

    Args:
        batch_config: BatchConfig object with specifications
        font_files: List of validated font paths
        background_images: List of background image paths
        args: Command line arguments
    """
    from batch_config import BatchManager

    # Initialize batch manager
    batch_manager = BatchManager(batch_config, font_files)

    # Prepare output
    existing_images = [f for f in os.listdir(args.output_dir)
                      if f.startswith('image_') and f.endswith('.png')]
    image_counter = len(existing_images)

    # Track corpora per batch
    batch_corpora = {}

    logging.info(f"Starting batch generation of {batch_config.total_images} images")

    # Interleaved generation
    while True:
        task = batch_manager.get_next_task()
        if task is None:
            break

        # Load corpus for this batch if not already loaded
        batch_name = task['batch_name']
        if batch_name not in batch_corpora:
            corpus_file = task['corpus_file'] or args.text_file
            with open(corpus_file, 'r') as cf:
                batch_corpora[batch_name] = cf.read().strip()

        corpus = batch_corpora[batch_name]

        # Validate corpus for this batch's fonts
        sample_chars = extract_sample_characters(corpus, max_samples=100)
        font_path = task['font_path']

        # Initialize generator with task-specific background images
        generator = OCRDataGenerator([font_path], background_images)

        # Extract text
        text_line = generator.extract_text_segment(
            corpus, task['min_text_length'], task['max_text_length']
        )

        if not text_line:
            logging.warning(f"Could not generate text for batch '{batch_name}'. Skipping.")
            continue

        # Check if font can render this corpus
        if can_font_render_text(task['font_path'], text_line, frozenset(corpus)):
            # Generate font size
            font_size = random.randint(28, 40)

            try:
                # Generate image with augmentations and canvas placement
                final_image, metadata, text = generator.generate_image(
                    text_line, font_path, font_size, task['text_direction'],
                    curve_type=task.get('curve_type', 'none'),
                    curve_intensity=task.get('curve_intensity', 0.0),
                    overlap_intensity=task.get('overlap_intensity', 0.0),
                    ink_bleed_intensity=task.get('ink_bleed_intensity', 0.0),
                    effect_type=task.get('effect_type', 'none'),
                    effect_depth=task.get('effect_depth', 0.5),
                    light_azimuth=task.get('light_azimuth', 135.0),
                    light_elevation=task.get('light_elevation', 45.0),
                    text_color_mode=task.get('text_color_mode', 'uniform'),
                    color_palette=task.get('color_palette', 'realistic_dark'),
                    custom_colors=task.get('custom_colors'),
                    background_color=task.get('background_color', 'auto'),
                    canvas_enabled=True,
                    canvas_min_padding=task.get('canvas_min_padding', 10),
                    canvas_placement=task.get('canvas_placement', 'weighted_random'),
                    canvas_max_megapixels=task.get('canvas_max_megapixels', 12.0)
                )

                # Save image
                image_filename = f'image_{image_counter:05d}.png'
                image_path = os.path.join(args.output_dir, image_filename)
                final_image.save(image_path)

                # Save JSON label
                from canvas_placement import save_label_json
                json_filename = f'image_{image_counter:05d}.json'
                json_path = os.path.join(args.output_dir, json_filename)
                save_label_json(json_path, image_filename, text, metadata)

                image_counter += 1

                logging.debug(f"Batch '{batch_name}' ({task['progress']}): "
                            f"{os.path.basename(font_path)}, direction={task['text_direction']}")

            except OSError as e:
                if "execution context too long" in str(e):
                    logging.warning(f"Skipping font {font_path} due to FreeType error: {e}")
                    continue
                else:
                    logging.error(f"Failed to generate image for batch '{batch_name}': {e}")
                    continue
            except Exception as e:
                logging.error(f"Failed to generate image for batch '{batch_name}': {e}")
                continue

    logging.info(f"\n{batch_manager.get_progress_summary()}")
    logging.info(f"Successfully generated {image_counter} images in {args.output_dir}")


def main():
    """Main entry point for OCR data generation."""

    # --- Configuration Loading ---
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Synthetic Data Foundry for OCR')
    parser.add_argument('--text-file', type=str, default=config.get('text_file'),
                       help='Path to the text corpus file.')
    parser.add_argument('--fonts-dir', type=str, default=config.get('fonts_dir'),
                       help='Path to the directory containing font files.')
    parser.add_argument('--output-dir', type=str, default=config.get('output_dir'),
                       help='Path to the directory to save the generated images and labels.')
    parser.add_argument('--backgrounds-dir', type=str, default=config.get('backgrounds_dir'),
                       help='Optional: Path to a directory of background images.')
    parser.add_argument('--num-images', type=int, default=config.get('num_images', 1000),
                       help='Number of images to generate.')
    parser.add_argument('--max-execution-time', type=float, default=config.get('max_execution_time'),
                       help='Optional: Maximum execution time in seconds.')
    parser.add_argument('--min-text-length', type=int, default=config.get('min_text_length', 1),
                       help='Minimum length of text to generate.')
    parser.add_argument('--max-text-length', type=int, default=config.get('max_text_length', 100),
                       help='Maximum length of text to generate.')
    parser.add_argument('--text-direction', type=str, default=config.get('text_direction', 'left_to_right'),
                       choices=['left_to_right', 'top_to_bottom', 'right_to_left', 'bottom_to_top'],
                       help='Direction of the text.')
    parser.add_argument('--log-level', type=str, default=config.get('log_level', 'INFO'),
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set the logging level.')
    parser.add_argument('--log-file', type=str, default=config.get('log_file', 'generation.log'),
                       help='Path to the log file.')
    parser.add_argument('--clear-output', action='store_true',
                       help='If set, clears the output directory before generating new images.')
    parser.add_argument('--force', action='store_true',
                       help='If set, bypasses the confirmation prompt when clearing the output directory.')
    parser.add_argument('--font-name', type=str, default=None,
                       help='Name of the font file to use.')
    parser.add_argument('--batch-config', type=str, default=None,
                       help='Path to YAML batch configuration file for proportional generation.')
    parser.add_argument('--overlap-intensity', type=float, default=0.0,
                       help='Glyph overlap intensity (0.0-1.0). Higher values increase character overlap.')
    parser.add_argument('--ink-bleed-intensity', type=float, default=0.0,
                       help='Ink bleed effect intensity (0.0-1.0). Simulates document scanning artifacts.')
    parser.add_argument('--effect-type', type=str, default='none',
                       choices=['none', 'raised', 'embossed', 'engraved'],
                       help='3D text effect type. Options: none (default), raised (drop shadow), embossed (raised with highlights), engraved (carved/debossed).')
    parser.add_argument('--effect-depth', type=float, default=0.5,
                       help='3D effect depth intensity (0.0-1.0). Higher values create more pronounced effects.')
    parser.add_argument('--light-azimuth', type=float, default=135.0,
                       help='Light direction angle in degrees (0-360). 0=top, 90=right, 180=bottom, 270=left.')
    parser.add_argument('--light-elevation', type=float, default=45.0,
                       help='Light elevation angle in degrees (0-90). Lower values create longer shadows.')
    parser.add_argument('--text-color-mode', type=str, default='uniform',
                       choices=['uniform', 'per_glyph', 'gradient', 'random'],
                       help='Text color mode.')
    parser.add_argument('--color-palette', type=str, default='realistic_dark',
                       choices=['realistic_dark', 'realistic_light', 'vibrant', 'pastels'],
                       help='Color palette to use.')
    parser.add_argument('--custom-colors', type=str, help='Comma-separated list of custom RGB colors (e.g., \'255,0,0;0,255,0\').')
    parser.add_argument('--background-color', type=str, default='auto', help='Background color (e.g., \'255,255,255\' or \'auto\').')

    args = parser.parse_args()

    # --- Configure Logging ---
    setup_logging(args.log_level, args.log_file)
    logging.info("Script started.")

    # --- Clear Output Directory (if requested) ---
    if args.clear_output:
        if not clear_output_directory(args.output_dir, args.force):
            return

    # --- Validate Essential Arguments ---
    if not args.text_file:
        logging.error("Error: Text file not specified in config.json or command line.")
        sys.exit(1)
    if not args.fonts_dir or not os.path.isdir(args.fonts_dir):
        logging.error("Error: Fonts directory not specified or is not a valid directory.")
        sys.exit(1)
    if not args.output_dir:
        logging.error("Error: Output directory not specified in config.json or command line.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Load Assets ---
    logging.info("Loading assets...")

    # Load fonts
    # Optimization: if --font-name is specified, only validate that one font
    if args.font_name:
        font_path = os.path.join(args.fonts_dir, args.font_name)
        if not os.path.exists(font_path):
            logging.error(f"Specified font {args.font_name} not found in {args.fonts_dir}")
            sys.exit(1)
        font_candidates = [font_path]
    else:
        font_candidates = [os.path.join(args.fonts_dir, f)
                          for f in os.listdir(args.fonts_dir)
                          if f.endswith(('.ttf', '.otf'))]

    # Validate fonts by attempting to load them
    font_files = []
    for font_path in font_candidates:
        try:
            # Try to load the font to validate it
            ImageFont.truetype(font_path, size=20)
            font_files.append(font_path)
        except Exception as e:
            logging.warning(f"Skipping invalid font {os.path.basename(font_path)}: {e}")

    if not font_files:
        logging.error(f"No valid font files found in {args.fonts_dir}")
        sys.exit(1)
    logging.debug(f"Found {len(font_files)} valid font files.")

    # Load background images
    background_images = []
    if args.backgrounds_dir and os.path.exists(args.backgrounds_dir):
        background_images = [os.path.join(args.backgrounds_dir, f)
                           for f in os.listdir(args.backgrounds_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(background_images)} background images.")

    # Load text corpus
    with open(args.text_file, 'r') as text_file:
        corpus = text_file.read()

    # Strip whitespace and validate corpus
    corpus = corpus.strip()
    if not corpus or len(corpus) < args.min_text_length:
        logging.error(f"Corpus must contain at least {args.min_text_length} characters. Found: {len(corpus)}")
        sys.exit(1)
    logging.debug(f"Corpus length: {len(corpus)}")

    if args.max_text_length > len(corpus):
        args.max_text_length = len(corpus)

    # Extract sample characters from corpus for font validation
    sample_chars = extract_sample_characters(corpus, max_samples=100)
    logging.debug(f"Extracted {len(sample_chars)} unique characters for font validation")

    # Validate fonts against corpus characters
    if sample_chars and font_files:
        compatible_fonts = []
    font_paths = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir) if f.lower().endswith(('.ttf', '.otf'))]
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text_corpus = f.read()
    character_set = frozenset(text_corpus)

    if not font_paths:
        logging.error("No valid fonts found in the specified directory.")
        return

    # Filter fonts based on character set coverage
    compatible_fonts = []
    for font_path in font_paths:
        if can_font_render_text(font_path, text_corpus, character_set):
            compatible_fonts.append(font_path)

        if not compatible_fonts:
            logging.error(f"No fonts can render the corpus text. Found {len(font_files)} valid fonts but none support the required characters.")
            sys.exit(1)

        font_files = compatible_fonts
        logging.info(f"Found {len(font_files)} fonts compatible with corpus characters")

    logging.info("Script finished.")

    # --- Check for Batch Configuration ---
    if args.batch_config:
        from batch_config import BatchConfig, BatchManager

        logging.info(f"Loading batch configuration from {args.batch_config}")
        batch_config = BatchConfig.from_yaml(args.batch_config)

        # Use batch manager for generation
        generate_with_batches(batch_config, font_files, background_images, args)
        return

    # --- Initialize Generator ---
    generator = OCRDataGenerator(font_files, background_images)

    # --- Generation Loop ---
    if args.num_images > 0:
        start_time = time.time()

        # Determine starting image counter from existing images
        existing_images = [f for f in os.listdir(args.output_dir)
                         if f.startswith('image_') and f.endswith('.png')]
        image_counter = len(existing_images)

        logging.info(f"Generating up to {args.num_images} images (starting from image_{image_counter:05d})...")

        # Check if corpus has enough content
        if len(corpus) < args.min_text_length:
            logging.error(f"Corpus is too short (length: {len(corpus)}). Need at least {args.min_text_length} characters.")
            return

        for i in range(args.num_images):
            # --- Time Limit Check ---
            if args.max_execution_time and (time.time() - start_time) > args.max_execution_time:
                logging.info(f"\nTime limit of {args.max_execution_time} seconds reached. Stopping generation.")
                break

            # Extract text segment from corpus
            text_line = generator.extract_text_segment(
                corpus, args.min_text_length, args.max_text_length
            )

            if not text_line:
                logging.warning(f"Could not generate text of minimum length {args.min_text_length}. Skipping image.")
                continue

            logging.debug(f"Selected text: {text_line}")

            # Select font
            if args.font_name:
                font_path = os.path.join(args.fonts_dir, args.font_name)
                if not os.path.exists(font_path):
                    logging.error(f"Error: Font file {args.font_name} not found in {args.fonts_dir}")
                    continue
            else:
                font_path = random.choice(font_files)
            logging.debug(f"Selected font: {font_path}")

            # Generate font size
            font_size = random.randint(28, 40)

            # Parse custom colors
            custom_colors = None
            if args.custom_colors:
                try:
                    custom_colors = [
                        tuple(map(int, color.split(',')))
                        for color in args.custom_colors.split(';')
                    ]
                except ValueError:
                    logging.warning(f"Invalid format for --custom-colors: {args.custom_colors}")

            try:
                # Generate image with augmentations and canvas placement
                final_image, metadata, text = generator.generate_image(
                    text_line, font_path, font_size, args.text_direction,
                    overlap_intensity=args.overlap_intensity,
                    ink_bleed_intensity=args.ink_bleed_intensity,
                    effect_type=args.effect_type,
                    effect_depth=args.effect_depth,
                    light_azimuth=args.light_azimuth,
                    light_elevation=args.light_elevation,
                    text_color_mode=args.text_color_mode,
                    color_palette=args.color_palette,
                    custom_colors=custom_colors,
                    background_color=args.background_color,
                    canvas_enabled=True,
                    canvas_min_padding=10,
                    canvas_placement='weighted_random',
                    canvas_max_megapixels=12.0
                )

                # Save image
                image_filename = f'image_{image_counter:05d}.png'
                image_path = os.path.join(args.output_dir, image_filename)
                final_image.save(image_path)
                logging.debug(f"Saved image to {image_path}")

                # Save JSON label
                from canvas_placement import save_label_json
                json_filename = f'image_{image_counter:05d}.json'
                json_path = os.path.join(args.output_dir, json_filename)
                save_label_json(json_path, image_filename, text, metadata)
                logging.debug(f"Saved label to {json_path}")

                image_counter += 1

            except Exception as e:
                logging.error(f"Failed to generate image: {e}")
                continue

        logging.info(f"Successfully generated {image_counter} images with JSON labels in {args.output_dir}")


if __name__ == "__main__":
    main()
