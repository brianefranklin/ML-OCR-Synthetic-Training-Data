"""
OCR Data Generator - Main generation class

Generates synthetic text images with character-level bounding boxes for OCR training.
Supports multiple text directions: left-to-right, right-to-left, top-to-bottom, bottom-to-top.
"""

import random
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import bidi.algorithm

from augmentations import apply_augmentations


import numpy as np


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
                    char_width, overlap_intensity, enable_variation=False
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
                spacing = OverlapRenderer.calculate_overlap_spacing(
                    char_width, overlap_intensity, enable_variation=False
                )
                total_width += spacing

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

            # Apply overlap to spacing
            spacing = OverlapRenderer.calculate_overlap_spacing(
                char_width, overlap_intensity, enable_variation=True
            )
            x_offset -= spacing

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
                reduced_spacing = OverlapRenderer.calculate_vertical_overlap_spacing(
                    char_height, overlap_intensity, enable_variation=False
                )
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
                      seed: Optional[int] = None,
                      augmentations: Optional[Dict] = None,
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
                      canvas_max_megapixels: float = 12.0,
                      text_offset: Tuple[int, int] = None) -> Tuple[Image.Image, Dict, str]:
        """
        Generate a single synthetic OCR image with augmentations.

        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size in points
            direction: Text direction
            seed: Optional random seed for deterministic generation
            curve_type: Type of text curvature ('none', 'arc', 'sine')
            curve_intensity: Strength of curve (0.0-1.0)
            overlap_intensity: Glyph overlap intensity (0.0-1.0)
            ink_bleed_intensity: Ink bleed effect intensity (0.0-1.0)
            canvas_enabled: Whether to place text on larger canvas
            canvas_size: Fixed canvas size (if None, generates random size)
            canvas_min_padding: Minimum padding around text
            canvas_placement: Placement strategy ('weighted_random', 'uniform_random', 'center')
            canvas_max_megapixels: Maximum canvas size in megapixels
            text_offset: Explicit text placement (x, y) for deterministic regeneration

        Returns:
            Tuple of (final_image, metadata_dict, text)
            If canvas_enabled=False: metadata_dict = {'char_bboxes': [...]}
            If canvas_enabled=True: metadata_dict = {'canvas_size': [...], 'text_placement': [...], 'line_bbox': [...], 'char_bboxes': [...]}
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

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
                                                effect_type, effect_depth, light_azimuth, light_elevation,
                                                text_color_mode, color_palette, custom_colors, background_color)

        # Extract just the bbox coordinates
        char_bboxes = [box.bbox for box in char_boxes]

        # Apply augmentations
        augmented_image, augmented_bboxes, augmentations_applied = apply_augmentations(
            image, char_bboxes, self.background_images, augmentations_to_apply=augmentations
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
                background_color=(255, 255, 255),
                text_offset=text_offset
            )

            return final_image, metadata, text, augmentations_applied
        else:
            # Return without canvas placement (legacy format)
            metadata = {'char_bboxes': augmented_bboxes}
            return augmented_image, metadata, text, augmentations_applied


