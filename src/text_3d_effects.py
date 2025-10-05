"""
3D Text Effects Module for OCR Synthetic Data Generation

Implements realistic 3D text effects:
- Drop shadow (raised text)
- Embossed text (raised edges with highlights/shadows)
- Debossed/engraved text (carved into surface)
- Bevel effects (optional advanced implementation)

All effects are language-agnostic and work with any script.
"""

import math
import random
from typing import Tuple
from PIL import Image, ImageDraw, ImageFilter, ImageChops
import numpy as np


class Text3DEffects:
    """
    Handles 3D text effect rendering.

    Supports multiple effect types with configurable depth and lighting.
    """

    @staticmethod
    def apply_effect(image: Image.Image,
                    effect_type: str = 'none',
                    effect_depth: float = 0.5,
                    light_azimuth: float = 135.0,
                    light_elevation: float = 45.0) -> Image.Image:
        """
        Apply 3D effect to rendered text image.

        Args:
            image: Input image with text rendered
            effect_type: Type of effect ('none', 'raised', 'embossed', 'engraved')
            effect_depth: Depth intensity (0.0-1.0)
            light_azimuth: Light direction angle in degrees (0-360)
            light_elevation: Light elevation angle in degrees (0-90)

        Returns:
            Image with 3D effect applied
        """
        # Validate and clamp parameters
        effect_depth = max(0.0, min(1.0, effect_depth))

        # No effect or zero depth
        if effect_type == 'none' or effect_depth <= 0.0:
            return image

        # Route to appropriate effect implementation
        if effect_type == 'raised':
            return Text3DEffects._apply_drop_shadow(image, effect_depth, light_azimuth, light_elevation)
        elif effect_type == 'embossed':
            return Text3DEffects._apply_emboss(image, effect_depth, light_azimuth, light_elevation)
        elif effect_type == 'engraved':
            return Text3DEffects._apply_deboss(image, effect_depth, light_azimuth, light_elevation)
        else:
            # Unknown effect type, return original
            return image

    @staticmethod
    def _apply_drop_shadow(image: Image.Image,
                          depth: float,
                          azimuth: float,
                          elevation: float) -> Image.Image:
        """
        Apply drop shadow effect for raised text appearance.

        Args:
            image: Input image
            depth: Shadow depth (0.0-1.0)
            azimuth: Light angle in degrees (determines shadow direction)
            elevation: Light elevation angle (0-90 degrees, affects shadow length)

        Returns:
            Image with drop shadow
        """
        # Calculate shadow offset based on light direction
        # Azimuth: 0°=top, 90°=right, 180°=bottom, 270°=left
        # Elevation: lower angle = longer shadow
        angle_rad = math.radians(azimuth)
        elev_rad = math.radians(max(1, min(89, elevation)))  # Clamp to avoid division by zero

        # Shadow offset (opposite to light direction)
        # Shadow length increases as elevation decreases
        max_offset = 10  # Maximum shadow offset in pixels
        elevation_factor = 1.0 / math.tan(elev_rad)  # Lower elevation = longer shadow
        offset_distance = depth * max_offset * min(elevation_factor, 3.0)  # Cap at 3x

        shadow_x = int(offset_distance * math.sin(angle_rad))
        shadow_y = int(offset_distance * math.cos(angle_rad))

        # Create shadow layer
        # Extract alpha channel or create mask from non-white pixels
        if image.mode == 'RGBA':
            alpha = image.split()[3]
        else:
            # Create mask from non-white pixels
            img_array = np.array(image.convert('L'))
            mask = Image.fromarray((img_array < 250).astype(np.uint8) * 255)
            alpha = mask

        # Create shadow image (darker version)
        shadow_color = int(150 - (depth * 100))  # Darker with more depth
        shadow = Image.new('RGB', image.size, 'white')
        shadow_draw = ImageDraw.Draw(shadow)

        # Fill shadow with gray color using alpha mask
        shadow_layer = Image.new('RGBA', image.size, (shadow_color, shadow_color, shadow_color, 0))
        shadow_layer.putalpha(alpha)

        # Blur shadow for softness
        blur_radius = 1 + (depth * 3)  # More blur with more depth
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Calculate new image size to accommodate shadow
        new_width = image.width + abs(shadow_x) + 10
        new_height = image.height + abs(shadow_y) + 10

        # Create composite image with transparent background
        result = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))

        # Position shadow (offset from text)
        shadow_pos_x = max(0, -shadow_x) + 5
        shadow_pos_y = max(0, -shadow_y) + 5

        # Position text (on top of shadow)
        text_pos_x = max(0, shadow_x) + 5
        text_pos_y = max(0, shadow_y) + 5

        # Paste shadow first
        result.paste(shadow_layer, (shadow_pos_x, shadow_pos_y), shadow_layer)

        # Paste original text on top
        if image.mode == 'RGBA':
            result.paste(image, (text_pos_x, text_pos_y), image)
        else:
            # Convert to RGBA if needed
            rgba_image = image.convert('RGBA')
            result.paste(rgba_image, (text_pos_x, text_pos_y), rgba_image)

        return result

    @staticmethod
    def _apply_emboss(image: Image.Image,
                     depth: float,
                     azimuth: float,
                     elevation: float) -> Image.Image:
        """
        Apply embossed effect (raised text with highlights and shadows).

        Args:
            image: Input image
            depth: Effect depth (0.0-1.0)
            azimuth: Light angle in degrees (0-360)
            elevation: Light elevation angle (0-90)

        Returns:
            Image with emboss effect
        """
        # Convert to grayscale for edge detection
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        # Calculate light direction
        azimuth_rad = math.radians(azimuth)
        elevation_rad = math.radians(elevation)

        # Light direction offsets
        offset_distance = 2 + int(depth * 3)  # 2-5 pixels based on depth
        highlight_x = int(offset_distance * math.sin(azimuth_rad) * math.cos(elevation_rad))
        highlight_y = int(offset_distance * math.cos(azimuth_rad) * math.cos(elevation_rad))

        # Shadow is opposite direction
        shadow_x = -highlight_x
        shadow_y = -highlight_y

        # Create highlight layer (shifted in light direction)
        highlight = Image.new('RGBA', image.size, (255, 255, 255, 0))

        # Extract text mask
        img_array = np.array(gray)
        text_mask = (img_array < 250).astype(np.uint8) * 255
        mask_img = Image.fromarray(text_mask)

        # Create highlight by shifting mask
        highlight_shifted = Image.new('L', image.size, 0)
        paste_x = max(0, highlight_x)
        paste_y = max(0, highlight_y)
        highlight_shifted.paste(mask_img, (paste_x, paste_y))

        # Blur highlight
        highlight_shifted = highlight_shifted.filter(ImageFilter.GaussianBlur(radius=1.5))

        # Create shadow layer (shifted opposite direction)
        shadow_shifted = Image.new('L', image.size, 0)
        paste_x = max(0, shadow_x)
        paste_y = max(0, shadow_y)
        shadow_shifted.paste(mask_img, (paste_x, paste_y))

        # Blur shadow
        shadow_shifted = shadow_shifted.filter(ImageFilter.GaussianBlur(radius=1.5))

        # Combine layers - preserve RGBA
        if image.mode == 'RGBA':
            result = image.copy()
        else:
            result = image.copy().convert('RGBA')

        # Apply shadow (darken)
        # Shadow alpha is modulated by depth parameter
        shadow_alpha_data = (np.array(shadow_shifted) * depth).astype(np.uint8)
        # Create shadow layer with alpha
        shadow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_array = np.array(shadow_layer)
        shadow_array[:, :, 0:3] = 0  # Black shadow
        shadow_array[:, :, 3] = shadow_alpha_data
        shadow_layer = Image.fromarray(shadow_array)
        result = Image.alpha_composite(result, shadow_layer)

        # Apply highlight (lighten)
        # Highlight is typically softer than shadow (70% strength)
        highlight_alpha_data = (np.array(highlight_shifted) * depth * 0.7).astype(np.uint8)
        highlight_layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
        highlight_array = np.array(highlight_layer)
        highlight_array[:, :, 0:3] = 255  # White highlight
        highlight_array[:, :, 3] = highlight_alpha_data
        highlight_layer = Image.fromarray(highlight_array)
        result = Image.alpha_composite(result, highlight_layer)

        # Paste original text on top for crispness
        if image.mode == 'RGBA':
            text_layer = image.copy()
        else:
            text_layer = image.convert('RGBA')
        result = Image.alpha_composite(result, text_layer)

        return result

    @staticmethod
    def _apply_deboss(image: Image.Image,
                     depth: float,
                     azimuth: float,
                     elevation: float) -> Image.Image:
        """
        Apply debossed/engraved effect (text carved into surface).

        Args:
            image: Input image
            depth: Effect depth (0.0-1.0)
            azimuth: Light angle in degrees (0-360)
            elevation: Light elevation angle (0-90)

        Returns:
            Image with deboss/engraved effect
        """
        # Deboss is essentially inverted emboss
        # We invert the light direction to create carved-in appearance

        # Invert azimuth (opposite light direction)
        inverted_azimuth = (azimuth + 180) % 360

        # Apply emboss with inverted direction
        result = Text3DEffects._apply_emboss(image, depth, inverted_azimuth, elevation)

        # For engraved effect, we want the text area to appear recessed
        # The text should be darker to show it's carved in
        # (The emboss already added highlights/shadows, we just need to darken the carved area)

        return result

    @staticmethod
    def calculate_bbox_adjustment(effect_type: str,
                                  effect_depth: float,
                                  light_azimuth: float,
                                  original_bbox: list) -> list:
        """
        Adjust bounding box to account for 3D effect expansion.

        Args:
            effect_type: Type of effect
            effect_depth: Depth intensity
            light_azimuth: Light angle
            original_bbox: Original bbox [x0, y0, x1, y1]

        Returns:
            Adjusted bbox [x0, y0, x1, y1]
        """
        if effect_type == 'none' or effect_depth <= 0.0:
            return original_bbox

        x0, y0, x1, y1 = original_bbox

        if effect_type == 'raised':
            # Drop shadow expands bbox
            max_offset = 10
            offset_distance = effect_depth * max_offset
            angle_rad = math.radians(light_azimuth)

            shadow_x = int(offset_distance * math.sin(angle_rad))
            shadow_y = int(offset_distance * math.cos(angle_rad))

            # Expand bbox to include shadow
            padding = 5 + int(offset_distance)
            return [
                x0 - padding,
                y0 - padding,
                x1 + padding,
                y1 + padding
            ]

        elif effect_type in ['embossed', 'engraved']:
            # Emboss/deboss adds slight padding for highlights/shadows
            padding = 3 + int(depth * 3)
            return [
                x0 - padding,
                y0 - padding,
                x1 + padding,
                y1 + padding
            ]

        return original_bbox

    @staticmethod
    def should_apply_effect(effect_type: str, effect_depth: float) -> bool:
        """
        Determine if effect should be applied.

        Args:
            effect_type: Type of effect
            effect_depth: Depth intensity

        Returns:
            True if effect should be applied
        """
        return effect_type != 'none' and effect_depth > 0.0


# Utility functions for integration
def add_3d_effect_to_image(image: Image.Image,
                           effect_type: str = 'none',
                           effect_depth: float = 0.5,
                           light_azimuth: float = 135.0,
                           light_elevation: float = 45.0) -> Image.Image:
    """
    Convenience function to apply 3D effect to an image.

    Args:
        image: Input image
        effect_type: Effect type ('none', 'raised', 'embossed', 'engraved')
        effect_depth: Depth intensity (0.0-1.0)
        light_azimuth: Light angle (0-360 degrees)
        light_elevation: Light elevation (0-90 degrees)

    Returns:
        Image with effect applied
    """
    return Text3DEffects.apply_effect(
        image, effect_type, effect_depth, light_azimuth, light_elevation
    )
