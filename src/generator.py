from PIL import Image, ImageDraw, ImageFont
import bidi.algorithm
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from src.canvas_placement import (
    generate_random_canvas_size,
    calculate_text_placement,
    place_on_canvas,
)
from src.effects import (
    apply_ink_bleed, 
    apply_drop_shadow, 
    add_noise, 
    apply_blur, 
    apply_brightness_contrast,
    apply_erosion_dilation,
    apply_block_shadow,
    apply_cutout
)
from src.augmentations import (
    apply_rotation, 
    apply_perspective_warp, 
    apply_elastic_distortion,
    apply_grid_distortion,
    apply_optical_distortion
)
from src.batch_config import BatchSpecification
from src.background_manager import BackgroundImageManager

class OCRDataGenerator:
    """Orchestrates the entire image generation pipeline.
    
    This class brings together all the components of the generation process,
    from planning the parameters to rendering text, applying effects, and augmenting
    the final image.
    """

    def __init__(self):
        """Initializes the OCRDataGenerator."""
        pass

    def plan_generation(
        self, 
        spec: BatchSpecification, 
        text: str, 
        font_path: str, 
        background_manager: Optional[BackgroundImageManager] = None
    ) -> Dict[str, Any]:
        """Creates a plan (a dictionary of truth data) for generating an image.

        Args:
            spec: The BatchSpecification defining the parameters for this generation.
            text: The text string to be rendered.
            font_path: The path to the font file to use.
            background_manager: An optional manager for selecting background images.

        Returns:
            A dictionary containing the complete plan for generating a single image.
        """
        # Render a temporary surface to get the dimensions for canvas calculation.
        # Use straight text for canvas sizing regardless of curve type
        text_surface, _ = self._render_text(text, font_path, spec.text_direction, 0.0, 'uniform', None, "none", 0.0, True, 0.0, 0.0, 0.0)
        
        canvas_w, canvas_h = generate_random_canvas_size(text_surface.width, text_surface.height)
        placement_x, placement_y = calculate_text_placement(
            canvas_w, canvas_h, text_surface.width, text_surface.height, "uniform_random"
        )

        background_path = background_manager.select_background() if background_manager else None

        # Build the final plan dictionary
        return {
            "text": text,
            "font_path": font_path,
            "direction": spec.text_direction,
            "seed": random.randint(0, 2**32 - 1),
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "placement_x": placement_x,
            "placement_y": placement_y,
            "glyph_overlap_intensity": random.uniform(spec.glyph_overlap_intensity_min, spec.glyph_overlap_intensity_max),
            "ink_bleed_radius": random.uniform(spec.ink_bleed_radius_min, spec.ink_bleed_radius_max),
            "drop_shadow_options": None, # Placeholder for more complex options
            "block_shadow_options": None, # Placeholder
            "color_mode": 'uniform', # Placeholder
            "color_palette": None, # Placeholder
            "rotation_angle": random.uniform(spec.rotation_angle_min, spec.rotation_angle_max),
            "perspective_warp_magnitude": random.uniform(spec.perspective_warp_magnitude_min, spec.perspective_warp_magnitude_max),
            "elastic_distortion_options": {
                "alpha": random.uniform(spec.elastic_distortion_alpha_min, spec.elastic_distortion_alpha_max),
                "sigma": random.uniform(spec.elastic_distortion_sigma_min, spec.elastic_distortion_sigma_max)
            },
            "grid_distortion_options": {
                "num_steps": random.randint(spec.grid_distortion_steps_min, spec.grid_distortion_steps_max),
                "distort_limit": random.randint(spec.grid_distortion_limit_min, spec.grid_distortion_limit_max)
            },
            "optical_distortion_options": {
                "distort_limit": random.uniform(spec.optical_distortion_limit_min, spec.optical_distortion_limit_max)
            },
            "cutout_options": {
                "cutout_size": (
                    random.randint(spec.cutout_width_min, spec.cutout_width_max),
                    random.randint(spec.cutout_height_min, spec.cutout_height_max)
                )
            },
            "noise_amount": random.uniform(spec.noise_amount_min, spec.noise_amount_max),
            "blur_radius": random.uniform(spec.blur_radius_min, spec.blur_radius_max),
            "brightness_factor": random.uniform(spec.brightness_factor_min, spec.brightness_factor_max),
            "contrast_factor": random.uniform(spec.contrast_factor_min, spec.contrast_factor_max),
            "erosion_dilation_options": {
                "mode": random.choice(['erode', 'dilate']),
                "kernel_size": random.randint(spec.erosion_dilation_kernel_min, spec.erosion_dilation_kernel_max)
            },
            "background_path": background_path,
            # Curve parameters - always included for consistent ML feature vectors
            "curve_type": spec.curve_type,
            "arc_radius": random.uniform(spec.arc_radius_min, spec.arc_radius_max),
            "arc_concave": spec.arc_concave,
            "sine_amplitude": random.uniform(spec.sine_amplitude_min, spec.sine_amplitude_max),
            "sine_frequency": random.uniform(spec.sine_frequency_min, spec.sine_frequency_max),
            "sine_phase": random.uniform(spec.sine_phase_min, spec.sine_phase_max),
        }

    def generate_from_plan(self, plan: Dict[str, Any]) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Generates an image deterministically from a plan dictionary.

        Args:
            plan: The dictionary containing all parameters for generation.

        Returns:
            A tuple containing the final generated PIL Image and a list of
            bounding box dictionaries.
        """
        # Seeding the random number generator is the key to deterministic output.
        random.seed(plan["seed"])

        # 1. Render the basic text surface with all text-level effects.
        text_surface, bboxes = self._render_text(
            plan["text"],
            plan["font_path"],
            plan["direction"],
            plan.get("glyph_overlap_intensity", 0.0),
            plan.get("color_mode", 'uniform'),
            plan.get("color_palette"),
            plan.get("curve_type", "none"),
            plan.get("arc_radius", 0.0),
            plan.get("arc_concave", True),
            plan.get("sine_amplitude", 0.0),
            plan.get("sine_frequency", 0.0),
            plan.get("sine_phase", 0.0),
        )

        # 2. Apply post-rendering text effects.
        ink_bleed_radius = plan.get("ink_bleed_radius", 0.0)
        if ink_bleed_radius > 0:
            text_surface = apply_ink_bleed(text_surface, ink_bleed_radius)
        
        drop_shadow_options = plan.get("drop_shadow_options")
        if drop_shadow_options:
            text_surface = apply_drop_shadow(text_surface, **drop_shadow_options)

        block_shadow_options = plan.get("block_shadow_options")
        if block_shadow_options:
            text_surface = apply_block_shadow(text_surface, **block_shadow_options)

        # 3. Place the text surface onto the final canvas.
        final_image, final_bboxes = place_on_canvas(
            text_image=text_surface,
            canvas_w=plan["canvas_w"],
            canvas_h=plan["canvas_h"],
            placement_x=plan["placement_x"],
            placement_y=plan["placement_y"],
            original_bboxes=bboxes,
            background_path=plan.get("background_path"),
        )

        # 4. Apply final image-level augmentations.
        rotation_angle = plan.get("rotation_angle", 0.0)
        if rotation_angle != 0.0:
            final_image, final_bboxes = apply_rotation(final_image, final_bboxes, rotation_angle)
        
        warp_magnitude = plan.get("perspective_warp_magnitude", 0.0)
        if warp_magnitude > 0.0:
            final_image, final_bboxes = apply_perspective_warp(final_image, final_bboxes, warp_magnitude)

        elastic_options = plan.get("elastic_distortion_options")
        if elastic_options and elastic_options['alpha'] > 0 and elastic_options['sigma'] > 0:
            final_image, final_bboxes = apply_elastic_distortion(final_image, final_bboxes, **elastic_options)

        grid_distortion_options = plan.get("grid_distortion_options")
        if grid_distortion_options and grid_distortion_options['distort_limit'] > 0:
            final_image, final_bboxes = apply_grid_distortion(final_image, final_bboxes, **grid_distortion_options)

        optical_distortion_options = plan.get("optical_distortion_options")
        if optical_distortion_options and optical_distortion_options['distort_limit'] > 0:
            final_image, final_bboxes = apply_optical_distortion(final_image, final_bboxes, **optical_distortion_options)

        cutout_options = plan.get("cutout_options")
        if cutout_options and cutout_options['cutout_size'][0] > 0 and cutout_options['cutout_size'][1] > 0:
            final_image = apply_cutout(final_image, **cutout_options)

        erosion_dilation_options = plan.get("erosion_dilation_options")
        if erosion_dilation_options and erosion_dilation_options['kernel_size'] > 1:
            final_image = apply_erosion_dilation(final_image, **erosion_dilation_options)

        noise_amount = plan.get("noise_amount", 0.0)
        if noise_amount > 0.0:
            final_image = add_noise(final_image, noise_amount)

        blur_radius = plan.get("blur_radius", 0.0)
        if blur_radius > 0.0:
            final_image = apply_blur(final_image, blur_radius)

        brightness_factor = plan.get("brightness_factor", 1.0)
        contrast_factor = plan.get("contrast_factor", 1.0)
        if brightness_factor != 1.0 or contrast_factor != 1.0:
            final_image = apply_brightness_contrast(final_image, brightness_factor, contrast_factor)

        return final_image, final_bboxes

    def _render_text(
        self,
        text: str,
        font_path: str,
        direction: str,
        glyph_overlap_intensity: float = 0.0,
        color_mode: str = 'uniform',
        color_palette: Optional[list] = None,
        curve_type: str = "none",
        arc_radius: float = 0.0,
        arc_concave: bool = True,
        sine_amplitude: float = 0.0,
        sine_frequency: float = 0.0,
        sine_phase: float = 0.0
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Internal dispatcher for rendering text surfaces.

        Args:
            text: The text string to render.
            font_path: Path to the font file.
            direction: Text direction ("left_to_right", "right_to_left", etc.).
            glyph_overlap_intensity: Intensity of character overlap.
            color_mode: Color mode for rendering.
            color_palette: Optional color palette.
            curve_type: Type of curve ("none", "arc", "sine").
            arc_radius: Radius for arc curves (0.0 = straight).
            arc_concave: Whether arc curves concavely.
            sine_amplitude: Amplitude for sine wave (0.0 = straight).
            sine_frequency: Frequency for sine wave.
            sine_phase: Phase offset for sine wave.

        Returns:
            Tuple of (image, bboxes).
        """
        # Check if we should apply curves
        # Use a threshold to avoid numerical issues with very small values
        if curve_type == "arc" and arc_radius > 1.0:  # Changed from > 0 to > 1.0
            return self._render_arc_text(
                text, font_path, direction, arc_radius, arc_concave,
                glyph_overlap_intensity, color_mode, color_palette
            )
        elif curve_type == "sine" and sine_amplitude > 0.1:  # Small threshold
            return self._render_sine_text(
                text, font_path, direction, sine_amplitude, sine_frequency, sine_phase,
                glyph_overlap_intensity, color_mode, color_palette
            )

        # Fall back to straight text rendering
        if direction == "left_to_right":
            return self._render_left_to_right(text, font_path, glyph_overlap_intensity, color_mode, color_palette)
        elif direction == "right_to_left":
            return self._render_right_to_left(text, font_path, glyph_overlap_intensity, color_mode, color_palette)
        elif direction == "top_to_bottom":
            return self._render_top_to_bottom(text, font_path, glyph_overlap_intensity, color_mode, color_palette)
        elif direction == "bottom_to_top":
            return self._render_bottom_to_top(text, font_path, glyph_overlap_intensity, color_mode, color_palette)
        else:
            raise ValueError(f"Unsupported text direction: {direction}")

    def _render_arc_text(
        self,
        text: str,
        font_path: str,
        direction: str,
        arc_radius: float,
        arc_concave: bool,
        glyph_overlap_intensity: float,
        color_mode: str,
        color_palette: Optional[list]
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Renders text along a circular arc.

        Args:
            text: The text string to render.
            font_path: Path to the font file.
            direction: Base text direction ("left_to_right", etc.).
            arc_radius: Radius of the arc circle.
            arc_concave: True for concave (curves "inward"), False for convex.
            glyph_overlap_intensity: Intensity of character overlap.
            color_mode: Color mode for rendering.
            color_palette: Optional color palette.

        Returns:
            Tuple of (image, bboxes).
        """
        import math

        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()
        char_height = ascent + descent

        # Measure all characters
        char_widths = []
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
            except AttributeError:
                char_width, _ = font.getsize(char)
            char_widths.append(char_width)

        # Calculate total arc length needed
        total_arc_length = sum(w * (1 - glyph_overlap_intensity) for w in char_widths)

        # Calculate arc span in radians
        arc_span_radians = total_arc_length / arc_radius if arc_radius > 0 else 0

        # Estimate canvas size (arc bounding box + margin)
        margin = 50
        canvas_size = int(arc_radius * 2 + char_height + margin * 2)
        image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

        # Calculate center point
        center_x = canvas_size / 2
        center_y = canvas_size / 2

        # Starting angle depends on direction and concavity
        if direction in ["left_to_right", "right_to_left"]:
            # For LTR concave, start at bottom-left, curve upward
            if arc_concave:
                start_angle = math.pi / 2 + arc_span_radians / 2  # Start from left side
            else:  # convex
                start_angle = -math.pi / 2 - arc_span_radians / 2  # Start from top
        else:  # vertical text
            if arc_concave:
                start_angle = math.pi  # Start from left
            else:
                start_angle = 0  # Start from right

        bboxes = []
        current_arc_length = 0

        # Process text in reverse for RTL/BTT
        chars_to_render = text
        if direction in ["right_to_left", "bottom_to_top"]:
            chars_to_render = text[::-1]

        for i, char in enumerate(chars_to_render):
            # Get original index for color/palette
            original_idx = i if direction in ["left_to_right", "top_to_bottom"] else len(text) - 1 - i

            char_width = char_widths[original_idx]

            # Calculate position along arc (at character center)
            arc_progress = current_arc_length + (char_width / 2)
            angle = start_angle - (arc_progress / arc_radius)  # Negative for counter-clockwise

            # Calculate character position on circle
            char_x = center_x + arc_radius * math.cos(angle)
            char_y = center_y - arc_radius * math.sin(angle)  # Negative because Y increases downward

            # Calculate rotation angle (tangent to circle)
            rotation_angle = -(angle * 180 / math.pi + 90)  # Convert to degrees, adjust for tangent

            # Apply concave/convex adjustment
            if not arc_concave:
                rotation_angle += 180

            # Determine fill color
            fill = "black"
            if color_mode == 'per_glyph' and color_palette:
                fill = color_palette[original_idx]
            elif color_mode == 'gradient' and color_palette:
                t = original_idx / (len(text) - 1) if len(text) > 1 else 0
                start_color = np.array(color_palette[0])
                end_color = np.array(color_palette[1])
                fill = tuple((start_color + t * (end_color - start_color)).astype(int))

            # Render character on temporary surface
            char_surface = Image.new("RGBA", (char_width * 3, char_height * 3), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_surface)
            char_draw.text((char_width, char_height), char, font=font, fill=fill)

            # Rotate character
            rotated_char = char_surface.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)

            # Calculate paste position (center the rotated character)
            paste_x = int(char_x - rotated_char.width / 2)
            paste_y = int(char_y - rotated_char.height / 2)

            # Paste onto main image
            image.paste(rotated_char, (paste_x, paste_y), rotated_char)

            # Calculate bounding box by transforming original bbox corners
            original_corners = np.array([
                [-char_width/2, -char_height/2],
                [char_width/2, -char_height/2],
                [char_width/2, char_height/2],
                [-char_width/2, char_height/2]
            ])

            # Rotate corners
            angle_rad = rotation_angle * math.pi / 180
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = original_corners @ rotation_matrix.T

            # Translate to character position
            rotated_corners[:, 0] += char_x
            rotated_corners[:, 1] += char_y

            # Find axis-aligned bounding box
            min_x = int(np.min(rotated_corners[:, 0]))
            max_x = int(np.max(rotated_corners[:, 0]))
            min_y = int(np.min(rotated_corners[:, 1]))
            max_y = int(np.max(rotated_corners[:, 1]))

            bboxes.append({
                "char": text[original_idx],  # Original character
                "x0": max(0, min_x),
                "y0": max(0, min_y),
                "x1": min(canvas_size, max_x),
                "y1": min(canvas_size, max_y)
            })

            # Advance arc position
            current_arc_length += char_width * (1 - glyph_overlap_intensity)

        # Ensure bboxes are in original text order
        if direction in ["right_to_left", "bottom_to_top"]:
            bboxes = bboxes[::-1]

        # Crop the image to its actual content bounds
        # Find the overall bounding box of all text
        if bboxes:
            min_x = min(bbox["x0"] for bbox in bboxes)
            min_y = min(bbox["y0"] for bbox in bboxes)
            max_x = max(bbox["x1"] for bbox in bboxes)
            max_y = max(bbox["y1"] for bbox in bboxes)

            # Add a small margin around the content
            crop_margin = 10
            crop_x0 = max(0, min_x - crop_margin)
            crop_y0 = max(0, min_y - crop_margin)
            crop_x1 = min(canvas_size, max_x + crop_margin)
            crop_y1 = min(canvas_size, max_y + crop_margin)

            # Crop the image
            image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            # Adjust all bboxes to the cropped coordinate system
            adjusted_bboxes = []
            for bbox in bboxes:
                adjusted_bbox = bbox.copy()
                adjusted_bbox["x0"] = bbox["x0"] - crop_x0
                adjusted_bbox["y0"] = bbox["y0"] - crop_y0
                adjusted_bbox["x1"] = bbox["x1"] - crop_x0
                adjusted_bbox["y1"] = bbox["y1"] - crop_y0
                adjusted_bboxes.append(adjusted_bbox)

            return image, adjusted_bboxes

        return image, bboxes

    def _render_sine_text(
        self,
        text: str,
        font_path: str,
        direction: str,
        sine_amplitude: float,
        sine_frequency: float,
        sine_phase: float,
        glyph_overlap_intensity: float,
        color_mode: str,
        color_palette: Optional[list]
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Renders text along a sine wave.

        Args:
            text: The text string to render.
            font_path: Path to the font file.
            direction: Base text direction ("left_to_right", etc.).
            sine_amplitude: Height of wave oscillation.
            sine_frequency: Frequency of oscillation (cycles per pixel).
            sine_phase: Phase offset in radians.
            glyph_overlap_intensity: Intensity of character overlap.
            color_mode: Color mode for rendering.
            color_palette: Optional color palette.

        Returns:
            Tuple of (image, bboxes).
        """
        import math

        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()
        char_height = ascent + descent

        # Measure all characters
        char_widths = []
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
            except AttributeError:
                char_width, _ = font.getsize(char)
            char_widths.append(char_width)

        # Calculate total text width
        total_text_width = sum(w * (1 - glyph_overlap_intensity) for w in char_widths)

        # For horizontal text (LTR/RTL), wave affects Y position
        # For vertical text (TTB/BTT), wave affects X position
        is_horizontal = direction in ["left_to_right", "right_to_left"]

        # Canvas size: text dimensions + extra space for wave oscillation
        margin = 20
        wave_buffer = int(sine_amplitude * 2) + margin

        if is_horizontal:
            canvas_w = int(total_text_width + margin * 2)
            canvas_h = char_height + wave_buffer * 2
        else:  # vertical text
            canvas_w = max(char_widths) + wave_buffer * 2 if char_widths else 100
            canvas_h = int(total_text_width + margin * 2)

        image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        bboxes = []
        current_position = 0  # position along the primary text direction

        # Process text in reverse for RTL/BTT
        chars_to_render = text
        if direction in ["right_to_left", "bottom_to_top"]:
            chars_to_render = text[::-1]

        for i, char in enumerate(chars_to_render):
            # Get original index for color/palette
            original_idx = i if direction in ["left_to_right", "top_to_bottom"] else len(text) - 1 - i
            char_width = char_widths[original_idx]

            # Calculate character center position
            char_center = current_position + (char_width / 2)

            # Calculate sine wave offset for this position
            wave_offset = sine_amplitude * math.sin(sine_frequency * char_center + sine_phase)

            # Calculate tangent angle for rotation (derivative of sine wave)
            tangent_slope = sine_amplitude * sine_frequency * math.cos(sine_frequency * char_center + sine_phase)
            rotation_angle = math.atan(tangent_slope) * 180 / math.pi

            # Determine fill color
            fill = "black"
            if color_mode == 'per_glyph' and color_palette:
                fill = color_palette[original_idx]
            elif color_mode == 'gradient' and color_palette:
                t = original_idx / (len(text) - 1) if len(text) > 1 else 0
                start_color = np.array(color_palette[0])
                end_color = np.array(color_palette[1])
                fill = tuple((start_color + t * (end_color - start_color)).astype(int))

            # Render character on temporary surface
            char_surface = Image.new("RGBA", (char_width * 3, char_height * 3), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_surface)
            char_draw.text((char_width, char_height), char, font=font, fill=fill)

            # Rotate character according to wave tangent
            rotated_char = char_surface.rotate(-rotation_angle, resample=Image.BICUBIC, expand=True)

            # Calculate paste position based on direction
            if is_horizontal:
                # Horizontal text: X follows text, Y oscillates with wave
                char_x = int(margin + current_position)
                char_y = int(canvas_h / 2 - char_height / 2 + wave_offset)
                paste_x = char_x - int((rotated_char.width - char_width) / 2)
                paste_y = char_y - int((rotated_char.height - char_height) / 2)
            else:
                # Vertical text: Y follows text, X oscillates with wave
                char_x = int(canvas_w / 2 - max(char_widths) / 2 + wave_offset)
                char_y = int(margin + current_position)
                paste_x = char_x - int((rotated_char.width - char_width) / 2)
                paste_y = char_y - int((rotated_char.height - char_height) / 2)

            # Paste rotated character
            image.paste(rotated_char, (paste_x, paste_y), rotated_char)

            # Calculate bbox by transforming corners
            original_corners = np.array([
                [-char_width/2, -char_height/2],
                [char_width/2, -char_height/2],
                [char_width/2, char_height/2],
                [-char_width/2, char_height/2]
            ])

            # Rotate corners
            angle_rad = -rotation_angle * math.pi / 180
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = original_corners @ rotation_matrix.T

            # Translate to character position (use character center)
            if is_horizontal:
                center_x = char_x + char_width / 2
                center_y = char_y + char_height / 2
            else:
                center_x = char_x + char_width / 2
                center_y = char_y + char_height / 2

            rotated_corners[:, 0] += center_x
            rotated_corners[:, 1] += center_y

            # Find axis-aligned bounding box
            min_x = int(np.min(rotated_corners[:, 0]))
            max_x = int(np.max(rotated_corners[:, 0]))
            min_y = int(np.min(rotated_corners[:, 1]))
            max_y = int(np.max(rotated_corners[:, 1]))

            bboxes.append({
                "char": text[original_idx],
                "x0": max(0, min_x),
                "y0": max(0, min_y),
                "x1": min(canvas_w, max_x),
                "y1": min(canvas_h, max_y)
            })

            # Advance position
            current_position += char_width * (1 - glyph_overlap_intensity)

        # Ensure bboxes are in original text order
        if direction in ["right_to_left", "bottom_to_top"]:
            bboxes = bboxes[::-1]

        # Crop image to actual content bounds
        if bboxes:
            min_x = min(bbox["x0"] for bbox in bboxes)
            min_y = min(bbox["y0"] for bbox in bboxes)
            max_x = max(bbox["x1"] for bbox in bboxes)
            max_y = max(bbox["y1"] for bbox in bboxes)

            # Add a small margin
            crop_margin = 10
            crop_x0 = max(0, min_x - crop_margin)
            crop_y0 = max(0, min_y - crop_margin)
            crop_x1 = min(canvas_w, max_x + crop_margin)
            crop_y1 = min(canvas_h, max_y + crop_margin)

            # Crop the image
            image = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

            # Adjust all bboxes to the cropped coordinate system
            adjusted_bboxes = []
            for bbox in bboxes:
                adjusted_bbox = bbox.copy()
                adjusted_bbox["x0"] = bbox["x0"] - crop_x0
                adjusted_bbox["y0"] = bbox["y0"] - crop_y0
                adjusted_bbox["x1"] = bbox["x1"] - crop_x0
                adjusted_bbox["y1"] = bbox["y1"] - crop_y0
                adjusted_bboxes.append(adjusted_bbox)

            return image, adjusted_bboxes

        return image, bboxes

    def _render_left_to_right(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """Renders left-to-right text."""
        return self._render_text_surface(text, font_path, glyph_overlap_intensity, color_mode, color_palette)

    def _render_right_to_left(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """Renders right-to-left text after reshaping."""
        reshaped_text = bidi.algorithm.get_display(text)
        return self._render_text_surface(reshaped_text, font_path, glyph_overlap_intensity, color_mode, color_palette)

    def _render_top_to_bottom(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """Renders text vertically from top to bottom."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=False, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette)

    def _render_bottom_to_top(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """Renders text vertically from bottom to top."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=True, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette)

    def _render_vertical_text(self, text: str, font_path: str, is_bottom_to_top: bool, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """Renders text vertically, either TTB or BTT."""
        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        bboxes = []
        
        char_widths = []
        char_heights = []
        total_height = 0
        ascent, descent = font.getmetrics()
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_height = ascent + descent
                char_widths.append(char_width)
                char_heights.append(char_height)
                total_height += char_height * (1 - glyph_overlap_intensity)
            except AttributeError:
                w, h = font.getsize(char)
                char_widths.append(w)
                h = ascent + descent
                char_heights.append(h)
                total_height += h * (1 - glyph_overlap_intensity)

        max_width = max(char_widths) if char_widths else 0

        margin = 10
        image_width = max_width + margin * 2
        image_height = int(total_height + margin * 2)
        image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

        if is_bottom_to_top:
            current_y = image_height - margin
            char_list = list(enumerate(reversed(text)))
            for i, char in char_list:
                char_index = len(text) - 1 - i
                char_height = char_heights[char_index]
                char_width = char_widths[char_index]
                current_y -= char_height * (1 - glyph_overlap_intensity)
                x_pos = (image_width - char_width) / 2
                fill = "black" # Default
                if color_mode == 'per_glyph' and color_palette:
                    fill = color_palette[char_index]
                elif color_mode == 'gradient' and color_palette:
                    t = char_index / (len(text) - 1) if len(text) > 1 else 0
                    start_color = np.array(color_palette[0])
                    end_color = np.array(color_palette[1])
                    fill = tuple((start_color + t * (end_color - start_color)).astype(int))

                char_image = Image.new("RGBA", (char_width, char_height), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_image)
                char_draw.text((0, 0), char, font=font, fill=fill)
                image.paste(char_image, (int(x_pos), int(current_y)), char_image)

                bboxes.append({"char": char, "x0": int(x_pos), "y0": int(current_y), "x1": int(x_pos + char_width), "y1": int(current_y + char_height)})
            bboxes.reverse() # Bboxes should be in original text order
        else:
            current_y = margin
            for i, char in enumerate(text):
                char_width = char_widths[i]
                char_height = char_heights[i]
                x_pos = (image_width - char_width) / 2
                
                fill = "black" # Default
                if color_mode == 'per_glyph' and color_palette:
                    fill = color_palette[i]
                elif color_mode == 'gradient' and color_palette:
                    t = i / (len(text) - 1) if len(text) > 1 else 0
                    start_color = np.array(color_palette[0])
                    end_color = np.array(color_palette[1])
                    fill = tuple((start_color + t * (end_color - start_color)).astype(int))

                char_image = Image.new("RGBA", (char_width, char_height), (0, 0, 0, 0))
                char_draw = ImageDraw.Draw(char_image)
                char_draw.text((0, 0), char, font=font, fill=fill)
                image.paste(char_image, (int(x_pos), int(current_y)), char_image)

                bboxes.append({"char": char, "x0": int(x_pos), "y0": int(current_y), "x1": int(x_pos + char_width), "y1": int(current_y + char_height)})
                current_y += char_height * (1 - glyph_overlap_intensity)

        return image, bboxes

    def _render_text_surface(self, text_to_render: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list]):
        """
        A common method to render a string of text onto a new image surface,
        calculating per-character bounding boxes.
        """
        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        bboxes = []

        # First pass: calculate total dimensions
        total_width = 0
        ascent, descent = font.getmetrics()
        max_height = ascent + descent
        char_widths = []
        for char in text_to_render:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_widths.append(char_width)
                total_width += char_width * (1 - glyph_overlap_intensity)
            except AttributeError:
                w, h = font.getsize(char)
                char_widths.append(w)
                total_width += w * (1 - glyph_overlap_intensity)

        margin = 10
        image_width = int(total_width + margin * 2)
        image_height = max_height + margin * 2
        image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

        # Second pass: draw characters and record bounding boxes
        current_x = margin
        for i, char in enumerate(text_to_render):
            char_width = char_widths[i]
            char_height = ascent + descent

            fill = "black" # Default
            if color_mode == 'per_glyph' and color_palette:
                fill = color_palette[i]
            elif color_mode == 'gradient' and color_palette:
                # Horizontal gradient
                t = i / (len(text_to_render) - 1) if len(text_to_render) > 1 else 0
                start_color = np.array(color_palette[0])
                end_color = np.array(color_palette[1])
                inter_color = tuple((start_color + t * (end_color - start_color)).astype(int))
                fill = inter_color

            # To handle RGBA colors correctly, we draw on a temp surface and paste
            char_image = Image.new("RGBA", (char_width, char_height), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_image)
            char_draw.text((0, 0), char, font=font, fill=fill)
            image.paste(char_image, (int(current_x), margin), char_image)

            # Record bounding box
            x0 = current_x
            x1 = current_x + char_width
            y0 = margin
            y1 = margin + char_height
            bboxes.append({"char": char, "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)})

            # Update position for next character
            current_x += char_width * (1 - glyph_overlap_intensity)

        return image, bboxes
