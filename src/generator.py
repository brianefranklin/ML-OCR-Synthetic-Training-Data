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
from src.distributions import sample_parameter
from src.text_layout import (
    break_into_lines,
    calculate_multiline_dimensions
)

class OCRDataGenerator:
    """Orchestrates the entire image generation pipeline.
    
    This class brings together all the components of the generation process,
    from planning the parameters to rendering text, applying effects, and augmenting
    the final image.
    """

    def __init__(self):
        """Initializes the OCRDataGenerator."""
        pass

    def _generate_color_palette(self, spec: BatchSpecification, text: str) -> Optional[List]:
        """Generates a color palette based on the specification's color mode.

        Args:
            spec: The BatchSpecification defining color parameters.
            text: The text string (used for per_glyph mode to determine palette size).

        Returns:
            - None for uniform mode with black text (backward compatible)
            - Single-element list with RGB tuple for uniform mode with color
            - List of RGB tuples (one per character) for per_glyph mode
            - List of two RGB tuples [start, end] for gradient mode

        Examples:
            >>> spec = BatchSpecification(...)
            >>> generator._generate_color_palette(spec, "hello")
            # Returns palette based on spec.color_mode
        """
        if spec.color_mode == "uniform":
            # Sample a single color from the range
            r = random.randint(spec.text_color_min[0], spec.text_color_max[0])
            g = random.randint(spec.text_color_min[1], spec.text_color_max[1])
            b = random.randint(spec.text_color_min[2], spec.text_color_max[2])
            color = (r, g, b)

            # Backward compatibility: return None for black text
            if color == (0, 0, 0):
                return None
            return color

        elif spec.color_mode == "per_glyph":
            # Generate N random colors based on palette size config
            # Colors will be cycled through the text (e.g., palette_size=2 for "HELLO" -> [color1, color2, color1, color2, color1])
            palette_size = random.randint(spec.per_glyph_palette_size_min, spec.per_glyph_palette_size_max)
            palette = []
            for _ in range(palette_size):
                r = random.randint(spec.text_color_min[0], spec.text_color_max[0])
                g = random.randint(spec.text_color_min[1], spec.text_color_max[1])
                b = random.randint(spec.text_color_min[2], spec.text_color_max[2])
                palette.append((r, g, b))
            return palette

        elif spec.color_mode == "gradient":
            # Sample start and end colors
            start_r = random.randint(spec.gradient_start_color_min[0], spec.gradient_start_color_max[0])
            start_g = random.randint(spec.gradient_start_color_min[1], spec.gradient_start_color_max[1])
            start_b = random.randint(spec.gradient_start_color_min[2], spec.gradient_start_color_max[2])
            start_color = (start_r, start_g, start_b)

            end_r = random.randint(spec.gradient_end_color_min[0], spec.gradient_end_color_max[0])
            end_g = random.randint(spec.gradient_end_color_min[1], spec.gradient_end_color_max[1])
            end_b = random.randint(spec.gradient_end_color_min[2], spec.gradient_end_color_max[2])
            end_color = (end_r, end_g, end_b)

            return [start_color, end_color]

        else:
            # Should never reach here due to validation, but provide fallback
            return None

    def _generate_shadow_options(
        self,
        offset_x_min: int, offset_x_max: int, offset_x_dist: str,
        offset_y_min: int, offset_y_max: int, offset_y_dist: str,
        radius_min: float, radius_max: float, radius_dist: str,
        color_min: Tuple[int, int, int, int], color_max: Tuple[int, int, int, int]
    ) -> Optional[Dict[str, Any]]:
        """Generates shadow options if shadow is enabled (non-zero offsets).

        Args:
            offset_x_min/max: X offset range for shadow.
            offset_x_dist: Distribution type for X offset.
            offset_y_min/max: Y offset range for shadow.
            offset_y_dist: Distribution type for Y offset.
            radius_min/max: Blur radius range for shadow.
            radius_dist: Distribution type for blur radius.
            color_min/max: RGBA color range for shadow (4-tuple).

        Returns:
            None if shadow is disabled (both offsets are 0), otherwise a dictionary
            with keys: "offset" (tuple), "radius" (float), "color" (4-tuple RGBA).
        """
        # Sample shadow parameters
        offset_x = int(sample_parameter(offset_x_min, offset_x_max, offset_x_dist))
        offset_y = int(sample_parameter(offset_y_min, offset_y_max, offset_y_dist))

        # If both offsets are 0, shadow is disabled
        if offset_x == 0 and offset_y == 0:
            return None

        radius = sample_parameter(radius_min, radius_max, radius_dist)

        # Sample RGBA color
        r = random.randint(color_min[0], color_max[0])
        g = random.randint(color_min[1], color_max[1])
        b = random.randint(color_min[2], color_max[2])
        a = random.randint(color_min[3], color_max[3])
        color = (r, g, b, a)

        return {
            "offset": (offset_x, offset_y),
            "radius": radius,
            "color": color
        }

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
        # Sample font size first (needed for canvas sizing)
        font_size = random.randint(spec.font_size_min, spec.font_size_max)

        # Sample multi-line parameters
        num_lines = random.randint(spec.min_lines, spec.max_lines)
        line_spacing = sample_parameter(
            spec.line_spacing_min,
            spec.line_spacing_max,
            spec.line_spacing_distribution
        )

        # Break text into lines if multi-line mode
        if num_lines > 1:
            # Calculate approximate chars per line for line breaking
            max_chars_per_line = max(1, len(text) // num_lines)
            lines = break_into_lines(text, max_chars_per_line, num_lines, spec.line_break_mode)

            # Calculate dimensions for multi-line text
            font = ImageFont.truetype(font_path, font_size)
            text_width, text_height = calculate_multiline_dimensions(
                lines, font, line_spacing, spec.text_direction, 0.0
            )
        else:
            lines = [text]
            # Render a temporary surface to get the dimensions for canvas calculation.
            # Use straight text for canvas sizing regardless of curve type
            text_surface, _ = self._render_text(text, font_path, spec.text_direction, 0.0, 'uniform', None, "none", 0.0, True, 0.0, 0.0, 0.0, font_size)
            text_width, text_height = text_surface.width, text_surface.height

        canvas_w, canvas_h = generate_random_canvas_size(text_width, text_height)
        placement_x, placement_y = calculate_text_placement(
            canvas_w, canvas_h, text_width, text_height, "uniform_random"
        )

        background_path = background_manager.select_background() if background_manager else None

        # Build the final plan dictionary
        return {
            "text": text,
            "lines": lines,  # Text broken into lines
            "num_lines": num_lines,
            "line_spacing": line_spacing,
            "line_break_mode": spec.line_break_mode,
            "text_alignment": spec.text_alignment,
            "font_path": font_path,
            "direction": spec.text_direction,
            "seed": random.randint(0, 2**32 - 1),
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "placement_x": placement_x,
            "placement_y": placement_y,
            "glyph_overlap_intensity": sample_parameter(
                spec.glyph_overlap_intensity_min,
                spec.glyph_overlap_intensity_max,
                spec.glyph_overlap_intensity_distribution
            ),
            "ink_bleed_radius": sample_parameter(
                spec.ink_bleed_radius_min,
                spec.ink_bleed_radius_max,
                spec.ink_bleed_radius_distribution
            ),
            "drop_shadow_options": self._generate_shadow_options(
                spec.drop_shadow_offset_x_min, spec.drop_shadow_offset_x_max, spec.drop_shadow_offset_x_distribution,
                spec.drop_shadow_offset_y_min, spec.drop_shadow_offset_y_max, spec.drop_shadow_offset_y_distribution,
                spec.drop_shadow_radius_min, spec.drop_shadow_radius_max, spec.drop_shadow_radius_distribution,
                spec.drop_shadow_color_min, spec.drop_shadow_color_max
            ),
            "block_shadow_options": self._generate_shadow_options(
                spec.block_shadow_offset_x_min, spec.block_shadow_offset_x_max, spec.block_shadow_offset_x_distribution,
                spec.block_shadow_offset_y_min, spec.block_shadow_offset_y_max, spec.block_shadow_offset_y_distribution,
                spec.block_shadow_radius_min, spec.block_shadow_radius_max, spec.block_shadow_radius_distribution,
                spec.block_shadow_color_min, spec.block_shadow_color_max
            ),
            "color_mode": spec.color_mode,
            "color_palette": self._generate_color_palette(spec, text),
            "rotation_angle": sample_parameter(
                spec.rotation_angle_min,
                spec.rotation_angle_max,
                spec.rotation_angle_distribution
            ),
            "perspective_warp_magnitude": sample_parameter(
                spec.perspective_warp_magnitude_min,
                spec.perspective_warp_magnitude_max,
                spec.perspective_warp_magnitude_distribution
            ),
            "elastic_distortion_options": {
                "alpha": sample_parameter(
                    spec.elastic_distortion_alpha_min,
                    spec.elastic_distortion_alpha_max,
                    spec.elastic_distortion_alpha_distribution
                ),
                "sigma": sample_parameter(
                    spec.elastic_distortion_sigma_min,
                    spec.elastic_distortion_sigma_max,
                    spec.elastic_distortion_sigma_distribution
                )
            },
            "grid_distortion_options": {
                "num_steps": int(sample_parameter(
                    spec.grid_distortion_steps_min,
                    spec.grid_distortion_steps_max,
                    spec.grid_distortion_steps_distribution
                )),
                "distort_limit": int(sample_parameter(
                    spec.grid_distortion_limit_min,
                    spec.grid_distortion_limit_max,
                    spec.grid_distortion_limit_distribution
                ))
            },
            "optical_distortion_options": {
                "distort_limit": sample_parameter(
                    spec.optical_distortion_limit_min,
                    spec.optical_distortion_limit_max,
                    spec.optical_distortion_limit_distribution
                )
            },
            "cutout_options": {
                "cutout_size": (
                    int(sample_parameter(
                        spec.cutout_width_min,
                        spec.cutout_width_max,
                        spec.cutout_width_distribution
                    )),
                    int(sample_parameter(
                        spec.cutout_height_min,
                        spec.cutout_height_max,
                        spec.cutout_height_distribution
                    ))
                )
            },
            "noise_amount": sample_parameter(
                spec.noise_amount_min,
                spec.noise_amount_max,
                spec.noise_amount_distribution
            ),
            "blur_radius": sample_parameter(
                spec.blur_radius_min,
                spec.blur_radius_max,
                spec.blur_radius_distribution
            ),
            "brightness_factor": sample_parameter(
                spec.brightness_factor_min,
                spec.brightness_factor_max,
                spec.brightness_factor_distribution
            ),
            "contrast_factor": sample_parameter(
                spec.contrast_factor_min,
                spec.contrast_factor_max,
                spec.contrast_factor_distribution
            ),
            "erosion_dilation_options": {
                "mode": random.choice(['erode', 'dilate']),
                "kernel_size": int(sample_parameter(
                    spec.erosion_dilation_kernel_min,
                    spec.erosion_dilation_kernel_max,
                    spec.erosion_dilation_kernel_distribution
                ))
            },
            "background_path": background_path,
            # Curve parameters - always included for consistent ML feature vectors
            "curve_type": spec.curve_type,
            "arc_radius": sample_parameter(
                spec.arc_radius_min,
                spec.arc_radius_max,
                spec.arc_radius_distribution
            ),
            "arc_concave": spec.arc_concave,
            "sine_amplitude": sample_parameter(
                spec.sine_amplitude_min,
                spec.sine_amplitude_max,
                spec.sine_amplitude_distribution
            ),
            "sine_frequency": sample_parameter(
                spec.sine_frequency_min,
                spec.sine_frequency_max,
                spec.sine_frequency_distribution
            ),
            "sine_phase": sample_parameter(
                spec.sine_phase_min,
                spec.sine_phase_max,
                spec.sine_phase_distribution
            ),
            "font_size": font_size,
        }

    def plan_generation_batch(
        self,
        tasks: List[Tuple[BatchSpecification, str, str]],
        background_manager: Optional[BackgroundImageManager] = None
    ) -> List[Dict[str, Any]]:
        """Creates multiple generation plans at once for batch processing.

        This method enables separation of planning from execution, which can be useful
        for pre-generating all parameters before starting the generation process,
        analyzing parameter distributions, or parallelizing generation workflows.

        Args:
            tasks: A list of tuples, each containing (spec, text, font_path).
                - spec (BatchSpecification): The batch specification for this image.
                - text (str): The text string to be rendered.
                - font_path (str): The path to the font file to use.
            background_manager: An optional manager for selecting background images.

        Returns:
            A list of plan dictionaries, one for each input task. Each plan contains
            all parameters needed to generate an image via generate_from_plan().

        Example:
            >>> generator = OCRDataGenerator()
            >>> tasks = [
            ...     (spec1, "hello", "/path/to/font1.ttf"),
            ...     (spec2, "world", "/path/to/font2.ttf")
            ... ]
            >>> plans = generator.plan_generation_batch(tasks, background_manager)
            >>> for plan in plans:
            ...     image, bboxes = generator.generate_from_plan(plan)
        """
        plans: List[Dict[str, Any]] = []
        for spec, text, font_path in tasks:
            plan = self.plan_generation(spec, text, font_path, background_manager)
            plans.append(plan)
        return plans

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
        num_lines = plan.get("num_lines", 1)

        if num_lines > 1:
            # Multi-line rendering
            text_surface, bboxes = self._render_multiline_text(
                plan["lines"],
                plan["font_path"],
                plan["direction"],
                plan.get("line_spacing", 1.0),
                plan.get("text_alignment", "left"),
                plan.get("glyph_overlap_intensity", 0.0),
                plan.get("color_mode", 'uniform'),
                plan.get("color_palette"),
                plan.get("curve_type", "none"),
                plan.get("arc_radius", 0.0),
                plan.get("arc_concave", True),
                plan.get("sine_amplitude", 0.0),
                plan.get("sine_frequency", 0.0),
                plan.get("sine_phase", 0.0),
                plan.get("font_size", 32),
            )
        else:
            # Single-line rendering
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
                plan.get("font_size", 32),
            )
            # Add line_index: 0 to all bboxes for consistency with multi-line format
            for bbox in bboxes:
                bbox["line_index"] = 0

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

    def _render_multiline_text(
        self,
        lines: List[str],
        font_path: str,
        direction: str,
        line_spacing: float,
        alignment: str,
        glyph_overlap_intensity: float = 0.0,
        color_mode: str = 'uniform',
        color_palette: Optional[list] = None,
        curve_type: str = "none",
        arc_radius: float = 0.0,
        arc_concave: bool = True,
        sine_amplitude: float = 0.0,
        sine_frequency: float = 0.0,
        sine_phase: float = 0.0,
        font_size: int = 32
    ) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Renders multiple lines of text with proper spacing and alignment.

        Args:
            lines: List of text strings (one per line).
            font_path: Path to the font file.
            direction: Text direction.
            line_spacing: Line spacing multiplier.
            alignment: Text alignment.
            glyph_overlap_intensity: Intensity of character overlap.
            color_mode: Color mode for rendering.
            color_palette: Optional color palette.
            curve_type: Type of curve.
            arc_radius: Radius for arc curves.
            arc_concave: Whether arc curves concavely.
            sine_amplitude: Amplitude for sine wave.
            sine_frequency: Frequency for sine wave.
            sine_phase: Phase offset for sine wave.
            font_size: Size of the font in pixels.

        Returns:
            Tuple of (combined_image, bboxes) where bboxes include line_index.
        """
        if not lines:
            # Return empty image
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), []

        # Render each line separately
        line_images = []
        line_bboxes_list = []
        char_offset = 0  # Track character position in original text

        for line_idx, line_text in enumerate(lines):
            if not line_text:
                # Empty line - skip but track offset
                line_images.append(None)
                line_bboxes_list.append([])
                char_offset += 0
                continue

            # Determine color palette for this line
            line_color_palette = None
            if color_mode == 'per_glyph' and color_palette:
                # Extract colors for this line's characters
                line_color_palette = color_palette[char_offset:char_offset + len(line_text)]
            elif color_mode == 'gradient' and color_palette:
                # Use same gradient for all lines
                line_color_palette = color_palette
            elif color_mode == 'uniform' and color_palette:
                line_color_palette = color_palette

            # Render this line
            line_img, line_bboxes = self._render_text(
                line_text,
                font_path,
                direction,
                glyph_overlap_intensity,
                color_mode,
                line_color_palette,
                curve_type,
                arc_radius,
                arc_concave,
                sine_amplitude,
                sine_frequency,
                sine_phase,
                font_size
            )

            line_images.append(line_img)
            line_bboxes_list.append(line_bboxes)
            char_offset += len(line_text) + 1  # +1 for space/newline between lines

        # Calculate total dimensions and positions for combining lines
        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()
        line_height = int((ascent + descent) * line_spacing)

        if direction in ["left_to_right", "right_to_left"]:
            # Horizontal text: stack lines vertically
            max_width = max((img.width if img else 0) for img in line_images)
            total_height = line_height * len(lines)
            combined_image = Image.new("RGBA", (max_width, total_height), (0, 0, 0, 0))

            combined_bboxes = []
            for line_idx, (line_img, line_bboxes) in enumerate(zip(line_images, line_bboxes_list)):
                if line_img is None:
                    continue

                y_offset = line_idx * line_height

                # Calculate x offset based on alignment
                if alignment == "left":
                    x_offset = 0
                elif alignment == "center":
                    x_offset = (max_width - line_img.width) // 2
                elif alignment == "right":
                    x_offset = max_width - line_img.width
                else:
                    x_offset = 0

                # Paste line onto combined image
                combined_image.paste(line_img, (x_offset, y_offset), line_img)

                # Adjust bounding boxes and add line_index
                for bbox in line_bboxes:
                    adjusted_bbox = {
                        "char": bbox["char"],
                        "x0": bbox["x0"] + x_offset,
                        "y0": bbox["y0"] + y_offset,
                        "x1": bbox["x1"] + x_offset,
                        "y1": bbox["y1"] + y_offset,
                        "line_index": line_idx
                    }
                    combined_bboxes.append(adjusted_bbox)

        else:  # vertical text
            # Vertical text: stack lines horizontally
            char_width = int(font.size * 0.6)
            line_offset = int(char_width * line_spacing * 2)
            max_height = max((img.height if img else 0) for img in line_images)
            total_width = line_offset * len(lines)
            combined_image = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))

            combined_bboxes = []
            for line_idx, (line_img, line_bboxes) in enumerate(zip(line_images, line_bboxes_list)):
                if line_img is None:
                    continue

                x_offset = line_idx * line_offset

                # Calculate y offset based on alignment
                if alignment == "top":
                    y_offset = 0
                elif alignment == "center":
                    y_offset = (max_height - line_img.height) // 2
                elif alignment == "bottom":
                    y_offset = max_height - line_img.height
                else:
                    y_offset = 0

                # Paste line onto combined image
                combined_image.paste(line_img, (x_offset, y_offset), line_img)

                # Adjust bounding boxes and add line_index
                for bbox in line_bboxes:
                    adjusted_bbox = {
                        "char": bbox["char"],
                        "x0": bbox["x0"] + x_offset,
                        "y0": bbox["y0"] + y_offset,
                        "x1": bbox["x1"] + x_offset,
                        "y1": bbox["y1"] + y_offset,
                        "line_index": line_idx
                    }
                    combined_bboxes.append(adjusted_bbox)

        return combined_image, combined_bboxes

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
        sine_phase: float = 0.0,
        font_size: int = 32
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
            font_size: Size of the font in pixels.

        Returns:
            Tuple of (image, bboxes).
        """
        # Check if we should apply curves
        # Use a threshold to avoid numerical issues with very small values
        if curve_type == "arc" and arc_radius > 1.0:  # Changed from > 0 to > 1.0
            return self._render_arc_text(
                text, font_path, direction, arc_radius, arc_concave,
                glyph_overlap_intensity, color_mode, color_palette, font_size
            )
        elif curve_type == "sine" and sine_amplitude > 0.1:  # Small threshold
            return self._render_sine_text(
                text, font_path, direction, sine_amplitude, sine_frequency, sine_phase,
                glyph_overlap_intensity, color_mode, color_palette, font_size
            )

        # Fall back to straight text rendering
        if direction == "left_to_right":
            return self._render_left_to_right(text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)
        elif direction == "right_to_left":
            return self._render_right_to_left(text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)
        elif direction == "top_to_bottom":
            return self._render_top_to_bottom(text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)
        elif direction == "bottom_to_top":
            return self._render_bottom_to_top(text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)
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
        color_palette: Optional[list],
        font_size: int = 32
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
            font_size: Size of the font in pixels.

        Returns:
            Tuple of (image, bboxes).
        """
        import math

        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()
        char_height = ascent + descent

        # Measure all characters
        char_widths = []
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
            except (AttributeError, OSError):
                try:
                    char_width, _ = font.getsize(char)
                except (AttributeError, OSError):
                    char_width = int(font_size * 0.6)  # Fallback estimate, must be int
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
                fill = color_palette[original_idx % len(color_palette)]
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
        color_palette: Optional[list],
        font_size: int = 32
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
            font_size: Size of the font in pixels.

        Returns:
            Tuple of (image, bboxes).
        """
        import math

        font = ImageFont.truetype(font_path, font_size)
        ascent, descent = font.getmetrics()
        char_height = ascent + descent

        # Measure all characters
        char_widths = []
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
            except (AttributeError, OSError):
                try:
                    char_width, _ = font.getsize(char)
                except (AttributeError, OSError):
                    char_width = int(font_size * 0.6)  # Fallback estimate, must be int
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
                fill = color_palette[original_idx % len(color_palette)]
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

    def _render_left_to_right(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """Renders left-to-right text."""
        return self._render_text_surface(text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)

    def _render_right_to_left(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """Renders right-to-left text after reshaping."""
        reshaped_text = bidi.algorithm.get_display(text)
        image, bboxes = self._render_text_surface(reshaped_text, font_path, glyph_overlap_intensity, color_mode, color_palette, font_size)

        # Bboxes are currently in reshaped (visual) order, but should be in original text order
        # For RTL, reverse the bboxes to match original text order
        # Also update the char field to match the original text
        reversed_bboxes = []
        for i, bbox in enumerate(reversed(bboxes)):
            # Update bbox to use character from original text
            bbox_copy = bbox.copy()
            bbox_copy["char"] = text[i]
            reversed_bboxes.append(bbox_copy)

        return image, reversed_bboxes

    def _render_top_to_bottom(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """Renders text vertically from top to bottom."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=False, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette, font_size=font_size)

    def _render_bottom_to_top(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """Renders text vertically from bottom to top."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=True, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette, font_size=font_size)

    def _render_vertical_text(self, text: str, font_path: str, is_bottom_to_top: bool, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """Renders text vertically, either TTB or BTT."""
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
            except (AttributeError, OSError):
                try:
                    w, h = font.getsize(char)
                    char_widths.append(w)
                    h = ascent + descent
                    char_heights.append(h)
                    total_height += h * (1 - glyph_overlap_intensity)
                except (AttributeError, OSError):
                    # Ultimate fallback
                    char_width = int(font_size * 0.6)  # Must be int
                    char_height = ascent + descent
                    char_widths.append(char_width)
                    char_heights.append(char_height)
                    total_height += char_height * (1 - glyph_overlap_intensity)

        max_width = max(char_widths) if char_widths else 0

        margin = 10
        image_width = max_width + margin * 2
        image_height = int(total_height + margin * 2)
        image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

        if is_bottom_to_top:
            # For bottom-to-top, start at the bottom and render characters upward
            # First character appears at bottom, last character at top
            current_y = image_height - margin
            for i, char in enumerate(text):
                char_height = char_heights[i]
                char_width = char_widths[i]
                current_y -= char_height * (1 - glyph_overlap_intensity)
                x_pos = (image_width - char_width) / 2
                fill = "black" # Default
                if color_mode == 'per_glyph' and color_palette:
                    fill = color_palette[i % len(color_palette)]
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
        else:
            current_y = margin
            for i, char in enumerate(text):
                char_width = char_widths[i]
                char_height = char_heights[i]
                x_pos = (image_width - char_width) / 2
                
                fill = "black" # Default
                if color_mode == 'per_glyph' and color_palette:
                    fill = color_palette[i % len(color_palette)]
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

    def _render_text_surface(self, text_to_render: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: Optional[list], font_size: int = 32):
        """
        A common method to render a string of text onto a new image surface,
        calculating per-character bounding boxes.
        """
        try:
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError) as e:
            # If font loading fails (corrupted file, invalid size, etc.), raise a more specific error
            raise ValueError(f"Failed to load font '{font_path}' at size {font_size}: {e}") from e
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
            except (AttributeError, OSError) as e:
                # Handle FreeType errors like "execution context too long"
                # Fall back to estimating width or using a default
                try:
                    w, h = font.getsize(char)
                    char_widths.append(w)
                    total_width += w * (1 - glyph_overlap_intensity)
                except (AttributeError, OSError):
                    # Ultimate fallback: estimate reasonable width
                    estimated_width = int(font_size * 0.6)  # Rough estimate, must be int
                    char_widths.append(estimated_width)
                    total_width += estimated_width * (1 - glyph_overlap_intensity)

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
                fill = color_palette[i % len(color_palette)]
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
