from PIL import Image, ImageDraw, ImageFont
import bidi.algorithm
import random
import numpy as np
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
    apply_erosion_dilation
)
from src.augmentations import (
    apply_rotation, 
    apply_perspective_warp, 
    apply_elastic_distortion,
    apply_grid_distortion,
    apply_optical_distortion
)

class OCRDataGenerator:
    """
    Orchestrates the entire image generation pipeline, from rendering text
    to applying augmentations.
    """
    def __init__(self):
        """Initializes the OCRDataGenerator."""
        pass

    def plan_generation(
        self, 
        text: str, 
        font_path: str, 
        direction: str, 
        glyph_overlap_intensity: float = 0.0, 
        ink_bleed_radius: float = 0.0,
        drop_shadow_options: dict = None,
        color_mode: str = 'uniform',
        color_palette: list = None,
        rotation_angle: float = 0.0,
        perspective_warp_magnitude: float = 0.0,
        elastic_distortion_options: dict = None,
        grid_distortion_options: dict = None,
        optical_distortion_options: dict = None,
        noise_amount: float = 0.0,
        blur_radius: float = 0.0,
        brightness_factor: float = 1.0,
        contrast_factor: float = 1.0,
        erosion_dilation_options: dict = None,
        background_manager = None,
    ):
        """Creates a plan (a dictionary of truth data) for generating an image."""
        text_surface, _ = self._render_text(
            text, font_path, direction, glyph_overlap_intensity, color_mode, color_palette
        )
        
        canvas_w, canvas_h = generate_random_canvas_size(text_surface.width, text_surface.height)
        placement_x, placement_y = calculate_text_placement(
            canvas_w, canvas_h, text_surface.width, text_surface.height, "uniform_random"
        )

        background_path = background_manager.select_background() if background_manager else None

        return {
            "text": text,
            "font_path": font_path,
            "direction": direction,
            "seed": random.randint(0, 2**32 - 1),
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "placement_x": placement_x,
            "placement_y": placement_y,
            "glyph_overlap_intensity": glyph_overlap_intensity,
            "ink_bleed_radius": ink_bleed_radius,
            "drop_shadow_options": drop_shadow_options,
            "color_mode": color_mode,
            "color_palette": color_palette,
            "rotation_angle": rotation_angle,
            "perspective_warp_magnitude": perspective_warp_magnitude,
            "elastic_distortion_options": elastic_distortion_options,
            "grid_distortion_options": grid_distortion_options,
            "optical_distortion_options": optical_distortion_options,
            "noise_amount": noise_amount,
            "blur_radius": blur_radius,
            "brightness_factor": brightness_factor,
            "contrast_factor": contrast_factor,
            "erosion_dilation_options": erosion_dilation_options,
            "background_path": background_path,
        }

    def generate_from_plan(self, plan: dict):
        """Generates an image deterministically from a plan dictionary."""
        random.seed(plan["seed"])

        text_surface, bboxes = self._render_text(
            plan["text"], 
            plan["font_path"], 
            plan["direction"], 
            plan.get("glyph_overlap_intensity", 0.0),
            plan.get("color_mode", 'uniform'),
            plan.get("color_palette"),
        )

        # Apply effects
        ink_bleed_radius = plan.get("ink_bleed_radius", 0.0)
        if ink_bleed_radius > 0:
            text_surface = apply_ink_bleed(text_surface, ink_bleed_radius)
        
        drop_shadow_options = plan.get("drop_shadow_options")
        if drop_shadow_options:
            text_surface = apply_drop_shadow(text_surface, **drop_shadow_options)

        final_image, final_bboxes = place_on_canvas(
            text_image=text_surface,
            canvas_w=plan["canvas_w"],
            canvas_h=plan["canvas_h"],
            placement_x=plan["placement_x"],
            placement_y=plan["placement_y"],
            original_bboxes=bboxes,
            background_path=plan.get("background_path"),
        )

        # Apply augmentations
        rotation_angle = plan.get("rotation_angle", 0.0)
        if rotation_angle != 0.0:
            final_image, final_bboxes = apply_rotation(final_image, final_bboxes, rotation_angle)
        
        warp_magnitude = plan.get("perspective_warp_magnitude", 0.0)
        if warp_magnitude > 0.0:
            final_image, final_bboxes = apply_perspective_warp(final_image, final_bboxes, warp_magnitude)

        elastic_options = plan.get("elastic_distortion_options")
        if elastic_options:
            final_image, final_bboxes = apply_elastic_distortion(final_image, final_bboxes, **elastic_options)

        grid_distortion_options = plan.get("grid_distortion_options")
        if grid_distortion_options:
            final_image, final_bboxes = apply_grid_distortion(final_image, final_bboxes, **grid_distortion_options)

        optical_distortion_options = plan.get("optical_distortion_options")
        if optical_distortion_options:
            final_image, final_bboxes = apply_optical_distortion(final_image, final_bboxes, **optical_distortion_options)

        erosion_dilation_options = plan.get("erosion_dilation_options")
        if erosion_dilation_options:
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

    def _render_text(self, text: str, font_path: str, direction: str, glyph_overlap_intensity: float = 0.0, color_mode: str = 'uniform', color_palette: list = None):
        """Internal dispatcher for rendering text surfaces."""
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

    def _render_left_to_right(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """Renders left-to-right text."""
        return self._render_text_surface(text, font_path, glyph_overlap_intensity, color_mode, color_palette)

    def _render_right_to_left(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """Renders right-to-left text after reshaping."""
        reshaped_text = bidi.algorithm.get_display(text)
        return self._render_text_surface(reshaped_text, font_path, glyph_overlap_intensity, color_mode, color_palette)

    def _render_top_to_bottom(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """Renders text vertically from top to bottom."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=False, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette)

    def _render_bottom_to_top(self, text: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """Renders text vertically from bottom to top."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=True, glyph_overlap_intensity=glyph_overlap_intensity, color_mode=color_mode, color_palette=color_palette)

    def _render_vertical_text(self, text: str, font_path: str, is_bottom_to_top: bool, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """Renders text vertically, either TTB or BTT."""
        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        bboxes = []
        
        char_widths = []
        char_heights = []
        total_height = 0
        for char in text:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                char_widths.append(char_width)
                char_heights.append(char_height)
                total_height += char_height * (1 - glyph_overlap_intensity)
            except AttributeError:
                w, h = font.getsize(char)
                char_widths.append(w)
                char_heights.append(h)
                total_height += h * (1 - glyph_overlap_intensity)

        max_width = max(char_widths) if char_widths else 0

        margin = 10
        image_width = max_width + margin * 2
        image_height = int(total_height + margin * 2)
        image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

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

    def _render_text_surface(self, text_to_render: str, font_path: str, glyph_overlap_intensity: float, color_mode: str, color_palette: list):
        """
        A common method to render a string of text onto a new image surface,
        calculating per-character bounding boxes.
        """
        font_size = 32
        font = ImageFont.truetype(font_path, font_size)
        bboxes = []

        # First pass: calculate total dimensions
        total_width = 0
        max_height = 0
        char_widths = []
        for char in text_to_render:
            try:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_widths.append(char_width)
                total_width += char_width * (1 - glyph_overlap_intensity)
                max_height = max(max_height, bbox[3] - bbox[1])
            except AttributeError:
                w, h = font.getsize(char)
                char_widths.append(w)
                total_width += w * (1 - glyph_overlap_intensity)
                max_height = max(max_height, h)

        margin = 10
        image_width = int(total_width + margin * 2)
        image_height = max_height + margin * 2
        image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Second pass: draw characters and record bounding boxes
        current_x = margin
        for i, char in enumerate(text_to_render):
            char_width = char_widths[i]
            try:
                bbox = font.getbbox(char)
                char_height = bbox[3] - bbox[1]
            except AttributeError:
                _, char_height = font.getsize(char)

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
            y0 = margin
            x1 = current_x + char_width
            y1 = margin + char_height
            bboxes.append({"char": char, "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)})

            # Update position for next character
            current_x += char_width * (1 - glyph_overlap_intensity)

        return image, bboxes