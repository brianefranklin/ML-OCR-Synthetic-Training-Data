"""
Core component for generating synthetic OCR data.
"""

from PIL import Image, ImageDraw, ImageFont
import bidi.algorithm
import random
from src.canvas_placement import (
    generate_random_canvas_size,
    calculate_text_placement,
    place_on_canvas,
)
from src.effects import apply_ink_bleed

class OCRDataGenerator:
    """
    Orchestrates the entire image generation pipeline, from rendering text
    to applying augmentations.
    """
    def __init__(self):
        """Initializes the OCRDataGenerator."""
        pass

    def plan_generation(self, text: str, font_path: str, direction: str, glyph_overlap_intensity: float = 0.0, ink_bleed_radius: float = 0.0):
        """Creates a plan (a dictionary of truth data) for generating an image."""
        # This is a temporary, minimal implementation to get the text surface size.
        # In the future, this will be more sophisticated.
        text_surface, _ = self._render_text(text, font_path, direction, glyph_overlap_intensity)
        
        canvas_w, canvas_h = generate_random_canvas_size(text_surface.width, text_surface.height)
        placement_x, placement_y = calculate_text_placement(
            canvas_w, canvas_h, text_surface.width, text_surface.height, "uniform_random"
        )

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
        }

    def generate_from_plan(self, plan: dict):
        """Generates an image deterministically from a plan dictionary."""
        random.seed(plan["seed"])

        text_surface, bboxes = self._render_text(
            plan["text"], 
            plan["font_path"], 
            plan["direction"], 
            plan.get("glyph_overlap_intensity", 0.0)
        )

        # Apply ink bleed if specified
        ink_bleed_radius = plan.get("ink_bleed_radius", 0.0)
        if ink_bleed_radius > 0:
            text_surface = apply_ink_bleed(text_surface, ink_bleed_radius)

        final_image, final_bboxes = place_on_canvas(
            text_image=text_surface,
            canvas_w=plan["canvas_w"],
            canvas_h=plan["canvas_h"],
            placement_x=plan["placement_x"],
            placement_y=plan["placement_y"],
            original_bboxes=bboxes,
        )

        return final_image, final_bboxes

    def _render_text(self, text: str, font_path: str, direction: str, glyph_overlap_intensity: float = 0.0):
        """Internal dispatcher for rendering text surfaces."""
        if direction == "left_to_right":
            return self._render_left_to_right(text, font_path, glyph_overlap_intensity)
        elif direction == "right_to_left":
            return self._render_right_to_left(text, font_path, glyph_overlap_intensity)
        elif direction == "top_to_bottom":
            return self._render_top_to_bottom(text, font_path, glyph_overlap_intensity)
        elif direction == "bottom_to_top":
            return self._render_bottom_to_top(text, font_path, glyph_overlap_intensity)
        else:
            raise ValueError(f"Unsupported text direction: {direction}")

    def _render_left_to_right(self, text: str, font_path: str, glyph_overlap_intensity: float):
        """Renders left-to-right text."""
        return self._render_text_surface(text, font_path, glyph_overlap_intensity)

    def _render_right_to_left(self, text: str, font_path: str, glyph_overlap_intensity: float):
        """Renders right-to-left text after reshaping."""
        reshaped_text = bidi.algorithm.get_display(text)
        return self._render_text_surface(reshaped_text, font_path, glyph_overlap_intensity)

    def _render_top_to_bottom(self, text: str, font_path: str, glyph_overlap_intensity: float):
        """Renders text vertically from top to bottom."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=False, glyph_overlap_intensity=glyph_overlap_intensity)

    def _render_bottom_to_top(self, text: str, font_path: str, glyph_overlap_intensity: float):
        """Renders text vertically from bottom to top."""
        return self._render_vertical_text(text, font_path, is_bottom_to_top=True, glyph_overlap_intensity=glyph_overlap_intensity)

    def _render_vertical_text(self, text: str, font_path: str, is_bottom_to_top: bool, glyph_overlap_intensity: float):
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
        image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)

        if is_bottom_to_top:
            current_y = image_height - margin
            char_list = list(enumerate(reversed(text)))
            for i, char in char_list:
                char_index = len(text) - 1 - i
                char_height = char_heights[char_index]
                char_width = char_widths[char_index]
                current_y -= char_height * (1 - glyph_overlap_intensity)
                draw.text(( (image_width - char_width) / 2, current_y), char, font=font, fill="black")
                bboxes.append({"char": char, "x0": int((image_width - char_width) / 2), "y0": int(current_y), "x1": int((image_width + char_width) / 2), "y1": int(current_y + char_height)})
            bboxes.reverse() # Bboxes should be in original text order
        else:
            current_y = margin
            for i, char in enumerate(text):
                char_width = char_widths[i]
                char_height = char_heights[i]
                x_pos = (image_width - char_width) / 2
                draw.text((x_pos, current_y), char, font=font, fill="black")
                bboxes.append({"char": char, "x0": int(x_pos), "y0": int(current_y), "x1": int(x_pos + char_width), "y1": int(current_y + char_height)})
                current_y += char_height * (1 - glyph_overlap_intensity)

        return image, bboxes

    def _render_text_surface(self, text_to_render: str, font_path: str, glyph_overlap_intensity: float):
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
        image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))
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

            # Draw character
            draw.text((current_x, margin), char, font=font, fill="black")

            # Record bounding box
            x0 = current_x
            y0 = margin
            x1 = current_x + char_width
            y1 = margin + char_height
            bboxes.append({"char": char, "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)})

            # Update position for next character
            current_x += char_width * (1 - glyph_overlap_intensity)

        return image, bboxes

