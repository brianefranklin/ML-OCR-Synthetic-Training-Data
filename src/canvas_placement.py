"""Functions for placing a rendered text image onto a larger canvas.

This module provides a set of functions to handle the creation of a final canvas,
placement of the text, and the critical adjustment of bounding box coordinates.
"""

import random
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

def generate_random_canvas_size(
    image_w: int, 
    image_h: int, 
    min_padding: int = 10, 
    max_padding: int = 50
) -> Tuple[int, int]:
    """Generates a random canvas size larger than the given image dimensions.

    Args:
        image_w: Width of the text image.
        image_h: Height of the text image.
        min_padding: The minimum padding to add to each side.
        max_padding: The maximum padding to add to each side.

    Returns:
        A tuple (canvas_width, canvas_height).
    """
    # Add random padding to each dimension.
    pad_w = random.randint(min_padding, max_padding)
    pad_h = random.randint(min_padding, max_padding)
    
    canvas_w = image_w + pad_w * 2
    canvas_h = image_h + pad_h * 2
    
    return canvas_w, canvas_h

def calculate_text_placement(
    canvas_w: int, 
    canvas_h: int, 
    text_w: int, 
    text_h: int, 
    strategy: str
) -> Tuple[int, int]:
    """Calculates the top-left (x, y) position to place the text image on the canvas.

    Args:
        canvas_w: Width of the canvas.
        canvas_h: Height of the canvas.
        text_w: Width of the text image.
        text_h: Height of the text image.
        strategy: The placement strategy to use (e.g., 'uniform_random').

    Returns:
        A tuple (x, y) for the top-left corner of the text.
    """
    if strategy == "uniform_random":
        # Calculate the maximum possible top-left coordinates.
        max_x = canvas_w - text_w
        max_y = canvas_h - text_h
        # Choose a random position within the valid range.
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return x, y
    else:
        raise ValueError(f"Unsupported placement strategy: {strategy}")

def place_on_canvas(
    text_image: Image.Image,
    canvas_w: int,
    canvas_h: int,
    placement_x: int,
    placement_y: int,
    original_bboxes: List[Dict[str, Any]],
    background_path: Optional[str] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Places the text image onto a new canvas and adjusts bounding boxes.

    Args:
        text_image: The rendered text image (with a transparent background).
        canvas_w: The width of the new canvas.
        canvas_h: The height of the new canvas.
        placement_x: The x-coordinate for the top-left corner of the text.
        placement_y: The y-coordinate for the top-left corner of the text.
        original_bboxes: The list of original bounding boxes for the text image.
        background_path: Optional path to a background image.

    Returns:
        A tuple containing the final canvas image and the adjusted bounding boxes.
    """
    if background_path:
        # If a background is provided, load and crop it to the canvas size.
        bg_image = Image.open(background_path).convert("RGBA")
        # A simple crop from the top-left is used for now.
        canvas = bg_image.crop((0, 0, canvas_w, canvas_h))
    else:
        # If no background, create a new transparent canvas.
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # Paste the text image onto the canvas.
    # The text_image itself is used as the mask to handle its own transparency.
    canvas.paste(text_image, (placement_x, placement_y), text_image)

    # Adjust all bounding box coordinates to be relative to the new canvas.
    adjusted_bboxes = []
    for bbox in original_bboxes:
        adj_bbox = bbox.copy()
        adj_bbox["x0"] += placement_x
        adj_bbox["y0"] += placement_y
        adj_bbox["x1"] += placement_x
        adj_bbox["y1"] += placement_y
        adjusted_bboxes.append(adj_bbox)
        
    return canvas, adjusted_bboxes