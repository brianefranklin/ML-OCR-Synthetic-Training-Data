"""
Canvas placement module for OCR data generation.
Places text images on larger canvases with configurable placement strategies.
"""

import random
import json
import math
from typing import List, Tuple, Dict, Union
from PIL import Image
import numpy as np


def generate_random_canvas_size(
    text_size: Tuple[int, int],
    min_padding: int = 10,
    max_megapixels: float = 12.0
) -> Tuple[int, int]:
    """
    Generate a random canvas size that fits the text with padding.

    Args:
        text_size: (width, height) of the text image
        min_padding: Minimum padding around text in pixels
        max_megapixels: Maximum canvas size in megapixels

    Returns:
        Tuple of (canvas_width, canvas_height)
    """
    text_width, text_height = text_size

    # Minimum canvas size (text + padding)
    min_width = text_width + 2 * min_padding
    min_height = text_height + 2 * min_padding

    # Calculate maximum dimensions based on max megapixels
    max_pixels = int(max_megapixels * 1_000_000)

    # Try to find a random size that respects both constraints
    # Use multipliers between 1.0 (minimum) and a reasonable maximum
    max_multiplier = math.sqrt(max_pixels / (min_width * min_height))
    max_multiplier = min(max_multiplier, 5.0)  # Cap at 5x to avoid extreme sizes

    # Generate random multipliers for width and height
    width_multiplier = random.uniform(1.0, max_multiplier)
    height_multiplier = random.uniform(1.0, max_multiplier)

    canvas_width = int(min_width * width_multiplier)
    canvas_height = int(min_height * height_multiplier)

    # Ensure we don't exceed max megapixels
    while canvas_width * canvas_height > max_pixels:
        # Scale down proportionally
        scale = math.sqrt(max_pixels / (canvas_width * canvas_height))
        canvas_width = int(canvas_width * scale)
        canvas_height = int(canvas_height * scale)

    # Ensure minimum size is still respected
    canvas_width = max(canvas_width, min_width)
    canvas_height = max(canvas_height, min_height)

    return (canvas_width, canvas_height)


def calculate_text_placement(
    text_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    min_padding: int,
    placement: str = 'weighted_random'
) -> Tuple[int, int]:
    """
    Calculate where to place text on canvas.

    Args:
        text_size: (width, height) of text image
        canvas_size: (width, height) of canvas
        min_padding: Minimum padding from edges
        placement: Placement strategy ('uniform_random', 'weighted_random', 'center')

    Returns:
        Tuple of (x_offset, y_offset) for text placement
    """
    text_width, text_height = text_size
    canvas_width, canvas_height = canvas_size

    # Calculate valid placement range
    max_x = canvas_width - text_width - min_padding
    max_y = canvas_height - text_height - min_padding
    min_x = min_padding
    min_y = min_padding

    if placement == 'center':
        # Place at exact center
        x_offset = (canvas_width - text_width) // 2
        y_offset = (canvas_height - text_height) // 2

    elif placement == 'uniform_random':
        # Uniform random placement within valid bounds
        x_offset = random.randint(min_x, max(min_x, max_x))
        y_offset = random.randint(min_y, max(min_y, max_y))

    elif placement == 'weighted_random':
        # Weighted toward center using triangular distribution
        center_x = (canvas_width - text_width) / 2
        center_y = (canvas_height - text_height) / 2

        # Use triangular distribution with mode at center
        x_offset = int(random.triangular(min_x, max(min_x, max_x), center_x))
        y_offset = int(random.triangular(min_y, max(min_y, max_y), center_y))

    else:
        # Default to weighted_random
        center_x = (canvas_width - text_width) / 2
        center_y = (canvas_height - text_height) / 2
        x_offset = int(random.triangular(min_x, max(min_x, max_x), center_x))
        y_offset = int(random.triangular(min_y, max(min_y, max_y), center_y))

    return (x_offset, y_offset)


def place_on_canvas(
    text_image: Image.Image,
    char_bboxes: List[List[float]],
    canvas_size: Tuple[int, int] = None,
    min_padding: int = 10,
    placement: str = 'weighted_random',
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[Image.Image, Dict]:
    """
    Place text image on a larger canvas.

    Args:
        text_image: The rendered text image
        char_bboxes: Character bounding boxes (relative to text_image)
        canvas_size: Canvas dimensions (if None, generates random size)
        min_padding: Minimum padding around text
        placement: Placement strategy ('uniform_random', 'weighted_random', 'center')
        background_color: Background color for canvas (default white)

    Returns:
        Tuple of (canvas_image, metadata_dict)
        metadata_dict contains:
            - canvas_size: [width, height]
            - text_placement: [x_offset, y_offset]
            - line_bbox: [x_min, y_min, x_max, y_max]
            - char_bboxes: List of adjusted character bboxes
    """
    text_width, text_height = text_image.size

    # Generate random canvas size if not provided
    if canvas_size is None:
        canvas_size = generate_random_canvas_size(
            (text_width, text_height),
            min_padding=min_padding
        )

    canvas_width, canvas_height = canvas_size

    # Create background canvas
    canvas = Image.new('RGB', canvas_size, color=background_color)

    # Calculate text placement
    x_offset, y_offset = calculate_text_placement(
        (text_width, text_height),
        canvas_size,
        min_padding,
        placement
    )

    # Text image should already be RGBA with transparent background
    # Paste directly using alpha channel as mask
    if text_image.mode == 'RGBA':
        canvas.paste(text_image, (x_offset, y_offset), text_image)
    else:
        # Fallback: convert to RGBA and paste
        text_rgba = text_image.convert('RGBA')
        canvas.paste(text_rgba, (x_offset, y_offset), text_rgba)

    # Calculate line-level bounding box
    line_bbox = [
        x_offset,
        y_offset,
        x_offset + text_width,
        y_offset + text_height
    ]

    # Adjust character bboxes for placement offset
    adjusted_char_bboxes = []
    for bbox in char_bboxes:
        adjusted_bbox = [
            bbox[0] + x_offset,
            bbox[1] + y_offset,
            bbox[2] + x_offset,
            bbox[3] + y_offset
        ]
        adjusted_char_bboxes.append(adjusted_bbox)

    # Create metadata dictionary
    metadata = {
        'canvas_size': list(canvas_size),
        'text_placement': [x_offset, y_offset],
        'line_bbox': line_bbox,
        'char_bboxes': adjusted_char_bboxes
    }

    return canvas, metadata


def create_label_json(
    image_file: str,
    text: str,
    metadata: Dict
) -> Dict:
    """
    Create JSON label data structure.

    Args:
        image_file: Name of the image file
        text: The rendered text string
        metadata: Metadata from place_on_canvas

    Returns:
        Dictionary ready for JSON serialization
    """
    return {
        'image_file': image_file,
        'text': text,
        'canvas_size': metadata['canvas_size'],
        'text_placement': metadata['text_placement'],
        'line_bbox': metadata['line_bbox'],
        'char_bboxes': metadata['char_bboxes']
    }


def save_label_json(
    output_path: str,
    image_file: str,
    text: str,
    metadata: Dict
) -> None:
    """
    Save label data to JSON file.

    Args:
        output_path: Path to output JSON file
        image_file: Name of the image file
        text: The rendered text string
        metadata: Metadata from place_on_canvas
    """
    label_data = create_label_json(image_file, text, metadata)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    label_data = convert_to_native(label_data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(label_data, f, indent=2, ensure_ascii=False)
