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
from typing import List, Tuple, Dict, Optional
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
                            font: ImageFont.FreeTypeFont) -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render left-to-right horizontal text with character bboxes.

        Args:
            text: Text to render
            font: Font to use

        Returns:
            Tuple of (image, character_boxes)
        """
        # Calculate image dimensions
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        total_text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        img_width = (total_text_bbox[2] - total_text_bbox[0]) + 40
        img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

        # Create actual image
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)

        # Render characters and collect bboxes
        char_boxes = []
        x_offset = 20
        y_offset = 15

        for char in text:
            char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
            draw.text((x_offset, y_offset), char, font=font, fill='black')
            char_boxes.append(CharacterBox(char, list(char_bbox)))
            x_offset += draw.textlength(char, font=font)

        return image, char_boxes

    def render_right_to_left(self,
                           text: str,
                           font: ImageFont.FreeTypeFont) -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render right-to-left horizontal text with proper BiDi handling.

        Args:
            text: Text to render
            font: Font to use

        Returns:
            Tuple of (image, character_boxes)
        """
        # Use BiDi algorithm for proper RTL display
        display_text = bidi.algorithm.get_display(text)

        # Calculate image dimensions
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        total_text_bbox = temp_draw.textbbox((0, 0), display_text, font=font)
        img_width = (total_text_bbox[2] - total_text_bbox[0]) + 40
        img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

        # Create actual image
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)

        # Render characters from right to left
        char_boxes = []
        x_offset = img_width - 20
        y_offset = 15

        for char in display_text:
            char_width = draw.textlength(char, font=font)
            x_offset -= char_width

            draw.text((x_offset, y_offset), char, font=font, fill='black')
            char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))

            x_offset -= 1  # Small spacing

        return image, char_boxes

    def render_top_to_bottom(self,
                           text: str,
                           font: ImageFont.FreeTypeFont) -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render top-to-bottom vertical text (traditional CJK style).

        Args:
            text: Text to render
            font: Font to use

        Returns:
            Tuple of (image, character_boxes)
        """
        # Calculate image dimensions
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_widths = []
        char_heights = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_widths.append(bbox[2] - bbox[0])
            char_heights.append(bbox[3] - bbox[1])

        max_char_width = max(char_widths) if char_widths else 0
        total_height = sum(char_heights) + (len(char_heights) - 1) * 5 + 30

        img_width = max_char_width + 40
        img_height = total_height

        # Create actual image
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)

        # Render characters top to bottom
        char_boxes = []
        y_cursor = 15

        for char in text:
            char_bbox_temp = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox_temp[2] - char_bbox_temp[0]
            char_height = char_bbox_temp[3] - char_bbox_temp[1]
            x_cursor = (img_width - char_width) / 2

            draw.text((x_cursor, y_cursor), char, font=font, fill='black')
            char_bbox = draw.textbbox((x_cursor, y_cursor), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))
            logging.debug(f"char: {char}, bbox: {char_bbox}")

            y_cursor += char_height + 5

        return image, char_boxes

    def render_bottom_to_top(self,
                           text: str,
                           font: ImageFont.FreeTypeFont) -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render bottom-to-top vertical text.

        Args:
            text: Text to render
            font: Font to use

        Returns:
            Tuple of (image, character_boxes)
        """
        # Calculate image dimensions
        temp_img = Image.new('RGBA', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        char_widths = []
        char_heights = []
        for char in text:
            bbox = temp_draw.textbbox((0, 0), char, font=font)
            char_widths.append(bbox[2] - bbox[0])
            char_heights.append(bbox[3] - bbox[1])

        max_char_width = max(char_widths) if char_widths else 0
        total_height = sum(char_heights) + (len(char_heights) - 1) * 5 + 30

        img_width = max_char_width + 40
        img_height = total_height

        # Create actual image
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)

        # Render characters bottom to top
        char_boxes = []
        y_cursor = img_height - 15

        for char in text:
            char_bbox_temp = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox_temp[2] - char_bbox_temp[0]
            char_height = char_bbox_temp[3] - char_bbox_temp[1]
            x_cursor = (img_width - char_width) / 2
            y_cursor -= char_height

            draw.text((x_cursor, y_cursor), char, font=font, fill='black')
            char_bbox = draw.textbbox((x_cursor, y_cursor), char, font=font)
            char_boxes.append(CharacterBox(char, list(char_bbox)))
            logging.debug(f"char: {char}, bbox: {char_bbox}")

            y_cursor -= 5  # Spacing

        return image, char_boxes

    def render_text(self,
                   text: str,
                   font: ImageFont.FreeTypeFont,
                   direction: str) -> Tuple[Image.Image, List[CharacterBox]]:
        """
        Render text in specified direction with character-level bboxes.

        Args:
            text: Text to render
            font: Font to use
            direction: Text direction ('left_to_right', 'right_to_left', etc.)

        Returns:
            Tuple of (image, character_boxes)
        """
        if direction == 'left_to_right':
            return self.render_left_to_right(text, font)
        elif direction == 'right_to_left':
            return self.render_right_to_left(text, font)
        elif direction == 'top_to_bottom':
            return self.render_top_to_bottom(text, font)
        elif direction == 'bottom_to_top':
            return self.render_bottom_to_top(text, font)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def generate_image(self,
                      text: str,
                      font_path: str,
                      font_size: int,
                      direction: str) -> Tuple[Image.Image, List[List[float]], str]:
        """
        Generate a single synthetic OCR image with augmentations.

        Args:
            text: Text to render
            font_path: Path to font file
            font_size: Font size in points
            direction: Text direction

        Returns:
            Tuple of (augmented_image, character_bboxes, text)
        """
        # Load font
        font = self.load_font(font_path, font_size)

        # Render text with character bboxes
        image, char_boxes = self.render_text(text, font, direction)

        # Extract just the bbox coordinates
        char_bboxes = [box.bbox for box in char_boxes]

        # Apply augmentations
        augmented_image, augmented_bboxes = apply_augmentations(
            image, char_bboxes, self.background_images
        )

        return augmented_image, augmented_bboxes, text


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


def can_font_render_text(font_path: str, sample_text: str, min_coverage: float = 0.9) -> Tuple[bool, float]:
    """
    Check if a font can render the given sample text adequately.

    Uses a pragmatic heuristic approach: renders each character and checks if
    the result has visible content. Accepts system font fallback rendering,
    which is acceptable for OCR training data generation.

    Args:
        font_path: Path to font file
        sample_text: Sample characters to test
        min_coverage: Minimum fraction of characters that must render successfully (default 0.9)

    Returns:
        Tuple of (can_render, coverage_ratio)
    """
    if not sample_text:
        return True, 1.0

    try:
        # Load font at a reasonable test size
        font = ImageFont.truetype(font_path, size=32)

        rendered_count = 0
        total_count = len(sample_text)

        for char in sample_text:
            # Create a small test image
            test_img = Image.new('L', (60, 60), 255)
            test_draw = ImageDraw.Draw(test_img)

            # Get bounding box
            bbox = test_draw.textbbox((15, 15), char, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Render the character
            test_draw.text((15, 15), char, font=font, fill=0)

            # Check if character rendered with visible content
            # Count darker pixels (actual glyph content)
            pixels = list(test_img.getdata())
            dark_pixels = sum(1 for p in pixels if p < 240)

            # Heuristic: A real glyph (even fallback) should have:
            # 1. Non-zero bounding box dimensions
            # 2. At least a few dark pixels indicating actual rendering
            # Very permissive threshold to accept system font fallback
            if width > 0 and height > 0 and dark_pixels >= 3:
                rendered_count += 1
            else:
                logging.debug(f"Character '{char}' may not render properly: "
                            f"bbox={width}x{height}, dark_pixels={dark_pixels}")

        coverage = rendered_count / total_count if total_count > 0 else 0.0
        return coverage >= min_coverage, coverage

    except Exception as e:
        logging.debug(f"Error testing font {os.path.basename(font_path)}: {e}")
        return False, 0.0


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
        for font_path in font_files:
            can_render, coverage = can_font_render_text(font_path, sample_chars, min_coverage=0.9)
            if can_render:
                compatible_fonts.append(font_path)
                logging.debug(f"Font {os.path.basename(font_path)}: {coverage*100:.1f}% coverage")
            else:
                logging.warning(f"Skipping font {os.path.basename(font_path)}: insufficient glyph coverage ({coverage*100:.1f}%)")

        if not compatible_fonts:
            logging.error(f"No fonts can render the corpus text. Found {len(font_files)} valid fonts but none support the required characters.")
            sys.exit(1)

        font_files = compatible_fonts
        logging.info(f"Found {len(font_files)} fonts compatible with corpus characters")

    logging.info("Script finished.")

    # --- Initialize Generator ---
    generator = OCRDataGenerator(font_files, background_images)

    # --- Generation Loop ---
    if args.num_images > 0:
        start_time = time.time()

        # Determine starting image counter from existing images
        existing_images = [f for f in os.listdir(args.output_dir)
                         if f.startswith('image_') and f.endswith('.png')]
        image_counter = len(existing_images)

        labels_file = os.path.join(args.output_dir, 'labels.csv')

        # Append to labels.csv if it exists, otherwise create it with header
        file_mode = 'a' if os.path.exists(labels_file) else 'w'

        with open(labels_file, file_mode) as f:
            if file_mode == 'w':
                f.write('filename,text\n')

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

                try:
                    # Generate image with augmentations
                    augmented_image, augmented_bboxes, text = generator.generate_image(
                        text_line, font_path, font_size, args.text_direction
                    )

                    # Save image
                    image_filename = f'image_{image_counter:05d}.png'
                    image_path = os.path.join(args.output_dir, image_filename)
                    augmented_image.save(image_path)
                    logging.debug(f"Saved image to {image_path}")

                    # Create label with text and bboxes
                    label_data = {
                        "text": text,
                        "bboxes": [[float(coord) for coord in bbox] for bbox in augmented_bboxes]
                    }
                    f.write(f'{image_filename},{json.dumps(label_data)}\n')
                    image_counter += 1

                except Exception as e:
                    logging.error(f"Failed to generate image: {e}")
                    continue

        logging.info(f"Successfully generated {image_counter} images and a labels.csv file in {args.output_dir}")


if __name__ == '__main__':
    main()
