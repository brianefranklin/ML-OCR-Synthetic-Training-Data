"""
OCR Synthetic Data Generator - Main Entry Point

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
from PIL import ImageFont

# Import our modularized components
from generator import OCRDataGenerator, CharacterBox
from font_health_manager import FontHealthManager
from font_utils import can_font_render_text, extract_sample_characters, set_font_health_manager as set_font_utils_health_manager
from generation_orchestrator import generate_with_batches, set_font_health_manager as set_orchestrator_health_manager


def setup_logging(log_level: str, log_file: str) -> None:
    """Configure logging with both file and console output."""
    # Create parent directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w',
        force=True
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
                       help='Path to a single text corpus file (backwards compatible).')
    parser.add_argument('--text-dir', type=str, default=config.get('text_dir'),
                       help='Path to directory containing corpus text files (for large-scale generation).')
    parser.add_argument('--text-pattern', type=str, default=config.get('text_pattern', '*.txt'),
                       help='Glob pattern for corpus files when using --text-dir (default: *.txt).')
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
    parser.add_argument('--log-dir', type=str, default=config.get('log_dir', 'logs'),
                       help='Directory for log files (timestamped log files will be created here).')
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

    # --- Generate Timestamp for This Run ---
    run_timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

    # --- Configure Logging ---
    # Create timestamped log file in log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_file_path = os.path.join(args.log_dir, f'generation_{run_timestamp}.log')
    setup_logging(args.log_level, log_file_path)
    logging.info("Script started.")

    # --- Clear Output Directory (if requested) ---
    if args.clear_output:
        if not clear_output_directory(args.output_dir, args.force):
            return
        if args.num_images == 0:
            logging.info("Output directory cleared. Exiting as num_images is 0.")
            return

    # --- Validate Essential Arguments ---
    # Handle corpus specification priority: explicit CLI args override config
    if args.text_dir:
        # If text_dir is explicitly specified, ignore text_file
        args.text_file = None
    elif not args.text_file and not args.text_dir:
        logging.error("Error: Either --text-file or --text-dir must be specified.")
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

    # Initialize Font Health Manager
    # Save font health to timestamped file in log directory
    font_health_path = os.path.join(args.log_dir, f'font_health_{run_timestamp}.json')
    font_health_manager = FontHealthManager(
        health_file=font_health_path,
        min_health_threshold=30.0,
        success_increment=1.0,
        failure_decrement=10.0,
        cooldown_base_seconds=300.0,  # 5 minutes
        auto_save_interval=50
    )
    logging.info(f"Font health tracking enabled (saving to {font_health_path})")

    # Set the global font health manager in other modules
    set_font_utils_health_manager(font_health_manager)
    set_orchestrator_health_manager(font_health_manager)

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
        font_name = os.path.basename(font_path)

        try:
            # Try to load the font to validate it
            ImageFont.truetype(font_path, size=20)
            font_files.append(font_path)
        except Exception as e:
            logging.warning(f"Skipping invalid font {font_name}: {e}")

    if not font_files:
        logging.error("Error: No valid fonts found.")
        sys.exit(1)

    logging.info(f"Loaded {len(font_files)} fonts")

    # Load background images (optional)
    background_images = []
    if args.backgrounds_dir and os.path.isdir(args.backgrounds_dir):
        background_images = [os.path.join(args.backgrounds_dir, f)
                           for f in os.listdir(args.backgrounds_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Loaded {len(background_images)} background images")

    # --- Check for Batch Configuration ---
    if args.batch_config:
        # Batch mode
        from batch_config import BatchConfig
        batch_config = BatchConfig.from_yaml(args.batch_config)
        logging.info(f"Loaded batch configuration from {args.batch_config}")
        logging.info(f"Total images to generate across all batches: {batch_config.total_images}")

        # Call batch generation with OCRDataGenerator class passed in
        generate_with_batches(batch_config, font_files, background_images, args, OCRDataGenerator)

    else:
        # --- Standard Generation Mode ---
        # Load corpus
        from corpus_manager import CorpusManager

        if args.text_dir:
            # Directory mode
            corpus_manager = CorpusManager.from_directory(args.text_dir, pattern=args.text_pattern)
        elif args.text_file:
            # Single file mode
            corpus_manager = CorpusManager([args.text_file])
        else:
            logging.error("No text source specified")
            sys.exit(1)

        # Sample character set for validation
        sample_corpus = corpus_manager.extract_text_segment(1, 10000)
        if not sample_corpus:
            # Fallback: try shorter segment
            sample_corpus = corpus_manager.extract_text_segment(1, 100)
        if not sample_corpus:
            logging.error("Corpus must contain at least 1 character of text.")
            sys.exit(1)
        character_set = frozenset(extract_sample_characters(sample_corpus))
        logging.info(f"Corpus loaded with {len(character_set)} unique characters")

        # Filter fonts by whether they can render the corpus
        valid_font_files = []
        for font_path in font_files:
            if can_font_render_text(font_path, sample_corpus, character_set):
                valid_font_files.append(font_path)
        font_files = valid_font_files

        if not font_files:
            logging.error("Error: No fonts can render the specified text corpus.")
            sys.exit(1)

        logging.info(f"{len(font_files)} fonts can render the corpus text")

        # Initialize generator
        generator = OCRDataGenerator(font_files, background_images)

        # Track execution time
        start_time = time.time()

        # Prepare output
        existing_images = [f for f in os.listdir(args.output_dir)
                         if f.startswith('image_') and f.endswith('.png')]
        image_counter = len(existing_images)

        logging.info(f"Generating up to {args.num_images} images (starting from image_{image_counter:05d})...")

        for i in range(args.num_images):
            # --- Time Limit Check ---
            if args.max_execution_time and (time.time() - start_time) > args.max_execution_time:
                logging.info(f"\nTime limit of {args.max_execution_time} seconds reached. Stopping generation.")
                break

            # Extract text segment from corpus manager
            text_line = corpus_manager.extract_text_segment(
                args.min_text_length, args.max_text_length
            )

            if not text_line:
                logging.warning(f"Could not generate text of minimum length {args.min_text_length}. Skipping image.")
                continue

            logging.debug(f"Selected text: {text_line}")

            # Select font with health awareness
            if args.font_name:
                font_path = os.path.join(args.fonts_dir, args.font_name)
                if not os.path.exists(font_path):
                    logging.error(f"Error: Font file {args.font_name} not found in {args.fonts_dir}")
                    continue
                # Check if specified font is healthy
                if font_health_manager and not font_health_manager.get_available_fonts([font_path]):
                    logging.warning(f"Specified font {args.font_name} is unhealthy, skipping")
                    continue
            else:
                # Select from healthy fonts only
                if font_health_manager:
                    healthy_fonts = font_health_manager.get_available_fonts(font_files)
                    if not healthy_fonts:
                        logging.error("No healthy fonts available")
                        break
                    font_path = font_health_manager.select_font_weighted(healthy_fonts)
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
                final_image, metadata, text, augmentations_applied = generator.generate_image(
                    text_line, font_path, font_size, args.text_direction,
                    curve_type='none',  # Not specified in CLI args for standard mode
                    curve_intensity=0.0,  # Not specified in CLI args for standard mode
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

                # Create generation_params dictionary for standard mode
                generation_params = {
                    'text': text,
                    'font_path': font_path,
                    'font_size': font_size,
                    'text_direction': args.text_direction,
                    'curve_type': 'none',
                    'curve_intensity': 0.0,
                    'overlap_intensity': args.overlap_intensity,
                    'ink_bleed_intensity': args.ink_bleed_intensity,
                    'effect_type': args.effect_type,
                    'effect_depth': args.effect_depth,
                    'light_azimuth': args.light_azimuth,
                    'light_elevation': args.light_elevation,
                    'text_color_mode': args.text_color_mode,
                    'color_palette': args.color_palette,
                    'custom_colors': custom_colors,
                    'background_color': args.background_color,
                    'augmentations': augmentations_applied
                }

                # Save JSON label
                from canvas_placement import save_label_json
                json_filename = f'image_{image_counter:05d}.json'
                json_path = os.path.join(args.output_dir, json_filename)
                save_label_json(json_path, image_filename, text, metadata, generation_params)
                logging.debug(f"Saved label to {json_path}")

                # Record success in font health manager
                if font_health_manager:
                    font_health_manager.record_success(font_path, text_line)

                image_counter += 1

            except Exception as e:
                # Record failure in font health manager
                if font_health_manager:
                    reason = "render_error" if "render" in str(e).lower() else type(e).__name__
                    font_health_manager.record_failure(font_path, reason=reason)
                logging.error(f"Failed to generate image: {e}")
                continue

        if image_counter > 0:
            logging.info(f"Successfully generated {image_counter} images with JSON labels in {args.output_dir}")
        else:
            logging.error("Failed to generate any images. The corpus may be too small for the specified text length.")
            sys.exit(1)

    # Save final font health state and report
    if font_health_manager:
        font_health_manager.save_state()
        report = font_health_manager.get_summary_report()
        logging.info(f"Final font health report: {report}")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()
