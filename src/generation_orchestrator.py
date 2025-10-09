"""
Generation orchestrator for batch OCR data generation.

Handles batch configuration, font health filtering, and coordinated
generation across multiple batches with different parameters.
"""

import os
import random
import logging
from font_utils import can_font_render_text


# Global font health manager (set by main.py)
_font_health_manager = None


def set_font_health_manager(manager):
    """Set the global font health manager instance."""
    global _font_health_manager
    _font_health_manager = manager


def generate_with_batches(batch_config, font_files, background_images, args, OCRDataGenerator):
    """
    Generate images using batch configuration.

    Args:
        batch_config: BatchConfig object with specifications
        font_files: List of validated font paths
        background_images: List of background image paths
        args: Command line arguments
        OCRDataGenerator: OCRDataGenerator class (passed to avoid circular import)
    """
    from batch_config import BatchManager
    from corpus_manager import CorpusManager
    from canvas_placement import save_label_json

    # Filter fonts using health manager
    if _font_health_manager:
        healthy_fonts = _font_health_manager.get_available_fonts(font_files)
        if not healthy_fonts:
            logging.error("No healthy fonts available for generation")
            return
        logging.info(f"Using {len(healthy_fonts)}/{len(font_files)} healthy fonts")
        font_files = healthy_fonts

    # Initialize batch manager
    batch_manager = BatchManager(batch_config, font_files)

    # Prepare output
    existing_images = [f for f in os.listdir(args.output_dir)
                      if f.startswith('image_') and f.endswith('.png')]
    image_counter = len(existing_images)

    # Track corpora per batch
    batch_corpora = {}

    # Track generation attempts and successes
    successful_count = 0
    target_count = batch_config.total_images
    attempt_count = 0
    max_attempts = target_count * 3  # Allow up to 3x attempts (safety limit)
    failed_attempts = 0

    logging.info(f"Starting batch generation of {target_count} images")

    # Keep generating until we reach target (with safety limit)
    while successful_count < target_count and attempt_count < max_attempts:
        task = batch_manager.get_next_task()
        if task is None:
            # All batches have reached their targets
            break

        attempt_count += 1

        # Get or create corpus manager for this batch
        batch_name = task['batch_name']
        if batch_name not in batch_corpora:
            # Support corpus_file, corpus_dir, or corpus_pattern
            corpus_spec = task.get('corpus_file') or task.get('corpus_dir') or task.get('corpus_pattern')

            if not corpus_spec:
                # Fall back to CLI args
                if args.text_file:
                    corpus_spec = args.text_file
                else:
                    corpus_spec = args.text_dir

            # Create corpus manager based on what we have
            if corpus_spec and os.path.isdir(corpus_spec):
                # Directory mode
                pattern = task.get('text_pattern', '*.txt')
                batch_corpora[batch_name] = CorpusManager.from_directory(
                    corpus_spec,
                    pattern=pattern,
                    weights=task.get('corpus_weights'),
                    seed=batch_config.seed
                )
            elif corpus_spec and os.path.isfile(corpus_spec):
                # File mode
                batch_corpora[batch_name] = CorpusManager([corpus_spec], seed=batch_config.seed)
            else:
                # Pattern mode
                batch_corpora[batch_name] = CorpusManager.from_pattern(
                    corpus_spec,
                    weights=task.get('corpus_weights'),
                    seed=batch_config.seed
                )

        corpus_mgr = batch_corpora[batch_name]

        font_path = task['font_path']

        # Initialize generator with task-specific background images
        generator = OCRDataGenerator([font_path], background_images)

        # Extract text from corpus manager
        text_line = corpus_mgr.extract_text_segment(
            task['min_text_length'], task['max_text_length']
        )

        if not text_line:
            logging.warning(f"Could not generate text for batch '{batch_name}'. Skipping.")
            failed_attempts += 1
            continue

        # Check if font can render this text
        if not can_font_render_text(task['font_path'], text_line, frozenset(text_line)):
            logging.debug(f"Font {os.path.basename(font_path)} cannot render text for batch '{batch_name}'. Trying different task.")
            failed_attempts += 1
            continue

        # Font can render text, proceed with generation
        # Generate font size
        font_size = random.randint(28, 40)

        try:
            # Generate image with augmentations and canvas placement
            final_image, metadata, text, augmentations_applied = generator.generate_image(
                text_line, font_path, font_size, task['text_direction'],
                seed=batch_config.seed,
                curve_type=task.get('curve_type', 'none'),
                curve_intensity=task.get('curve_intensity', 0.0),
                overlap_intensity=task.get('overlap_intensity', 0.0),
                ink_bleed_intensity=task.get('ink_bleed_intensity', 0.0),
                effect_type=task.get('effect_type', 'none'),
                effect_depth=task.get('effect_depth', 0.5),
                light_azimuth=task.get('light_azimuth', 135.0),
                light_elevation=task.get('light_elevation', 45.0),
                text_color_mode=task.get('text_color_mode', 'uniform'),
                color_palette=task.get('color_palette', 'realistic_dark'),
                custom_colors=task.get('custom_colors'),
                background_color=task.get('background_color', 'auto'),
                canvas_enabled=True,
                canvas_min_padding=task.get('canvas_min_padding', 10),
                canvas_placement=task.get('canvas_placement', 'weighted_random'),
                canvas_max_megapixels=task.get('canvas_max_megapixels', 12.0)
            )

            # Save image
            image_filename = f'image_{image_counter:05d}.png'
            image_path = os.path.join(args.output_dir, image_filename)
            final_image.save(image_path)

            # Create generation_params dictionary
            generation_params = {
                'seed': batch_config.seed,
                'text': text,
                'font_path': font_path,
                'font_size': font_size,
                'text_direction': task['text_direction'],
                'curve_type': task.get('curve_type', 'none'),
                'curve_intensity': task.get('curve_intensity', 0.0),
                'overlap_intensity': task.get('overlap_intensity', 0.0),
                'ink_bleed_intensity': task.get('ink_bleed_intensity', 0.0),
                'effect_type': task.get('effect_type', 'none'),
                'effect_depth': task.get('effect_depth', 0.5),
                'light_azimuth': task.get('light_azimuth', 135.0),
                'light_elevation': task.get('light_elevation', 45.0),
                'text_color_mode': task.get('text_color_mode', 'uniform'),
                'color_palette': task.get('color_palette', 'realistic_dark'),
                'custom_colors': task.get('custom_colors'),
                'background_color': task.get('background_color', 'auto'),
                'augmentations': augmentations_applied
            }

            # Save JSON label
            json_filename = f'image_{image_counter:05d}.json'
            json_path = os.path.join(args.output_dir, json_filename)
            save_label_json(json_path, image_filename, text, metadata, generation_params)

            image_counter += 1
            successful_count += 1

            # Record success in font health manager
            if _font_health_manager:
                _font_health_manager.record_success(font_path, text_line)

            # Mark task as successfully completed in batch manager
            batch_manager.mark_task_success(task)

            logging.debug(f"Batch '{batch_name}' ({task['progress']}): "
                        f"{os.path.basename(font_path)}, direction={task['text_direction']}")

        except OSError as e:
            failed_attempts += 1
            # Record specific failure types
            if _font_health_manager:
                if "execution context too long" in str(e):
                    _font_health_manager.record_failure(font_path, reason="freetype_error")
                else:
                    _font_health_manager.record_failure(font_path, reason="os_error")
            if "execution context too long" in str(e):
                logging.warning(f"Skipping font {os.path.basename(font_path)} due to FreeType error: {e}")
                continue
            else:
                logging.error(f"Failed to generate image for batch '{batch_name}': {e}")
                continue
        except Exception as e:
            failed_attempts += 1
            # Record general failures
            if _font_health_manager:
                _font_health_manager.record_failure(font_path, reason=type(e).__name__)
            logging.error(f"Failed to generate image for batch '{batch_name}': {e}")
            continue

    # Save font health state and report
    if _font_health_manager:
        _font_health_manager.save_state()
        health_report = _font_health_manager.get_summary_report()
        logging.info(f"Font health summary: {health_report}")

    # Report final statistics
    logging.info(f"\n{batch_manager.get_progress_summary()}")
    logging.info(f"Successfully generated {successful_count}/{target_count} images "
                f"({attempt_count} attempts, {failed_attempts} failures)")

    if successful_count < target_count:
        logging.warning(f"Generated {successful_count} images, but target was {target_count}. "
                       f"{target_count - successful_count} images missing due to errors.")

    logging.info(f"Images saved to {args.output_dir}")
