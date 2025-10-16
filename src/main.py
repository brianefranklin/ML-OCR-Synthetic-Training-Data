import argparse
import yaml
import json
import uuid
import sys
import logging
import multiprocessing
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image

# Set a new limit for image size to handle large backgrounds (e.g., 100MP) and avoid DecompressionBombWarning.
Image.MAX_IMAGE_PIXELS = 100000001

from src.batch_config import BatchConfig
from src.corpus_manager import CorpusManager
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager
from src.generation_orchestrator import GenerationOrchestrator, GenerationTask
from src.generator import OCRDataGenerator
from src.batch_validation import BatchValidator, ValidationError
from src.checkpoint_manager import CheckpointManager

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types.

    This encoder converts NumPy integers and floats to their Python equivalents
    to enable JSON serialization of data structures containing NumPy types.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_image_from_task(
    args: Tuple[GenerationTask, int, Any]
) -> Tuple[int, Image.Image, Dict[str, Any], Optional[str]]:
    """Worker function for parallel image generation.

    This function is designed to be used with multiprocessing.Pool for
    parallel generation of images from tasks. It generates a plan, renders
    the image, and applies all effects and augmentations.

    For deterministic generation, this function sets the random seed based on
    the task index, ensuring that the same task index always produces the
    same output regardless of execution order or parallel vs sequential processing.

    Args:
        args: A tuple containing:
            - task: GenerationTask containing spec, text, font_path, background_path.
            - index: Integer index of this task in the overall batch.
            - background_manager: Optional BackgroundImageManager for selecting backgrounds.

    Returns:
        A tuple containing:
            - index: The task index (for maintaining order).
            - image: The generated PIL Image (or None if generation failed).
            - plan: Dictionary containing the generation plan with bboxes added (or None if failed).
            - error: Error message if generation failed, None otherwise.

    Note:
        This function must be defined at module level (not nested) for
        multiprocessing compatibility due to pickle requirements.

    Examples:
        >>> from src.generation_orchestrator import GenerationTask
        >>> from src.batch_config import BatchSpecification
        >>> spec = BatchSpecification(...)
        >>> task = GenerationTask(spec, "hello", "/path/to/font.ttf", None)
        >>> idx, image, plan, error = generate_image_from_task((task, 0, None))
    """
    import random

    task, index, background_manager = args

    try:
        # Set random seed based on task index for deterministic generation
        # This ensures the same index always produces the same output
        random.seed(index)
        np.random.seed(index)

        # Create a generator instance (each worker needs its own)
        generator = OCRDataGenerator()

        # Generate a plan for this task
        plan: Dict[str, Any] = generator.plan_generation(
            spec=task.source_spec,
            text=task.text,
            font_path=task.font_path,
            background_manager=background_manager
        )

        # Generate the image from the plan
        image, bboxes = generator.generate_from_plan(plan)

        # Add the final bounding boxes to the plan
        plan["bboxes"] = bboxes

        return index, image, plan, None

    except Exception as e:
        # Catch ALL exceptions to prevent worker crash
        # This includes OSError, IOError, ValueError, TypeError, and any unexpected errors
        error_msg = f"Failed to generate image {index} with font '{task.font_path}': {type(e).__name__}: {e}"
        return index, None, None, error_msg


def save_image_and_label(
    args: Tuple[Image.Image, Dict[str, Any], Path, Path]
) -> None:
    """Worker function for parallel image and label saving with retry logic.

    This function is designed to be used with multiprocessing.Pool for
    parallel saving of images and their corresponding JSON label files.
    It includes retry logic with exponential backoff to handle transient
    filesystem issues (e.g., virtiofs permission errors).

    Args:
        args: A tuple containing:
            - image: PIL Image to save (RGBA, RGB, or L mode).
            - plan: Dictionary containing generation plan and bboxes. May contain
                   NumPy data types which will be automatically converted to
                   native Python types during JSON serialization.
            - image_path: Path where image should be saved (typically .png).
            - label_path: Path where JSON label should be saved (typically .json).

    Returns:
        None. Files are written to disk as side effects.

    Note:
        This function must be defined at module level (not nested) for
        multiprocessing compatibility due to pickle requirements.

        All exceptions are caught and logged to prevent worker crashes.
        Failed saves are logged as warnings but don't crash the worker.

        The retry logic is specifically designed to handle virtiofs filesystem
        limitations. Virtiofs (used by Docker/devcontainers to mount host
        directories) can experience transient PermissionErrors under high-concurrency
        workloads when multiple parallel I/O workers write files simultaneously.
        The FUSE layer's file handle management and locking mechanisms struggle
        with rapid parallel operations from multiprocessing pools. The exponential
        backoff with random jitter gives the filesystem time to release locks
        and complete pending operations before retrying.

    Examples:
        >>> from PIL import Image
        >>> from pathlib import Path
        >>> image = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        >>> plan = {"text": "hello", "bboxes": []}
        >>> save_image_and_label((image, plan, Path("out.png"), Path("out.json")))
    """
    import time
    import random
    import logging

    logger = logging.getLogger(__name__)
    image, plan, image_path, label_path = args

    max_retries = 3
    base_delay = 0.1  # 100ms base delay

    # Try to save with retries
    for attempt in range(max_retries):
        try:
            # Save image as PNG
            image.save(image_path)

            # Save label as JSON with NumpyEncoder for numpy type handling
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=4, cls=NumpyEncoder)

            # Success - return immediately
            return

        except PermissionError as e:
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.05)
                logger.warning(
                    f"PermissionError saving {image_path.name}, "
                    f"retry {attempt + 1}/{max_retries} after {delay:.3f}s: {e}"
                )
                time.sleep(delay)
            else:
                # Final attempt failed
                logger.error(
                    f"Failed to save {image_path.name} after {max_retries} attempts: {e}"
                )
                return  # Don't crash worker, just skip this file

        except Exception as e:
            # Catch all other exceptions to prevent worker crash
            logger.error(
                f"Unexpected error saving {image_path.name}: {type(e).__name__}: {e}"
            )
            return  # Don't crash worker, just skip this file


def main():
    """Main entry point for the OCR data generation script.

    This function parses command-line arguments, initializes all necessary
    managers and components, and runs the main image generation loop.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic OCR data.")
    parser.add_argument("--batch-config", type=str, required=True, help="Path to the batch configuration YAML file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the generated images and labels.")
    parser.add_argument("--font-dir", type=str, required=True, help="Directory containing font files.")
    parser.add_argument("--background-dir", type=str, required=True, help="Directory containing background image files.")
    parser.add_argument("--corpus-dir", type=str, required=True, help="Directory containing corpus text files.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Directory for log files (default: ./logs)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker processes for parallel I/O (0 or 1 = sequential, default: 4)")
    parser.add_argument("--io-batch-size", type=int, default=10,
                        help="Number of images to accumulate before parallel I/O save (default: 10)")
    parser.add_argument("--generation-workers", type=int, default=0,
                        help="Number of worker processes for parallel image generation (0 or 1 = sequential, default: 0)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of images to generate per chunk in streaming mode (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous incomplete generation (skips existing files)")
    args = parser.parse_args()

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"generation_{timestamp}"

    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{run_id}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Print and log startup message with timestamp
    print(f"\n{'='*60}")
    print(f"Starting OCR generation run: {run_id}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    logger.info(f"Starting OCR generation run: {run_id}")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Configuration: {args.batch_config}")

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Load batch configuration
    batch_config: BatchConfig = BatchConfig.from_yaml(args.batch_config)
    logger.info(f"Loaded batch config: {batch_config.total_images} total images, {len(batch_config.specifications)} specifications")

    # Validate configuration before proceeding
    logger.info("Validating batch configuration...")
    try:
        # Load raw config for validation
        with open(args.batch_config, 'r') as f:
            raw_config = yaml.safe_load(f)

        validator = BatchValidator(
            config=raw_config,
            corpus_dir=args.corpus_dir,
            font_dir=args.font_dir,
            background_dir=args.background_dir
        )
        validator.validate()
        logger.info("Configuration validation passed")
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        print(f"\n{'='*60}")
        print("ERROR: Configuration validation failed")
        print(f"{'='*60}")
        print(f"\n{e}\n")
        print("Please fix the configuration and try again.")
        print(f"{'='*60}\n")
        sys.exit(1)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        output_dir=str(output_path),
        config=raw_config
    )

    # Handle resume mode
    completed_indices = set()
    if args.resume:
        logger.info("Resume mode enabled - checking for existing checkpoint...")
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            logger.info(f"Found checkpoint: {checkpoint_data['completed_images']} images previously completed")
            completed_indices = checkpoint_manager.get_completed_indices()
            logger.info(f"Scanned output directory: {len(completed_indices)} existing images found")
            if len(completed_indices) > 0:
                print(f"Resuming generation - skipping {len(completed_indices)} existing images")
        else:
            logger.info("No previous checkpoint found - starting fresh generation")
            print("No previous checkpoint found - starting fresh generation")

    # Initialize managers
    font_health_manager = FontHealthManager()
    background_manager = BackgroundImageManager(dir_weights={args.background_dir: 1.0})
    logger.debug("Initialized font health manager and background manager")

    # Create a map of corpus file names to their full paths
    corpus_map: Dict[str, str] = {str(f.relative_to(args.corpus_dir)): str(f) for f in Path(args.corpus_dir).rglob('*.txt')}
    logger.info(f"Found {len(corpus_map)} corpus files in {args.corpus_dir}")

    # Get a list of all available fonts
    all_fonts: List[str] = [str(p) for p in Path(args.font_dir).rglob('*.ttf')]
    logger.info(f"Found {len(all_fonts)} fonts in {args.font_dir}")

    # Initialize the main components
    orchestrator = GenerationOrchestrator(
        batch_config=batch_config,
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager
    )
    generator = OCRDataGenerator()
    logger.debug("Initialized orchestrator and generator")

    print(f"Generating {batch_config.total_images} images...")

    # Pre-generate a list of unique filenames for the entire batch using UUID4
    # UUID4 has negligible collision probability (<10^-15 for 10k images)
    unique_filenames = [str(uuid.uuid4()) for _ in range(batch_config.total_images)]

    # Create the full list of generation tasks
    tasks: List[GenerationTask] = orchestrator.create_task_list(
        min_text_len=10,
        max_text_len=50,
        unique_filenames=unique_filenames
    )
    logger.info(f"Created {len(tasks)} generation tasks")

    # Determine if we should use parallel generation
    use_parallel_generation = args.generation_workers > 1
    use_parallel_io = args.workers > 1

    if use_parallel_generation:
        logger.info(f"Using streaming parallel generation with {args.generation_workers} workers, chunk size {args.chunk_size}")

        # Filter out completed tasks in resume mode
        if args.resume and len(completed_indices) > 0:
            remaining_tasks = [(i, task) for i, task in enumerate(tasks) if i not in completed_indices]
            logger.info(f"Resume mode: {len(remaining_tasks)} tasks remaining, {len(completed_indices)} already completed")
        else:
            remaining_tasks = list(enumerate(tasks))

        # Create pools
        gen_pool = multiprocessing.Pool(processes=args.generation_workers)
        io_pool = multiprocessing.Pool(processes=args.workers) if use_parallel_io else None

        if use_parallel_io:
            logger.info(f"Using parallel I/O with {args.workers} workers, batch size {args.io_batch_size}")
        else:
            logger.info("Using sequential I/O")

        # Overall progress bar (total includes already completed)
        total_images = len(tasks)
        total_progress = tqdm(initial=len(completed_indices), total=total_images, desc="Processing Images")

        # Track completed count for checkpointing
        images_completed = len(completed_indices)

        try:
            # Process remaining tasks in chunks
            for chunk_offset in range(0, len(remaining_tasks), args.chunk_size):
                chunk_end_offset = min(chunk_offset + args.chunk_size, len(remaining_tasks))
                chunk_items = remaining_tasks[chunk_offset:chunk_end_offset]

                # Extract indices and tasks
                chunk_indices = [idx for idx, _ in chunk_items]
                chunk_tasks = [task for _, task in chunk_items]

                logger.debug(f"Processing chunk {chunk_offset}-{chunk_end_offset-1} ({len(chunk_tasks)} images)")

                # Prepare arguments for this chunk
                chunk_args = [
                    (task, idx, background_manager)
                    for idx, task in zip(chunk_indices, chunk_tasks)
                ]

                # Generate chunk in parallel with error handling
                try:
                    chunk_results = gen_pool.map(generate_image_from_task, chunk_args)
                except Exception as e:
                    logger.error(f"Generation pool.map() failed for chunk {chunk_offset}-{chunk_end_offset-1}: {type(e).__name__}: {e}")
                    logger.warning(f"Skipping {len(chunk_tasks)} images in failed chunk, continuing with next chunk")
                    total_progress.update(len(chunk_tasks))
                    continue

                # Save chunk immediately, filtering out failed generations
                chunk_saved_count = 0
                chunk_failed_count = 0
                if use_parallel_io:
                    # Parallel I/O: batch saves within chunk
                    io_batch_size = args.io_batch_size
                    save_tasks: List[Tuple[Image.Image, Dict[str, Any], Path, Path]] = []

                    for idx, image, plan, error in chunk_results:
                        if error is not None:
                            # Generation failed - log and skip
                            logger.warning(error)
                            chunk_failed_count += 1
                            total_progress.update(1)
                            continue

                        task = tasks[idx]
                        image_path = output_path / f"{task.output_filename}.png"
                        label_path = output_path / f"{task.output_filename}.json"
                        save_tasks.append((image, plan, image_path, label_path))

                        # Save batch when full
                        if len(save_tasks) >= io_batch_size:
                            try:
                                io_pool.map(save_image_and_label, save_tasks)
                                chunk_saved_count += len(save_tasks)
                                total_progress.update(len(save_tasks))
                            except Exception as e:
                                logger.error(f"I/O pool.map() failed for batch: {type(e).__name__}: {e}")
                                logger.warning(f"Skipping {len(save_tasks)} images in failed I/O batch")
                                total_progress.update(len(save_tasks))
                            finally:
                                save_tasks = []

                    # Save any remaining tasks at end of chunk
                    if len(save_tasks) > 0:
                        try:
                            io_pool.map(save_image_and_label, save_tasks)
                            chunk_saved_count += len(save_tasks)
                            total_progress.update(len(save_tasks))
                        except Exception as e:
                            logger.error(f"I/O pool.map() failed for final batch: {type(e).__name__}: {e}")
                            logger.warning(f"Skipping {len(save_tasks)} images in failed I/O batch")
                            total_progress.update(len(save_tasks))
                        finally:
                            save_tasks = []
                else:
                    # Sequential I/O
                    for idx, image, plan, error in chunk_results:
                        if error is not None:
                            # Generation failed - log and skip
                            logger.warning(error)
                            chunk_failed_count += 1
                            total_progress.update(1)
                            continue

                        task = tasks[idx]
                        image_path = output_path / f"{task.output_filename}.png"
                        label_path = output_path / f"{task.output_filename}.json"
                        image.save(image_path)
                        with open(label_path, 'w', encoding='utf-8') as f:
                            json.dump(plan, f, indent=4, cls=NumpyEncoder)
                        chunk_saved_count += 1
                        total_progress.update(1)

                # Update checkpoint after each chunk
                images_completed += chunk_saved_count
                checkpoint_manager.save_checkpoint(completed_images=images_completed)
                if chunk_failed_count > 0:
                    logger.info(f"Chunk complete: {chunk_saved_count} images saved, {chunk_failed_count} skipped due to errors, total {images_completed}/{total_images}")
                else:
                    logger.debug(f"Chunk complete: {chunk_saved_count} images, total {images_completed}/{total_images}")

        finally:
            # Clean up progress bar and pools
            total_progress.close()
            gen_pool.close()
            gen_pool.join()
            if io_pool:
                io_pool.close()
                io_pool.join()
            logger.debug("Closed all worker pools")
    else:
        # Sequential generation mode
        logger.info("Using sequential image generation")

        # Filter out completed tasks in resume mode
        if args.resume and len(completed_indices) > 0:
            remaining_tasks = [(i, task) for i, task in enumerate(tasks) if i not in completed_indices]
            logger.info(f"Resume mode: {len(remaining_tasks)} tasks remaining, {len(completed_indices)} already completed")
        else:
            remaining_tasks = list(enumerate(tasks))

        if use_parallel_io:
            logger.info(f"Using parallel I/O with {args.workers} workers, batch size {args.io_batch_size}")
            io_pool = multiprocessing.Pool(processes=args.workers)
            batch_size = args.io_batch_size
            save_tasks: List[Tuple[Image.Image, Dict[str, Any], Path, Path]] = []
        else:
            logger.info("Using sequential I/O")

        # Track completed count for checkpointing
        images_completed = len(completed_indices)

        # Sequential generation loop with progress bar
        progress = tqdm(remaining_tasks, desc="Generating Images", initial=len(completed_indices), total=len(tasks))
        for i, task in progress:
            logger.debug(f"Generating image {i+1}/{len(tasks)} (spec: {task.source_spec.name})")

            # Generate a plan for this task
            plan: Dict[str, Any] = generator.plan_generation(
                spec=task.source_spec,
                text=task.text,
                font_path=task.font_path,
                background_manager=background_manager
            )

            # Generate the image from the plan
            image, bboxes = generator.generate_from_plan(plan)

            # Add the final bounding boxes to the plan for the label file
            plan["bboxes"] = bboxes

            # Prepare file paths
            image_path = output_path / f"{task.output_filename}.png"
            label_path = output_path / f"{task.output_filename}.json"

            if use_parallel_io:
                # Add to batch for parallel saving
                save_tasks.append((image, plan, image_path, label_path))

                # Save batch when full or at end
                is_last = (i == remaining_tasks[-1][0])
                if len(save_tasks) >= batch_size or is_last:
                    try:
                        io_pool.map(save_image_and_label, save_tasks)
                        images_completed += len(save_tasks)
                    except Exception as e:
                        logger.error(f"I/O pool.map() failed in sequential mode: {type(e).__name__}: {e}")
                        logger.warning(f"Skipping {len(save_tasks)} images in failed I/O batch")
                    finally:
                        save_tasks = []
                    # Save checkpoint after each batch
                    checkpoint_manager.save_checkpoint(completed_images=images_completed)
            else:
                # Sequential mode (backwards compatible)
                image.save(image_path)
                with open(label_path, 'w', encoding='utf-8') as f:
                    json.dump(plan, f, indent=4, cls=NumpyEncoder)
                images_completed += 1

                # Save checkpoint every 10 images in sequential mode
                if images_completed % 10 == 0:
                    checkpoint_manager.save_checkpoint(completed_images=images_completed)

            # Log progress every 10 images
            if images_completed % 10 == 0:
                logger.info(f"Progress: {images_completed}/{len(tasks)} images generated")

        # Clean up I/O pool if used in sequential generation mode
        if use_parallel_io:
            io_pool.close()
            io_pool.join()
            logger.debug("Closed I/O pool")

    logger.info(f"Generation complete. Generated {len(tasks)} images to {output_path}")
    print("Generation complete.")

if __name__ == "__main__":
    main()