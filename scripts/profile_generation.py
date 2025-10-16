#!/usr/bin/env python3
"""Profile OCR data generation performance.

This script runs cProfile on the image generation process to identify
performance bottlenecks. Use this to optimize generation speed.

Usage:
    python scripts/profile_generation.py --config CONFIG_PATH --output PROFILE_OUTPUT

Example:
    python scripts/profile_generation.py \
        --config test_configs/comprehensive_test.yaml \
        --output profile_results.txt
"""

import argparse
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List
from src.batch_config import BatchConfig
from src.corpus_manager import CorpusManager
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager
from src.generation_orchestrator import GenerationOrchestrator
from src.generator import OCRDataGenerator


def run_generation(config_path: str, limit_images: int = 10) -> None:
    """Run a limited generation job for profiling.

    Args:
        config_path: Path to the batch configuration YAML file.
        limit_images: Maximum number of images to generate (default: 10).
    """
    # Load configuration
    batch_config = BatchConfig.from_yaml(config_path)

    # Limit images for profiling
    original_total = batch_config.total_images
    batch_config.total_images = min(limit_images, original_total)

    print(f"Profiling generation of {batch_config.total_images} images (limited from {original_total})")
    print(f"Config: {config_path}")
    print("-" * 60)

    # Initialize managers (minimal setup for profiling)
    # Note: These paths may need to be adjusted for your environment
    font_health_manager = FontHealthManager()
    background_manager = BackgroundImageManager(dir_weights={"./data.nosync/backgrounds": 1.0})

    corpus_map: Dict[str, str] = {}
    for spec in batch_config.specifications:
        if spec.corpus_file not in corpus_map:
            # Use placeholder if actual corpus not found
            corpus_path = Path("./data.nosync/corpus_text/ltr") / spec.corpus_file
            if corpus_path.exists():
                corpus_map[spec.corpus_file] = str(corpus_path)
            else:
                print(f"Warning: Corpus file {spec.corpus_file} not found, using placeholder")
                corpus_map[spec.corpus_file] = "./data.nosync/corpus_text/ltr/gutenberg_42671.txt"

    # Get available fonts
    font_dir = Path("./data.nosync/fonts")
    if font_dir.exists():
        all_fonts: List[str] = [str(p) for p in font_dir.rglob('*.ttf')]
    else:
        # Fallback to system fonts
        all_fonts = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    if not all_fonts:
        raise RuntimeError("No fonts found for profiling")

    # Initialize orchestrator
    orchestrator = GenerationOrchestrator(
        batch_config=batch_config,
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager
    )

    # Get task list
    tasks = orchestrator.create_task_list(min_text_len=10, max_text_len=50)

    # Initialize generator
    generator = OCRDataGenerator()

    # Generate images (this is what we profile)
    print("Starting generation...")
    for i, task in enumerate(tasks):
        plan = generator.plan_generation(
            spec=task.source_spec,
            text=task.text,
            font_path=task.font_path,
            background_manager=background_manager
        )

        image, bboxes = generator.generate_from_plan(plan)

        if i % 5 == 0:
            print(f"  Generated {i+1}/{len(tasks)} images...")

    print(f"Completed generation of {len(tasks)} images")


def main():
    """Main entry point for profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile OCR data generation performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to batch configuration YAML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profile_results.txt",
        help="Output file for profiling results (default: profile_results.txt)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of images to generate (default: 10)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="cumulative",
        choices=["cumulative", "time", "calls"],
        help="Sort profiling results by: cumulative time, time per call, or call count"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top functions to display (default: 20)"
    )

    args = parser.parse_args()

    # Run profiler
    print("=" * 60)
    print("OCR Data Generation Profiler")
    print("=" * 60)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        run_generation(args.config, limit_images=args.limit)
    except Exception as e:
        print(f"\nError during generation: {e}")
        raise
    finally:
        profiler.disable()

    # Analyze results
    print("\n" + "=" * 60)
    print("Profiling Results")
    print("=" * 60)

    # Create string buffer for output
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)

    # Sort and print statistics
    stats.sort_stats(args.sort)
    stats.print_stats(args.top)

    # Get the output
    profile_output = s.getvalue()

    # Print to console
    print(profile_output)

    # Save to file
    with open(args.output, 'w') as f:
        f.write("OCR Data Generation Profile\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Images generated: {args.limit}\n")
        f.write(f"Sort by: {args.sort}\n")
        f.write("=" * 60 + "\n\n")
        f.write(profile_output)

    print(f"\nFull results saved to: {args.output}")

    # Print summary statistics
    stats.sort_stats('cumulative')
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    total_time = 0
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        total_time += tt

    print(f"Total time: {total_time:.3f} seconds")
    print(f"Time per image: {total_time / args.limit:.3f} seconds")
    print(f"Images per second: {args.limit / total_time:.2f}")

    print("\nTop 5 functions by cumulative time:")
    stats.print_stats(5)


if __name__ == "__main__":
    main()
