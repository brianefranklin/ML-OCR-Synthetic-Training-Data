import argparse
import yaml
import json
from pathlib import Path
from tqdm import tqdm

from src.batch_config import BatchConfig
from src.corpus_manager import CorpusManager
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager
from src.generation_orchestrator import GenerationOrchestrator
from src.generator import OCRDataGenerator

def main():
    """Main entry point for the OCR data generation script."""
    parser = argparse.ArgumentParser(description="Generate synthetic OCR data.")
    parser.add_argument("--batch-config", type=str, required=True, help="Path to the batch configuration YAML file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the generated images and labels.")
    parser.add_argument("--font-dir", type=str, required=True, help="Directory containing font files.")
    parser.add_argument("--background-dir", type=str, required=True, help="Directory containing background image files.")
    parser.add_argument("--corpus-dir", type=str, required=True, help="Directory containing corpus text files.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load batch configuration
    batch_config = BatchConfig.from_yaml(args.batch_config)

    # Initialize managers
    font_health_manager = FontHealthManager()
    background_manager = BackgroundImageManager(dir_weights={args.background_dir: 1.0})
    
    # Create a map of corpus file names to their full paths
    corpus_map = {f.name: str(f) for f in Path(args.corpus_dir).rglob('*.txt')}

    # Get a list of all available fonts
    all_fonts = [str(p) for p in Path(args.font_dir).rglob('*.ttf')]

    # Initialize the main components
    orchestrator = GenerationOrchestrator(
        batch_config=batch_config,
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager
    )
    generator = OCRDataGenerator()

    print(f"Generating {batch_config.total_images} images...")
    
    # Create the full list of generation tasks
    tasks = orchestrator.create_task_list(min_text_len=10, max_text_len=50)

    # Main generation loop
    for i, task in enumerate(tqdm(tasks, desc="Generating Images")):
        # Generate a plan for this task
        plan = generator.plan_generation(
            spec=task.source_spec,
            text=task.text,
            font_path=task.font_path,
            background_manager=background_manager
        )

        # Generate the image from the plan
        image, bboxes = generator.generate_from_plan(plan)

        # Save the image and the label file
        image_path = output_path / f"image_{i:05d}.png"
        label_path = output_path / f"image_{i:05d}.json"

        image.save(image_path)

        # Add the final bounding boxes to the plan for the label file
        plan["bboxes"] = bboxes
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=4)

    print("Generation complete.")

if __name__ == "__main__":
    main()
