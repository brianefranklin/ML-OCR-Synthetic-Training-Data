"""Orchestrates the generation of synthetic OCR data.

This module is the central hub that connects the batch configuration with all
the resource managers to produce a list of concrete generation tasks.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import fnmatch

from src.batch_config import BatchConfig, BatchSpecification, BatchManager
from src.corpus_manager import CorpusManager
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager

@dataclass
class GenerationTask:
    """Represents a single, complete set of parameters for generating one image.

    This is a simple data structure to hold the results of the orchestration
    process before they are passed to the final planning and generation stage.
    """
    index: int
    source_spec: BatchSpecification
    text: str
    font_path: str
    background_path: Optional[str]
    output_filename: str

class GenerationOrchestrator:
    """Orchestrates the creation of generation tasks.

    This class integrates the BatchManager, CorpusManager, FontHealthManager, and
    BackgroundImageManager to create a definitive list of tasks for the entire
    batch, with all resources pre-selected.
    """
    def __init__(self, batch_config: BatchConfig, corpus_map: Dict[str, str],
                 all_fonts: List[str], background_manager: BackgroundImageManager,
                 font_health_manager: FontHealthManager):
        """Initializes the GenerationOrchestrator.

        Args:
            batch_config: The main batch configuration.
            corpus_map: A map from corpus file names to their full paths.
            all_fonts: A list of all available font paths.
            background_manager: An initialized background manager.
            font_health_manager: An initialized font health manager.
        """
        self.batch_config = batch_config
        self.all_fonts = all_fonts

        self.batch_manager = BatchManager(batch_config)
        self.font_health_manager = font_health_manager
        self.background_manager = background_manager
        
        # Create a dictionary of CorpusManager instances, one for each unique corpus file.
        self._corpus_managers: Dict[str, CorpusManager] = {}
        for spec in batch_config.specifications:
            corpus_file_name = spec.corpus_file
            if corpus_file_name not in self._corpus_managers:
                full_path = corpus_map.get(corpus_file_name)
                if not full_path:
                    raise FileNotFoundError(f"Corpus file '{corpus_file_name}' not found in corpus_map.")
                self._corpus_managers[corpus_file_name] = CorpusManager.from_file(full_path)

    def create_task_list(self, min_text_len: int, max_text_len: int, unique_filenames: List[str], start_index: int = 0) -> List[GenerationTask]:
        """Creates a complete list of GenerationTask objects for the entire batch.

        Args:
            min_text_len: The minimum length of text segments to extract.
            max_text_len: The maximum length of text segments to extract.
            unique_filenames: A list of unique filenames (one per image) to use for output.
            start_index: The global index to start creating tasks from (for resuming).

        Returns:
            A list of GenerationTask objects.

        Raises:
            RuntimeError: If no healthy fonts or backgrounds are available.
        """
        tasks_to_run: List[GenerationTask] = []

        # Get the interleaved list of which batch spec to use for each image.
        spec_list = self.batch_manager.task_list()

        # Get the initial set of healthy resources.
        available_fonts = self.font_health_manager.get_available_fonts(self.all_fonts)
        if not available_fonts:
            raise RuntimeError("No available fonts to generate images with.")

        available_backgrounds = self.background_manager.get_available_backgrounds()
        if not available_backgrounds:
            # It's okay to not have backgrounds, we can use a transparent canvas.
            print("Warning: No available backgrounds found.")

        # If resuming, only process the specs for the remaining images.
        specs_to_process = spec_list[start_index:]

        # For each spec in the interleaved list, create a concrete task.
        for i, spec in enumerate(specs_to_process):
            global_index = start_index + i
            # Filter fonts if a filter is specified in the batch
            if spec.font_filter:
                available_fonts_for_spec = [f for f in available_fonts if fnmatch.fnmatch(f, spec.font_filter)]
            else:
                available_fonts_for_spec = list(available_fonts)

            if not available_fonts_for_spec:
                print(f"Warning: No fonts matched the filter '{spec.font_filter}' for batch '{spec.name}'. Skipping.")
                continue

            corpus_manager = self._corpus_managers[spec.corpus_file]
            text = corpus_manager.extract_text_segment(min_text_len, max_text_len)

            font_path = self.font_health_manager.select_font(available_fonts_for_spec)
            background_path = self.background_manager.select_background()

            task = GenerationTask(
                index=global_index,
                source_spec=spec,
                text=text,
                font_path=font_path,
                background_path=background_path,
                output_filename=unique_filenames[global_index]
            )
            tasks_to_run.append(task)
            
        return tasks_to_run