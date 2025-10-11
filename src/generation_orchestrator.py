"""
Orchestrates the generation of synthetic OCR data.
"""

from dataclasses import dataclass
from typing import List, Dict

from src.batch_config import BatchConfig, BatchSpecification, BatchManager
from src.corpus_manager import CorpusManager
from src.font_health_manager import FontHealthManager
from src.background_manager import BackgroundImageManager

@dataclass
class GenerationTask:
    """
    Represents a single, complete set of parameters for generating one image.
    """
    source_spec: BatchSpecification
    text: str
    font_path: str
    background_path: str

class GenerationOrchestrator:
    """
    Orchestrates the creation of generation tasks by integrating the
    BatchManager and various resource managers.
    """
    def __init__(self, batch_config: BatchConfig, corpus_map: Dict[str, str], 
                 all_fonts: List[str], all_backgrounds: List[str]):
        """
        Initializes the GenerationOrchestrator.

        Args:
            batch_config (BatchConfig): The main batch configuration.
            corpus_map (Dict[str, str]): A map from corpus file names to paths.
            all_fonts (List[str]): A list of all available font paths.
            all_backgrounds (List[str]): A list of all available background paths.
        """
        self.batch_config = batch_config
        self.all_fonts = all_fonts
        self.all_backgrounds = all_backgrounds
        
        self.batch_manager = BatchManager(batch_config)
        self.font_health_manager = FontHealthManager()
        self.background_manager = BackgroundImageManager()
        
        self._corpus_managers: Dict[str, CorpusManager] = {}
        for spec in batch_config.specifications:
            corpus_file_name = spec.corpus_file
            if corpus_file_name not in self._corpus_managers:
                full_path = corpus_map.get(corpus_file_name)
                if not full_path:
                    raise FileNotFoundError(f"Corpus file '{corpus_file_name}' not found in corpus_map.")
                self._corpus_managers[corpus_file_name] = CorpusManager.from_file(full_path)

    def create_task_list(self, min_text_len: int, max_text_len: int) -> List[GenerationTask]:
        """
        Creates a complete list of GenerationTask objects for the entire batch.
        """
        full_task_list: List[GenerationTask] = []
        
        spec_list = self.batch_manager.task_list()
        
        available_fonts = self.font_health_manager.get_available_fonts(self.all_fonts)
        if not available_fonts:
            raise RuntimeError("No available fonts to generate images with.")

        available_backgrounds = self.background_manager.get_available_backgrounds(self.all_backgrounds)
        if not available_backgrounds:
            raise RuntimeError("No available backgrounds to generate images with.")

        for spec in spec_list:
            corpus_manager = self._corpus_managers[spec.corpus_file]
            text = corpus_manager.extract_text_segment(min_text_len, max_text_len)
            
            font_path = self.font_health_manager.select_font(list(available_fonts))
            background_path = self.background_manager.select_background(list(available_backgrounds))
            
            task = GenerationTask(
                source_spec=spec, 
                text=text, 
                font_path=font_path,
                background_path=background_path
            )
            full_task_list.append(task)
            
        return full_task_list
