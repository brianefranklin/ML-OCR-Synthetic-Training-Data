"""Manages the health and reliability of background images.

This module extends the generic ResourceManager to discover, manage, and select
background images from specified directories.
"""

from typing import List, Set, Dict, Optional
from pathlib import Path
from src.resource_manager import ResourceManager

class BackgroundImageManager(ResourceManager):
    """Manages and selects background images from multiple sources.

    This class is responsible for finding all valid background images in a given
    set of directories, and then using the ResourceManager's health tracking
    and weighted selection to provide a background for generation.

    Attributes:
        background_paths (List[str]): A list of all the valid image paths found.
    """

    def __init__(self, dir_weights: Optional[Dict[str, float]] = None):
        """Initializes the BackgroundImageManager.

        Args:
            dir_weights (Optional[Dict[str, float]]): A dictionary mapping directory
                paths to their selection weight. Defaults to None.
        """
        super().__init__()
        self.background_paths: List[str] = self._discover_backgrounds(
            list(dir_weights.keys()) if dir_weights else []
        )

    def _discover_backgrounds(self, dirs: List[str]) -> List[str]:
        """Finds all image files in the specified directories using pathlib.

        Args:
            dirs (List[str]): A list of directory paths to search.

        Returns:
            A list of string paths for all found image files.
        """
        paths: List[str] = []
        allowed_extensions = {".png", ".jpg", ".jpeg"}

        for d in dirs:
            dir_path = Path(d)
            if not dir_path.is_dir():
                print(f"Warning: Background directory not found: {d}")
                continue
            
            # Use rglob('*') to find all files recursively and then filter.
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                    paths.append(str(file_path))
        return paths

    def get_available_backgrounds(self) -> Set[str]:
        """Returns a set of all healthy and available background paths.

        Returns:
            A set of string paths for the available backgrounds.
        """
        return self.get_available_resources(self.background_paths)

    def select_background(self) -> Optional[str]:
        """Selects a single background from the available list using weighted random selection.

        Returns:
            A string path to a background image, or None if no backgrounds are available.
        """
        available_bgs = list(self.get_available_backgrounds())
        if not available_bgs:
            return None
        return self.select_resource(available_bgs)