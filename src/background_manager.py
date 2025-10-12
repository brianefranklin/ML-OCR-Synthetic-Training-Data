"""
Manages the health and reliability of background images by extending the generic ResourceManager.
"""

import glob

from typing import List, Set, Dict
from pathlib import Path
from src.resource_manager import ResourceManager



class BackgroundImageManager(ResourceManager):

    """

    Manages and selects background images from multiple sources, with validation

    and performance scoring.

    """

    def __init__(self, dir_weights: Dict[str, float] = None):

        super().__init__()

        self.background_paths = self._discover_backgrounds(dir_weights.keys() if dir_weights else [])




    def _discover_backgrounds(self, dirs: List[str]) -> List[str]:
        """Finds all image files in the specified directories."""
        paths = []
        # Use a set of lowercase extensions for efficient, case-insensitive checks
        allowed_extensions = {".png", ".jpg", ".jpeg"}

        for d in dirs:
            # Path() handles OS-specific separators (e.g., / or \) automatically
            dir_path = Path(d)
            
            # rglob('*') finds all items recursively. We then filter them.
            # This is more efficient than searching 3 separate times.
            for file_path in dir_path.rglob('*'):
                # Check if it's a file and has an allowed extension
                if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
                    paths.append(str(file_path))

        return paths



    def get_available_backgrounds(self) -> Set[str]:

        """Returns a set of available backgrounds."""

        return self.get_available_resources(self.background_paths)



    def select_background(self) -> str:

        """Selects a single background from the available list."""

        available_bgs = list(self.get_available_backgrounds())

        if not available_bgs:

            return None

        return self.select_resource(available_bgs)
