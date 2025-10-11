"""
Manages the health and reliability of background images by extending the generic ResourceManager.
"""

from typing import List, Set
from src.resource_manager import ResourceManager

class BackgroundImageManager(ResourceManager):
    """
    A specialized ResourceManager for managing background image health.
    It provides a domain-specific API for background-related operations.
    """
    def __init__(self):
        super().__init__()

    def get_available_backgrounds(self, all_bgs: List[str]) -> Set[str]:
        """Returns a set of available backgrounds."""
        return self.get_available_resources(all_bgs)

    def select_background(self, bg_list: List[str]) -> str:
        """Selects a single background from a list."""
        return self.select_resource(bg_list)