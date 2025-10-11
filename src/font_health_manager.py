"""
Manages the health and reliability of fonts by extending the generic ResourceManager.
"""

from typing import List, Set
from src.resource_manager import ResourceManager

class FontHealthManager(ResourceManager):
    """
    A specialized ResourceManager for managing font health.
    It provides a domain-specific API for font-related operations.
    """
    def __init__(self):
        super().__init__()

    def get_available_fonts(self, all_fonts: List[str]) -> Set[str]:
        """Returns a set of available fonts."""
        return self.get_available_resources(all_fonts)

    def select_font(self, font_list: List[str]) -> str:
        """Selects a single font from a list."""
        return self.select_resource(font_list)