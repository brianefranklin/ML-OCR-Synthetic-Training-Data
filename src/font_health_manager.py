"""Manages the health and reliability of fonts.

This module provides a FontHealthManager that extends the generic ResourceManager
to provide a domain-specific API for font-related operations.
"""

from typing import List, Set, Optional
from src.resource_manager import ResourceManager

class FontHealthManager(ResourceManager):
    """A specialized ResourceManager for managing font health.

    This class provides a clear, domain-specific API for font-related operations
    while leveraging the underlying generic health tracking and selection logic
    of the ResourceManager.
    """
    def __init__(self):
        """Initializes the FontHealthManager."""
        super().__init__()

    def get_available_fonts(self, all_fonts: List[str]) -> Set[str]:
        """Returns a set of all healthy and available font paths.

        Args:
            all_fonts: A list of all font paths to check.

        Returns:
            A set of string paths for the available fonts.
        """
        return self.get_available_resources(all_fonts)

    def select_font(self, font_list: List[str]) -> Optional[str]:
        """Selects a single font from a list using weighted random selection.

        Args:
            font_list: The list of font paths to select from.

        Returns:
            A string path to a font, or None if the list is empty.
        """
        if not font_list:
            return None
        return self.select_resource(font_list)
