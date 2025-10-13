"""Manages efficient extraction of text segments from large corpus files.

This module provides a CorpusManager class that can handle terabyte-scale
text corpora with minimal memory footprint by using a streaming, round-robin
reading strategy.
"""

import random
from typing import List, Type, TypeVar, Dict, Optional

T = TypeVar('T')

class CorpusManager:
    """Handles reading from and extracting text segments from corpus files.

    This implementation uses a round-robin strategy, loading one file at a time
    into a cache to ensure all provided corpus files are used, while keeping
    memory usage low.

    Attributes:
        corpus_files (List[str]): The list of paths to the corpus files.
    """
    def __init__(self, corpus_files: List[str]):
        """Initializes the CorpusManager.

        Args:
            corpus_files (List[str]): A list of paths to the text files.
        
        Raises:
            ValueError: If the corpus_files list is empty.
        """
        if not corpus_files:
            raise ValueError("Corpus file list cannot be empty.")
        self.corpus_files: List[str] = corpus_files
        self._current_file_index: int = -1
        self._content_cache: Dict[str, str] = {}

    def _get_content_for_next_file(self) -> str:
        """Rotates to the next file in the list and returns its content.
        
        This method uses a cache to avoid re-reading files from disk on every call.

        Returns:
            The string content of the next file in the sequence.
        """
        # Move to the next file index, wrapping around if necessary.
        self._current_file_index = (self._current_file_index + 1) % len(self.corpus_files)
        file_path = self.corpus_files[self._current_file_index]
        
        # If the file content is not in our cache, read it from disk.
        if file_path not in self._content_cache:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._content_cache[file_path] = f.read()
        
        return self._content_cache[file_path]

    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        """Convenience method to create a CorpusManager from a single file path."""
        return cls([file_path])

    def extract_text_segment(self, min_length: int, max_length: int) -> Optional[str]:
        """Extracts a random text segment from the corpus.

        This method uses the next file in a round-robin sequence for each call.

        Args:
            min_length: The minimum desired length of the text segment.
            max_length: The maximum desired length of the text segment.

        Returns:
            A random string segment from the corpus, or None if a valid
            segment cannot be extracted.
        """
        content = self._get_content_for_next_file()

        if len(content) < min_length:
            return None

        # Ensure the chosen length is valid for the content.
        actual_max_len = min(max_length, len(content))
        if min_length > actual_max_len:
            return None
        
        segment_len = random.randint(min_length, actual_max_len)
        
        # Choose a random starting point for the segment.
        max_start_index = len(content) - segment_len
        if max_start_index < 0:
            return None
            
        start_index = random.randint(0, max_start_index)
        
        return content[start_index : start_index + segment_len]