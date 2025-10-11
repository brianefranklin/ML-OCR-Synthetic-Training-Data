"""
Manages efficient extraction of text segments from large corpus files.
"""
import random
from typing import List, Type, TypeVar, Dict

T = TypeVar('T')

class CorpusManager:
    """
    Handles reading from and extracting text segments from corpus files.
    This implementation uses a round-robin strategy, loading one file per call
    to extract_text_segment to ensure all files are used.
    """
    def __init__(self, corpus_files: List[str]):
        if not corpus_files:
            raise ValueError("Corpus file list cannot be empty.")
        self.corpus_files = corpus_files
        self._current_file_index = -1
        self._content_cache: Dict[str, str] = {}

    def __del__(self):
        pass # No file handles are kept open

    def _get_content_for_next_file(self) -> str:
        """Rotates to the next file and returns its content, using a cache."""
        self._current_file_index = (self._current_file_index + 1) % len(self.corpus_files)
        file_path = self.corpus_files[self._current_file_index]
        
        if file_path not in self._content_cache:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._content_cache[file_path] = f.read()
        
        return self._content_cache[file_path]

    @classmethod
    def from_file(cls: Type[T], file_path: str) -> T:
        """Convenience method to create a CorpusManager from a single file path."""
        return cls([file_path])

    def extract_text_segment(self, min_length: int, max_length: int) -> str:
        """
        Extracts a random text segment from the corpus, using the next file
        in a round-robin sequence for each call.
        """
        content = self._get_content_for_next_file()

        if len(content) < min_length:
            return None

        actual_max_len = min(max_length, len(content))
        if min_length > actual_max_len:
            return None
        
        segment_len = random.randint(min_length, actual_max_len)
        
        max_start_index = len(content) - segment_len
        if max_start_index < 0:
            return None
            
        start_index = random.randint(0, max_start_index)
        
        return content[start_index : start_index + segment_len]
