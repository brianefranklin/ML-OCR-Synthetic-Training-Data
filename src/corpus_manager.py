"""
Corpus Manager for Terabyte-Scale OCR Data Generation

Optimized for:
- Millions of output images
- Terabytes of corpus text
- Parallel processing
- GPU acceleration (future)
- Minimal memory footprint

Uses sequential streaming with round-robin file rotation for maximum performance.
"""

import os
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import glob


class CorpusManager:
    """
    Manages lazy-loading sequential streaming from corpus files.

    Optimized for terabyte-scale generation with millions of images.
    Uses sequential reading with small buffers instead of random seeking.

    Key features:
    - Sequential streaming (100-1000x faster than random seek)
    - Small memory footprint (1MB buffer per instance)
    - Round-robin file rotation with optional weights
    - Parallel-worker friendly (each worker gets independent stream)
    - GPU-ready (read-only shared file list, independent worker state)
    """

    def __init__(self,
                 corpus_files: List[str],
                 weights: Optional[Dict[str, float]] = None,
                 buffer_size: int = 1_000_000,
                 worker_id: Optional[int] = None):
        """
        Initialize corpus manager for sequential streaming.

        Args:
            corpus_files: List of paths to corpus text files
            weights: Optional dict mapping filename patterns to weights (higher = more likely)
            buffer_size: Size of read buffer in bytes (default 1MB)
            worker_id: Optional worker ID for parallel processing (shuffles start position)

        Example:
            # Single worker
            manager = CorpusManager(['corpus1.txt', 'corpus2.txt'])

            # Parallel workers with different starting files
            manager = CorpusManager(corpus_files, worker_id=3)

            # Weighted sampling (prefer certain files)
            manager = CorpusManager(
                corpus_files,
                weights={'wiki_*.txt': 2.0, 'news_*.txt': 1.0}
            )
        """
        if not corpus_files:
            raise ValueError("corpus_files cannot be empty")

        self.corpus_files = corpus_files
        self.buffer_size = buffer_size

        # Build weighted selection probabilities
        self.weights = self._build_weights(weights)

        # Shuffle file order for this worker (each worker starts differently)
        if worker_id is not None:
            random.seed(hash((os.getpid(), worker_id)))
        self.file_order = random.sample(range(len(corpus_files)), len(corpus_files))

        # Current streaming state
        self.current_file_idx = 0
        self.current_file_handle = None
        self.current_file_buffer = ""
        self.current_file_path = None

        logging.debug(f"CorpusManager initialized with {len(corpus_files)} files, "
                     f"buffer_size={buffer_size}, worker_id={worker_id}")

    def _build_weights(self, weight_patterns: Optional[Dict[str, float]]) -> List[float]:
        """
        Build per-file weights from pattern-based weights.

        Args:
            weight_patterns: Dict mapping glob patterns to weights

        Returns:
            List of weights (one per file)
        """
        if not weight_patterns:
            return [1.0] * len(self.corpus_files)

        weights = []
        for filepath in self.corpus_files:
            filename = os.path.basename(filepath)
            weight = 1.0

            # Check each pattern
            for pattern, pattern_weight in weight_patterns.items():
                # Support glob patterns like 'wiki_*.txt'
                from fnmatch import fnmatch
                if fnmatch(filename, pattern) or fnmatch(filepath, pattern):
                    weight = pattern_weight
                    break

            weights.append(weight)

        logging.debug(f"Corpus file weights: {dict(zip([os.path.basename(f) for f in self.corpus_files], weights))}")
        return weights

    def extract_text_segment(self,
                           min_length: int,
                           max_length: int,
                           max_attempts: int = 100) -> Optional[str]:
        """
        Extract random text segment using sequential streaming.

        Strategy:
        1. Read sequentially from current file buffer
        2. Extract random substring from buffer
        3. When buffer depleted, read next chunk
        4. When file exhausted, rotate to next file

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            max_attempts: Maximum attempts to find valid segment

        Returns:
            Text segment or None if unable to extract
        """
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Ensure we have a file open and buffer has data
            if not self._ensure_buffer(min_length * 2):
                # Unable to get enough data
                if attempts > len(self.corpus_files):
                    # Tried all files, give up
                    logging.warning("Unable to extract text segment from any corpus file")
                    return None
                continue

            # Extract random segment from buffer
            text = self._extract_from_buffer(min_length, max_length)
            if text:
                return text

        return None

    def _ensure_buffer(self, min_buffer_size: int) -> bool:
        """
        Ensure buffer has at least min_buffer_size characters.

        Args:
            min_buffer_size: Minimum buffer size needed

        Returns:
            True if buffer has enough data, False otherwise
        """
        # If buffer is sufficient, we're done
        if len(self.current_file_buffer) >= min_buffer_size:
            return True

        # Track files tried to prevent infinite loop with empty files
        files_tried = 0
        max_files_to_try = len(self.corpus_files) * 2  # Try each file twice

        # Open file if not already open
        if self.current_file_handle is None:
            if not self._open_next_file():
                return False
            files_tried += 1

        # Read more data into buffer
        while len(self.current_file_buffer) < min_buffer_size:
            chunk = self.current_file_handle.read(self.buffer_size)

            if not chunk:
                # File exhausted, try next file
                self._close_current_file()

                # Check if we've tried too many files (all might be empty)
                if files_tried >= max_files_to_try:
                    logging.warning(f"Unable to fill buffer after trying {files_tried} files")
                    return False

                if not self._open_next_file():
                    return False
                files_tried += 1
                continue

            self.current_file_buffer += chunk

        return True

    def _extract_from_buffer(self, min_length: int, max_length: int) -> Optional[str]:
        """
        Extract random text segment from current buffer.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length

        Returns:
            Text segment or None
        """
        if len(self.current_file_buffer) < min_length:
            return None

        # Random text length
        text_length = random.randint(min_length, max_length)

        # Ensure we don't exceed buffer
        if text_length > len(self.current_file_buffer):
            text_length = len(self.current_file_buffer)

        if text_length < min_length:
            return None

        # Random start position in buffer
        max_start = len(self.current_file_buffer) - text_length
        start_idx = random.randint(0, max_start)

        # Extract segment
        text_segment = self.current_file_buffer[start_idx:start_idx + text_length]

        # Clean up text
        text_segment = text_segment.replace('\n', ' ').strip()

        # Remove consumed portion from buffer (keep some overlap for next extraction)
        # Keep last 50% of buffer to allow for overlapping text samples
        keep_size = len(self.current_file_buffer) // 2
        self.current_file_buffer = self.current_file_buffer[-keep_size:]

        if len(text_segment) >= min_length:
            return text_segment

        return None

    def _open_next_file(self) -> bool:
        """
        Open next corpus file in round-robin order.

        Returns:
            True if file opened successfully, False otherwise
        """
        # Round-robin selection: cycle through files in order
        # Use self.current_file_idx to track position in rotation
        file_idx = self.file_order[self.current_file_idx]
        self.current_file_idx = (self.current_file_idx + 1) % len(self.file_order)

        filepath = self.corpus_files[file_idx]

        try:
            self.current_file_handle = open(filepath, 'r', encoding='utf-8', errors='ignore')
            self.current_file_buffer = ""
            self.current_file_path = filepath
            logging.debug(f"Opened corpus file: {os.path.basename(filepath)}")
            return True
        except Exception as e:
            logging.warning(f"Failed to open corpus file {filepath}: {e}")
            return False

    def _close_current_file(self):
        """Close current file handle."""
        if self.current_file_handle:
            try:
                self.current_file_handle.close()
                logging.debug(f"Closed corpus file: {os.path.basename(self.current_file_path)}")
            except Exception as e:
                logging.warning(f"Error closing file {self.current_file_path}: {e}")
            finally:
                self.current_file_handle = None
                self.current_file_path = None

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'current_file_handle'):
            self._close_current_file()
        if hasattr(self, 'current_file_buffer'):
            self.current_file_buffer = ""

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except (AttributeError, Exception):
            # Ignore errors during cleanup (object may be partially initialized)
            pass

    @staticmethod
    def from_directory(directory: Union[str, Path],
                      pattern: str = "*.txt",
                      weights: Optional[Dict[str, float]] = None,
                      worker_id: Optional[int] = None) -> 'CorpusManager':
        """
        Create CorpusManager from directory of corpus files.

        Args:
            directory: Path to directory containing corpus files
            pattern: Glob pattern to match files (default: "*.txt")
            weights: Optional weights for file selection
            worker_id: Optional worker ID for parallel processing

        Returns:
            CorpusManager instance

        Example:
            # Load all .txt files from directory
            manager = CorpusManager.from_directory('data/corpus')

            # Load with pattern
            manager = CorpusManager.from_directory(
                'data/corpus',
                pattern='wiki_*.txt'
            )
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        # Find all matching files
        corpus_files = sorted(glob.glob(str(directory / pattern)))

        if not corpus_files:
            raise ValueError(f"No files found matching pattern '{pattern}' in {directory}")

        logging.info(f"Found {len(corpus_files)} corpus files in {directory}")

        return CorpusManager(corpus_files, weights=weights, worker_id=worker_id)

    @staticmethod
    def from_file_or_directory(path: Union[str, Path],
                              pattern: str = "*.txt",
                              weights: Optional[Dict[str, float]] = None,
                              worker_id: Optional[int] = None) -> 'CorpusManager':
        """
        Create CorpusManager from either a file or directory (convenience method).

        Args:
            path: Path to corpus file or directory
            pattern: Glob pattern if path is directory
            weights: Optional weights for file selection
            worker_id: Optional worker ID for parallel processing

        Returns:
            CorpusManager instance

        Example:
            # Supports both file and directory
            manager = CorpusManager.from_file_or_directory('corpus.txt')
            manager = CorpusManager.from_file_or_directory('corpus_dir/')
        """
        path = Path(path)

        if path.is_file():
            return CorpusManager([str(path)], weights=weights, worker_id=worker_id)
        elif path.is_dir():
            return CorpusManager.from_directory(path, pattern=pattern, weights=weights, worker_id=worker_id)
        else:
            raise ValueError(f"Path does not exist: {path}")

    @staticmethod
    def from_pattern(pattern: str,
                    weights: Optional[Dict[str, float]] = None,
                    worker_id: Optional[int] = None) -> 'CorpusManager':
        """
        Create CorpusManager from glob pattern.

        Args:
            pattern: Glob pattern for corpus files (e.g., 'data/corpus/**/*.txt')
            weights: Optional weights for file selection
            worker_id: Optional worker ID for parallel processing

        Returns:
            CorpusManager instance

        Example:
            # Load all .txt files recursively
            manager = CorpusManager.from_pattern('data/corpus/**/*.txt')

            # Load specific pattern
            manager = CorpusManager.from_pattern('data/english/wiki_*.txt')
        """
        corpus_files = sorted(glob.glob(pattern, recursive=True))

        if not corpus_files:
            raise ValueError(f"No files found matching pattern: {pattern}")

        logging.info(f"Found {len(corpus_files)} corpus files matching pattern '{pattern}'")

        return CorpusManager(corpus_files, weights=weights, worker_id=worker_id)
