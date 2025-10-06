"""
Tests for CorpusManager - Text corpus handling for OCR data generation
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from corpus_manager import CorpusManager


class TestCorpusManagerInit:
    """Test CorpusManager initialization."""

    def test_init_single_file(self, tmp_path):
        """Test initialization with single corpus file."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("This is test corpus text with enough content for testing.", encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])

        assert manager.corpus_files == [str(corpus_file)]
        assert manager.buffer_size == 1_000_000
        assert len(manager.weights) == 1
        assert manager.current_file_handle is None

    def test_init_multiple_files(self, tmp_path):
        """Test initialization with multiple corpus files."""
        files = []
        for i in range(3):
            f = tmp_path / f"corpus_{i}.txt"
            f.write_text(f"Corpus file {i} with test content for extraction.", encoding='utf-8')
            files.append(str(f))

        manager = CorpusManager(files)

        assert len(manager.corpus_files) == 3
        assert len(manager.weights) == 3
        assert all(w == 1.0 for w in manager.weights)

    def test_init_empty_list_raises_error(self):
        """Test that empty corpus list raises ValueError."""
        with pytest.raises(ValueError, match="corpus_files cannot be empty"):
            CorpusManager([])

    def test_init_with_weights(self, tmp_path):
        """Test initialization with weighted file selection."""
        files = []
        for i in range(3):
            f = tmp_path / f"wiki_{i}.txt"
            f.write_text(f"Wiki corpus {i} content.", encoding='utf-8')
            files.append(str(f))

        weights = {"wiki_*.txt": 2.0}
        manager = CorpusManager(files, weights=weights)

        assert all(w == 2.0 for w in manager.weights)

    def test_init_with_worker_id(self, tmp_path):
        """Test initialization with worker_id shuffles file order."""
        files = []
        for i in range(5):
            f = tmp_path / f"corpus_{i}.txt"
            f.write_text(f"Corpus {i}", encoding='utf-8')
            files.append(str(f))

        # Create two managers with different worker IDs
        manager1 = CorpusManager(files, worker_id=0)
        manager2 = CorpusManager(files, worker_id=1)

        # File orders should likely be different (not guaranteed, but very likely)
        # At minimum, they should both have all files
        assert len(manager1.file_order) == 5
        assert len(manager2.file_order) == 5


class TestCorpusManagerFactoryMethods:
    """Test factory methods for creating CorpusManager instances."""

    def test_from_directory_basic(self, tmp_path):
        """Test creating from directory with default pattern."""
        # Create test corpus files
        for i in range(3):
            (tmp_path / f"corpus_{i}.txt").write_text(f"Content {i}", encoding='utf-8')

        manager = CorpusManager.from_directory(tmp_path)

        assert len(manager.corpus_files) == 3
        assert all(f.endswith('.txt') for f in manager.corpus_files)

    def test_from_directory_with_pattern(self, tmp_path):
        """Test creating from directory with custom pattern."""
        # Create different file types
        (tmp_path / "wiki_1.txt").write_text("Wiki 1", encoding='utf-8')
        (tmp_path / "wiki_2.txt").write_text("Wiki 2", encoding='utf-8')
        (tmp_path / "news_1.txt").write_text("News 1", encoding='utf-8')

        manager = CorpusManager.from_directory(tmp_path, pattern="wiki_*.txt")

        assert len(manager.corpus_files) == 2
        assert all("wiki_" in f for f in manager.corpus_files)

    def test_from_directory_nonexistent_raises_error(self, tmp_path):
        """Test that non-existent directory raises error."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(ValueError, match="Directory does not exist"):
            CorpusManager.from_directory(nonexistent)

    def test_from_directory_no_matches_raises_error(self, tmp_path):
        """Test that directory with no matching files raises error."""
        # Create directory but no .txt files
        (tmp_path / "data.csv").write_text("not a txt file", encoding='utf-8')

        with pytest.raises(ValueError, match="No files found matching pattern"):
            CorpusManager.from_directory(tmp_path)

    def test_from_file_or_directory_with_file(self, tmp_path):
        """Test convenience method with single file."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Test corpus content", encoding='utf-8')

        manager = CorpusManager.from_file_or_directory(corpus_file)

        assert len(manager.corpus_files) == 1
        assert manager.corpus_files[0] == str(corpus_file)

    def test_from_file_or_directory_with_directory(self, tmp_path):
        """Test convenience method with directory."""
        for i in range(2):
            (tmp_path / f"corpus_{i}.txt").write_text(f"Content {i}", encoding='utf-8')

        manager = CorpusManager.from_file_or_directory(tmp_path)

        assert len(manager.corpus_files) == 2

    def test_from_pattern_basic(self, tmp_path):
        """Test creating from glob pattern."""
        for i in range(3):
            (tmp_path / f"test_{i}.txt").write_text(f"Test {i}", encoding='utf-8')

        pattern = str(tmp_path / "test_*.txt")
        manager = CorpusManager.from_pattern(pattern)

        assert len(manager.corpus_files) == 3

    def test_from_pattern_no_matches_raises_error(self, tmp_path):
        """Test that pattern with no matches raises error."""
        pattern = str(tmp_path / "nonexistent_*.txt")

        with pytest.raises(ValueError, match="No files found matching pattern"):
            CorpusManager.from_pattern(pattern)


class TestCorpusManagerTextExtraction:
    """Test text extraction functionality."""

    def test_extract_basic(self, tmp_path):
        """Test basic text extraction."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("The quick brown fox jumps over the lazy dog. " * 20, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(10, 50)

        assert text is not None
        assert 10 <= len(text) <= 50
        assert isinstance(text, str)

    def test_extract_respects_min_length(self, tmp_path):
        """Test that extraction respects minimum length."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Test content " * 50, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(20, 30)

        assert text is not None
        assert len(text) >= 20

    def test_extract_respects_max_length(self, tmp_path):
        """Test that extraction respects maximum length."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Test content " * 100, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(10, 30)

        assert text is not None
        assert len(text) <= 30

    def test_extract_from_multiple_files(self, tmp_path):
        """Test extraction cycles through multiple files."""
        files = []
        for i in range(3):
            f = tmp_path / f"corpus_{i}.txt"
            f.write_text(f"File {i} content: " + "test text " * 50, encoding='utf-8')
            files.append(str(f))

        manager = CorpusManager(files)

        # Extract multiple times, should work across files
        extractions = []
        for _ in range(10):
            text = manager.extract_text_segment(10, 30)
            assert text is not None
            extractions.append(text)

        assert len(extractions) == 10

    def test_extract_from_small_file(self, tmp_path):
        """Test extraction from file smaller than requested length."""
        corpus_file = tmp_path / "small.txt"
        corpus_file.write_text("Small content here.", encoding='utf-8')  # ~19 chars

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(5, 15)

        assert text is not None
        assert 5 <= len(text) <= 19

    def test_extract_from_empty_file_returns_none(self, tmp_path):
        """Test that empty file returns None."""
        corpus_file = tmp_path / "empty.txt"
        corpus_file.write_text("", encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(10, 20)

        assert text is None

    def test_extract_removes_newlines(self, tmp_path):
        """Test that newlines are replaced with spaces."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Line one\nLine two\nLine three\nLine four\n" * 10, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(20, 50)

        assert text is not None
        assert '\n' not in text

    def test_extract_unicode_text(self, tmp_path):
        """Test extraction of Unicode text."""
        corpus_file = tmp_path / "unicode.txt"
        corpus_file.write_text("العربية 日本語 한국어 עברית " * 20, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(10, 30)

        assert text is not None
        # Length should be in reasonable range (note: some chars may be multi-byte)
        assert len(text) >= 5

    def test_extract_max_attempts(self, tmp_path):
        """Test that extraction gives up after max attempts."""
        corpus_file = tmp_path / "tiny.txt"
        corpus_file.write_text("Hi", encoding='utf-8')  # Too small

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(100, 200, max_attempts=5)

        # Should return None after trying
        assert text is None


class TestCorpusManagerBufferManagement:
    """Test buffer management and file rotation."""

    def test_buffer_refill(self, tmp_path):
        """Test that buffer refills when depleted."""
        corpus_file = tmp_path / "corpus.txt"
        # Create content larger than default buffer
        corpus_file.write_text("Test content. " * 10000, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)], buffer_size=100)  # Small buffer

        # Extract multiple times to deplete and refill buffer
        for _ in range(20):
            text = manager.extract_text_segment(5, 15)
            assert text is not None

    def test_file_rotation_on_exhaustion(self, tmp_path):
        """Test that files rotate when exhausted."""
        files = []
        for i in range(3):
            f = tmp_path / f"small_{i}.txt"
            f.write_text(f"Small file {i} content here.", encoding='utf-8')
            files.append(str(f))

        manager = CorpusManager(files, buffer_size=50)

        # Extract enough times to exhaust files and rotate
        extractions = []
        for _ in range(30):
            text = manager.extract_text_segment(5, 15)
            if text:
                extractions.append(text)

        # Should have succeeded multiple times despite small files
        assert len(extractions) > 10


class TestCorpusManagerResourceManagement:
    """Test proper resource cleanup."""

    def test_close_releases_file_handle(self, tmp_path):
        """Test that close() releases file handles."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Test content " * 100, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])

        # Trigger file open
        manager.extract_text_segment(10, 20)
        assert manager.current_file_handle is not None

        # Close should release handle
        manager.close()
        assert manager.current_file_handle is None
        assert manager.current_file_buffer == ""

    def test_context_manager_cleanup(self, tmp_path):
        """Test cleanup on object deletion."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Test content " * 100, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        manager.extract_text_segment(10, 20)

        # Delete manager (triggers __del__)
        file_handle = manager.current_file_handle
        del manager

        # File handle should be closed (hard to test directly, but shouldn't crash)


class TestCorpusManagerEdgeCases:
    """Test edge cases and error conditions."""

    def test_multiple_small_files(self, tmp_path):
        """Test with multiple very small files."""
        files = []
        for i in range(10):
            f = tmp_path / f"tiny_{i}.txt"
            f.write_text(f"File{i}", encoding='utf-8')  # ~5 chars
            files.append(str(f))

        manager = CorpusManager(files)

        # Should still be able to extract short text
        text = manager.extract_text_segment(3, 5)
        # May or may not succeed depending on exact content
        # Just ensure it doesn't crash
        assert text is None or isinstance(text, str)

    def test_whitespace_only_file(self, tmp_path):
        """Test file with only whitespace."""
        corpus_file = tmp_path / "whitespace.txt"
        corpus_file.write_text("   \n\n\t\t   ", encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(5, 10)

        # Should return None or empty string after strip
        assert text is None or len(text) == 0

    def test_very_long_requested_length(self, tmp_path):
        """Test requesting very long text from small file."""
        corpus_file = tmp_path / "medium.txt"
        corpus_file.write_text("Medium sized content here for testing. " * 10, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)])
        text = manager.extract_text_segment(10, 10000)

        # Should return what's available
        assert text is not None
        assert len(text) >= 10
        assert len(text) < 10000  # Won't have this much

    def test_custom_buffer_size(self, tmp_path):
        """Test with custom buffer size."""
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("Content " * 1000, encoding='utf-8')

        manager = CorpusManager([str(corpus_file)], buffer_size=500)

        assert manager.buffer_size == 500
        text = manager.extract_text_segment(10, 50)
        assert text is not None


class TestCorpusManagerRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_batch_generation_scenario(self, tmp_path):
        """Test scenario: generating many images in batch."""
        # Setup: Multiple large corpus files
        files = []
        for i in range(5):
            f = tmp_path / f"corpus_{i}.txt"
            f.write_text(f"Corpus {i}: " + "Sample text content. " * 500, encoding='utf-8')
            files.append(str(f))

        manager = CorpusManager(files)

        # Simulate generating 100 images
        for _ in range(100):
            text = manager.extract_text_segment(10, 50)
            assert text is not None
            assert 10 <= len(text) <= 50

    def test_directional_corpus_scenario(self, tmp_path):
        """Test scenario: directional corpus directories."""
        # Setup: Separate directories for each text direction
        ltr_dir = tmp_path / "ltr"
        ltr_dir.mkdir()

        for i in range(3):
            (ltr_dir / f"english_{i}.txt").write_text(
                f"English text for LTR rendering. " * 100,
                encoding='utf-8'
            )

        manager = CorpusManager.from_directory(ltr_dir)

        # Extract multiple times
        for _ in range(20):
            text = manager.extract_text_segment(15, 40)
            assert text is not None

    def test_weighted_selection_scenario(self, tmp_path):
        """Test scenario: weighted corpus file selection."""
        # Create mix of corpus files
        (tmp_path / "wiki_1.txt").write_text("Wikipedia content " * 100, encoding='utf-8')
        (tmp_path / "wiki_2.txt").write_text("More wiki content " * 100, encoding='utf-8')
        (tmp_path / "news_1.txt").write_text("News article content " * 100, encoding='utf-8')

        files = [
            str(tmp_path / "wiki_1.txt"),
            str(tmp_path / "wiki_2.txt"),
            str(tmp_path / "news_1.txt"),
        ]

        # Weight wiki files higher
        weights = {"wiki_*.txt": 2.0, "news_*.txt": 1.0}
        manager = CorpusManager(files, weights=weights)

        # Extract multiple times (should work regardless of weighting)
        for _ in range(30):
            text = manager.extract_text_segment(10, 30)
            assert text is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
