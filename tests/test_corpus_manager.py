"""
Tests for the CorpusManager class.
"""

import pytest
from pathlib import Path
from typing import List
import random
from src.corpus_manager import CorpusManager

@pytest.fixture
def corpus_file(tmp_path: Path) -> str:
    """Creates a dummy corpus file for testing."""
    corpus_content = "This is a sample corpus for testing the CorpusManager. It contains several sentences and words, which is long enough for various tests."
    file_path = tmp_path / "test_corpus.txt"
    file_path.write_text(corpus_content, encoding="utf-8")
    return str(file_path)

@pytest.fixture
def multi_corpus_files(tmp_path: Path) -> List[str]:
    """Creates two dummy corpus files with distinct content."""
    content1 = "This is the first corpus file, containing the unique keyword APPLE."
    file1 = tmp_path / "corpus1.txt"
    file1.write_text(content1, encoding="utf-8")
    
    content2 = "This is the second corpus file, with the special term BANANA."
    file2 = tmp_path / "corpus2.txt"
    file2.write_text(content2, encoding="utf-8")
    
    return [str(file1), str(file2)]

def test_extract_text_segment(corpus_file: str):
    """
    Tests that the CorpusManager can successfully extract a text segment
    of a specified length.
    """
    manager = CorpusManager.from_file(corpus_file)
    
    min_len = 10
    max_len = 20
    
    segment = manager.extract_text_segment(min_len, max_len)
    
    assert segment is not None
    assert min_len <= len(segment) <= max_len
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        original_text = f.read()
    assert segment in original_text

def test_multiple_files_are_used(multi_corpus_files: List[str]):
    """
    Tests that the CorpusManager reads from multiple files in a round-robin fashion.
    """
    manager = CorpusManager(multi_corpus_files)
    random.seed(0)

    # The first call to extract_text_segment should use the first file.
    # We extract a large segment to guarantee it contains the unique keyword.
    segment1 = manager.extract_text_segment(min_length=60, max_length=70)
    assert segment1 is not None
    assert "APPLE" in segment1
    assert "BANANA" not in segment1

    # The second call should use the second file.
    segment2 = manager.extract_text_segment(min_length=60, max_length=70)
    assert segment2 is not None
    assert "BANANA" in segment2
    assert "APPLE" not in segment2