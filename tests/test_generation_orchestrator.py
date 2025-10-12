"""
Tests for the GenerationOrchestrator class.
"""

import pytest
from pathlib import Path
from pathlib import Path
from typing import List, Dict

from src.batch_config import BatchConfig, BatchSpecification
from src.generation_orchestrator import GenerationOrchestrator, GenerationTask
from src.background_manager import BackgroundImageManager

@pytest.fixture
def corpus_map(tmp_path: Path) -> Dict[str, str]:
    """Creates dummy corpus files and returns a map of their names to paths."""
    corpus_dir = tmp_path / "corpora"
    corpus_dir.mkdir()
    
    content1 = "This is the first corpus file, used for spec_a."
    file1 = corpus_dir / "corpus1.txt"
    file1.write_text(content1, encoding="utf-8")
    
    content2 = "This is the second corpus file, intended for spec_b."
    file2 = corpus_dir / "corpus2.txt"
    file2.write_text(content2, encoding="utf-8")
    
    return {"corpus1.txt": str(file1), "corpus2.txt": str(file2)}


@pytest.fixture
def batch_config() -> BatchConfig:
    """A sample BatchConfig for testing."""
    spec1 = BatchSpecification(name="spec_a", proportion=0.6, text_direction="ltr", corpus_file="corpus1.txt")
    spec2 = BatchSpecification(name="spec_b", proportion=0.4, text_direction="rtl", corpus_file="corpus2.txt")
    return BatchConfig(total_images=10, specifications=[spec1, spec2])

@pytest.fixture
def background_manager(tmp_path: Path) -> BackgroundImageManager:
    """Creates a dummy background manager with some files."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    (bg_dir / "bg_a.jpg").touch()
    (bg_dir / "bg_b.jpg").touch()
    return BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

def test_create_generation_tasks(batch_config: BatchConfig, corpus_map: Dict[str, str], background_manager: BackgroundImageManager):
    """
    Tests that the GenerationOrchestrator can create a list of tasks,
    integrating all necessary managers.
    """
    all_fonts = ["/fonts/font_a.ttf", "/fonts/font_b.ttf"]

    orchestrator = GenerationOrchestrator(
        batch_config=batch_config, 
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager
    )
    
    tasks = orchestrator.create_task_list(min_text_len=5, max_text_len=10)
    
    # 1. Check total number of tasks
    assert len(tasks) == 10
    
    # 2. Check the type and content of each task
    for task in tasks:
        assert isinstance(task, GenerationTask)
        assert isinstance(task.source_spec, BatchSpecification)
        assert isinstance(task.text, str)
        assert 5 <= len(task.text) <= 10
        
        assert hasattr(task, 'font_path')
        assert isinstance(task.font_path, str)
        assert task.font_path in all_fonts

        assert hasattr(task, 'background_path')
        assert isinstance(task.background_path, str)
        assert task.background_path in background_manager.background_paths        
    # 3. Check that the correct corpus was used for each spec
    spec_a_texts = "".join([task.text for task in tasks if task.source_spec.name == "spec_a"])
    spec_b_texts = "".join([task.text for task in tasks if task.source_spec.name == "spec_b"])
    
    assert len([t for t in tasks if t.source_spec.name == "spec_a"]) == 6
    assert len([t for t in tasks if t.source_spec.name == "spec_b"]) == 4
    
    assert "spec_b" not in spec_a_texts
    assert "spec_a" not in spec_b_texts
    
    # 4. Check for interleaved order
    assert tasks[0].source_spec.name == "spec_a"
    assert tasks[1].source_spec.name == "spec_b"
    assert tasks[2].source_spec.name == "spec_a"
    assert tasks[3].source_spec.name == "spec_b"
