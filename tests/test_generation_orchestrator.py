import pytest
from pathlib import Path
from typing import List, Dict

from src.batch_config import BatchConfig, BatchSpecification
from src.generation_orchestrator import GenerationOrchestrator, GenerationTask
from src.background_manager import BackgroundImageManager
from src.font_health_manager import FontHealthManager

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
    spec1 = BatchSpecification(name="spec_a", proportion=0.6, text_direction="ltr", corpus_file="corpus1.txt", font_filter="*Bold.ttf")
    spec2 = BatchSpecification(name="spec_b", proportion=0.4, text_direction="rtl", corpus_file="corpus2.txt", font_filter="*Regular.ttf")
    return BatchConfig(total_images=10, specifications=[spec1, spec2])

@pytest.fixture
def background_manager(tmp_path: Path) -> BackgroundImageManager:
    """Creates a dummy background manager with some files."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    (bg_dir / "bg_a.jpg").touch()
    (bg_dir / "bg_b.jpg").touch()
    return BackgroundImageManager(dir_weights={str(bg_dir): 1.0})

def test_create_generation_tasks_with_font_filter(batch_config: BatchConfig, corpus_map: Dict[str, str], background_manager: BackgroundImageManager):
    """Tests that the GenerationOrchestrator correctly filters fonts based on the spec."""
    all_fonts = ["/fonts/font_a_Bold.ttf", "/fonts/font_b_Regular.ttf"]
    font_health_manager = FontHealthManager()

    orchestrator = GenerationOrchestrator(
        batch_config=batch_config,
        corpus_map=corpus_map,
        all_fonts=all_fonts,
        background_manager=background_manager,
        font_health_manager=font_health_manager
    )

    # Generate unique filenames (using simple counter for tests)
    unique_filenames = [f"test_{i}" for i in range(batch_config.total_images)]

    tasks = orchestrator.create_task_list(min_text_len=5, max_text_len=10, unique_filenames=unique_filenames)

    # Check that the correct font was used for each spec
    for task in tasks:
        if task.source_spec.name == "spec_a":
            assert "Bold" in task.font_path
        elif task.source_spec.name == "spec_b":
            assert "Regular" in task.font_path

        # Check that output_filename is set
        assert hasattr(task, 'output_filename')
        assert isinstance(task.output_filename, str)
        assert len(task.output_filename) > 0