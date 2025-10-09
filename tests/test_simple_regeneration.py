import pytest
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator import OCRDataGenerator


def test_simple_regeneration():
    """A simple test to check if regeneration is deterministic."""
    font_path = str(Path(__file__).parent.parent / "data.nosync" / "fonts" / "ABeeZee-Regular.ttf")
    generator = OCRDataGenerator(font_files=[font_path], background_images=[])

    # First generation
    image1, metadata1, _, _ = generator.generate_image(
        text="Hello, world!",
        font_path=font_path,
        font_size=32,
        direction="left_to_right",
        seed=42,
        canvas_size=(500, 100)
    )

    # Second generation with the same parameters
    image2, metadata2, _, _ = generator.generate_image(
        text="Hello, world!",
        font_path=font_path,
        font_size=32,
        direction="left_to_right",
        seed=42,
        canvas_size=(500, 100)
    )

    assert np.array_equal(np.array(image1), np.array(image2)), "Regenerated image is not pixel-perfect identical"
