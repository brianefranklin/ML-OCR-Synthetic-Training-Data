"""
Tests for the canvas_placement module.
"""

import pytest
from PIL import Image
from src.canvas_placement import generate_random_canvas_size, calculate_text_placement, place_on_canvas

def test_generate_random_canvas_size():
    """Tests that the generated canvas is larger than the text image."""
    text_w, text_h = 100, 50
    
    canvas_w, canvas_h = generate_random_canvas_size(text_w, text_h)
    
    assert canvas_w >= text_w
    assert canvas_h >= text_h

def test_calculate_text_placement():
    """Tests that the calculated placement is valid and within bounds."""
    canvas_w, canvas_h = 200, 100
    text_w, text_h = 80, 40
    
    x, y = calculate_text_placement(canvas_w, canvas_h, text_w, text_h, "uniform_random")
    
    assert 0 <= x <= (canvas_w - text_w)
    assert 0 <= y <= (canvas_h - text_h)

def test_place_on_canvas():
    """Tests that the text image is placed correctly and bboxes are adjusted."""
    text_image = Image.new("RGBA", (80, 40))
    original_bboxes = [
        {"char": "a", "x0": 5, "y0": 10, "x1": 15, "y1": 30}
    ]
    canvas_w, canvas_h = 200, 100
    placement_x, placement_y = 50, 30

    final_image, adjusted_bboxes = place_on_canvas(
        text_image, canvas_w, canvas_h, placement_x, placement_y, original_bboxes
    )

    assert final_image.size == (canvas_w, canvas_h)
    assert len(adjusted_bboxes) == len(original_bboxes)
    
    # Check if bounding boxes are correctly offset
    adj_bbox = adjusted_bboxes[0]
    orig_bbox = original_bboxes[0]
    assert adj_bbox["x0"] == orig_bbox["x0"] + placement_x
    assert adj_bbox["y0"] == orig_bbox["y0"] + placement_y
    assert adj_bbox["x1"] == orig_bbox["x1"] + placement_x
    assert adj_bbox["y1"] == orig_bbox["y1"] + placement_y
