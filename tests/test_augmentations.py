
import pytest
from PIL import Image, ImageDraw
import sys
import os

# Add src directory to path to import augmentations
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from augmentations import (
    add_noise,
    rotate_image,
    blur_image,
    perspective_transform,
    elastic_distortion,
    adjust_brightness_contrast,
    erode_dilate,
    add_background,
    add_shadow,
    cutout,
    apply_augmentations,
    pil_to_cv2,
    cv2_to_pil
)

@pytest.fixture
def base_image():
    """Creates a simple Pillow image with text for testing."""
    image = Image.new('RGB', (300, 100), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "Hello", fill="black")
    return image

@pytest.fixture
def black_image():
    """Creates a simple, black Pillow image for testing."""
    return Image.new('RGB', (100, 100), color='black')

@pytest.fixture
def background_image_list(tmp_path):
    """Creates a dummy background image for testing."""
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_file = bg_dir / "bg.png"
    Image.new('RGB', (300, 100), color='blue').save(bg_file)
    return [str(bg_file)]

@pytest.fixture
def dummy_bboxes():
    """Creates a sample list of bounding boxes."""
    return [[10, 10, 30, 30], [40, 10, 60, 30]]

@pytest.fixture
def empty_background_list():
    """Provides an empty list for backgrounds."""
    return []


def test_pil_to_cv2(base_image):
    """Test the Pillow to OpenCV conversion."""
    cv2_img = pil_to_cv2(base_image)
    assert isinstance(cv2_img, np.ndarray)
    assert cv2_img.shape == (100, 300, 3)


def test_cv2_to_pil(base_image):
    """Test the OpenCV to Pillow conversion."""
    cv2_img = pil_to_cv2(base_image)
    pil_img = cv2_to_pil(cv2_img)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == base_image.size


def test_add_noise(black_image):
    augmented = add_noise(black_image)
    assert isinstance(augmented, Image.Image)
    # Check if noise is added (i.e., the image is not all black)
    assert not all(p == (0, 0, 0) for p in augmented.getdata())


def test_rotate_image_no_bboxes(base_image):
    """Test rotate_image with an empty list of bounding boxes."""
    augmented, bboxes = rotate_image(base_image, [])
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert bboxes == []

def test_rotate_image(base_image, dummy_bboxes):
    augmented, bboxes = rotate_image(base_image, dummy_bboxes)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert len(bboxes) == len(dummy_bboxes)

def test_blur_image(base_image):
    # Create an image with a sharp edge for blur detection
    img = Image.new('RGB', (100, 100), color = 'white')
    draw = ImageDraw.Draw(img)
    draw.line((50, 0, 50, 100), fill='black', width=1)
    
    blurred_img = blur_image(img)
    # A simple check: a blurred image should have more unique colors (gray pixels)
    assert len(blurred_img.getcolors(maxcolors=256*256*256)) > len(img.getcolors(maxcolors=256*256*256))


def test_perspective_transform_no_bboxes(base_image):
    """Test perspective_transform with an empty list of bounding boxes."""
    augmented, bboxes = perspective_transform(base_image, [])
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert bboxes == []

def test_perspective_transform(base_image, dummy_bboxes):
    augmented, bboxes = perspective_transform(base_image, dummy_bboxes)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert len(bboxes) == len(dummy_bboxes)

def test_elastic_distortion_no_bboxes(base_image):
    """Test elastic_distortion with an empty list of bounding boxes."""
    augmented, bboxes = elastic_distortion(base_image, [])
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert bboxes == []

def test_elastic_distortion(base_image, dummy_bboxes):
    augmented, bboxes = elastic_distortion(base_image, dummy_bboxes)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    assert len(bboxes) == len(dummy_bboxes)

def test_adjust_brightness_contrast(base_image):
    augmented = adjust_brightness_contrast(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_erode_dilate(base_image):
    augmented = erode_dilate(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_add_background(base_image, background_image_list):
    augmented = add_background(base_image, background_image_list)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size
    # Check if the blue background color is present
    colors = [item[1] for item in augmented.getcolors(maxcolors=10000)]
    assert (0, 0, 255) in colors 

def test_add_shadow(base_image):
    augmented = add_shadow(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_cutout(base_image):
    augmented = cutout(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_apply_augmentations(base_image, dummy_bboxes, empty_background_list):
    """Test the main pipeline function to ensure it runs."""
    augmented, bboxes = apply_augmentations(base_image, dummy_bboxes, empty_background_list)
    assert isinstance(augmented, Image.Image)
    assert len(bboxes) == len(dummy_bboxes)
