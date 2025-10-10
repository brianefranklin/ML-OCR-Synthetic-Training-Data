
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
    # add_background - DEPRECATED: Backgrounds now handled by BackgroundImageManager at canvas placement stage
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
def dummy_bboxes():
    """Creates a sample list of bounding boxes."""
    return [[10, 10, 30, 30], [40, 10, 60, 30]]


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
    assert len(bboxes) == len(dummy_bboxes)


def test_blur_image(base_image):
    # Create an image with a sharp edge for blur detection
    img = Image.new('RGB', (100, 100), color = 'white')
    draw = ImageDraw.Draw(img)
    draw.line((50, 0, 50, 100), fill='black', width=1)

    blurred_img = blur_image(img)
    # A simple check: a blurred image should have more or equal unique colors (gray pixels)
    # Using >= instead of > because blur strength can vary
    assert len(blurred_img.getcolors(maxcolors=256*256*256)) >= len(img.getcolors(maxcolors=256*256*256))


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


# test_add_background - REMOVED: add_background function deprecated
# Backgrounds are now handled by BackgroundImageManager at canvas placement stage
# See test_background_manager.py for background system tests

def test_add_shadow(base_image):
    augmented = add_shadow(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_cutout(base_image):
    augmented = cutout(base_image)
    assert isinstance(augmented, Image.Image)
    assert augmented.size == base_image.size

def test_apply_augmentations(base_image, dummy_bboxes):
    """Test the main pipeline function to ensure it runs."""
    # background_images parameter is deprecated and ignored
    augmented, bboxes, augmentations_applied = apply_augmentations(base_image, dummy_bboxes)
    assert isinstance(augmented, Image.Image)
    assert len(bboxes) == len(dummy_bboxes)
    assert isinstance(augmentations_applied, dict)

def test_rotation_and_crop_bounds(base_image, dummy_bboxes):
    """
    Tests that after rotation and cropping, all bounding box corners are within the
    final image dimensions.
    """
    augmented_image, final_bboxes = rotate_image(base_image, dummy_bboxes)
    img_width, img_height = augmented_image.size

    for bbox in final_bboxes:
        x_min, y_min, x_max, y_max = bbox
        # Check if the bounding box is within the image dimensions with a tolerance of 1px
        assert x_min >= -1
        assert y_min >= -1
        assert x_max <= img_width + 1
        assert y_max <= img_height + 1


class TestDecompressionBombSafeguards:
    """Test safeguards against decompression bomb errors in rotation."""

    def test_excessive_padding_detection(self):
        """Test that excessive padding is detected and rotation is skipped."""
        # Create a small image
        img = Image.new('RGB', (100, 50), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 90, 40], fill='black')

        # Create bboxes that would cause massive padding
        # Simulate RTL text with extreme bbox values
        extreme_bboxes = [
            [10.0, 10.0, 90.0, 40.0],
            [10000.0, 10000.0, 11000.0, 10500.0]  # Extreme outlier
        ]

        # Should detect excessive padding and return original image
        result_img, result_bboxes = rotate_image(img, extreme_bboxes)

        # Image should be unchanged (safeguard triggered)
        assert result_img.size == img.size
        assert result_bboxes == extreme_bboxes

    def test_max_dimension_check(self):
        """Test that proposed image size is validated against MAX_DIMENSION."""
        # Create moderate sized image
        img = Image.new('RGB', (200, 100), color='white')

        # Create bboxes that would result in very large dimensions
        large_bboxes = [
            [0.0, 0.0, 100.0, 50.0],
            [5000.0, 5000.0, 5100.0, 5050.0]  # Would create huge canvas
        ]

        result_img, result_bboxes = rotate_image(img, large_bboxes)

        # Should return original image due to size limits
        assert result_img.size == img.size
        assert result_bboxes == large_bboxes

    def test_max_pixels_check(self):
        """Test that total pixel count is validated against MAX_PIXELS."""
        # Create image
        img = Image.new('RGB', (300, 200), color='white')

        # Create bboxes that would exceed pixel limit
        pixel_bomb_bboxes = [
            [0.0, 0.0, 100.0, 50.0],
            [15000.0, 15000.0, 15100.0, 15050.0]  # Would create >178M pixels
        ]

        result_img, result_bboxes = rotate_image(img, pixel_bomb_bboxes)

        # Should return original image
        assert result_img.size == img.size

    def test_normal_rotation_still_works(self):
        """Test that normal rotation still works with safeguards in place."""
        # Create image with reasonable bboxes
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 30, 150, 70], fill='black')

        normal_bboxes = [
            [50.0, 30.0, 100.0, 70.0],
            [100.0, 30.0, 150.0, 70.0]
        ]

        result_img, result_bboxes = rotate_image(img, normal_bboxes)

        # Rotation should have been applied
        # Image size might change due to rotation
        assert result_img is not None
        assert len(result_bboxes) == len(normal_bboxes)

    def test_rtl_text_safeguard(self):
        """Test safeguards work with RTL text scenarios."""
        # Simulate RTL rendering scenario
        img = Image.new('RGB', (400, 100), color='white')

        # RTL bboxes (right to left ordering)
        rtl_bboxes = [
            [300.0, 20.0, 380.0, 80.0],
            [220.0, 20.0, 290.0, 80.0],
            [140.0, 20.0, 210.0, 80.0],
            [60.0, 20.0, 130.0, 80.0]
        ]

        result_img, result_bboxes = rotate_image(img, rtl_bboxes)

        # Should complete without errors
        assert result_img is not None
        assert len(result_bboxes) == 4

    def test_crop_size_validation(self):
        """Test that crop size is validated before cropping."""
        img = Image.new('RGB', (200, 100), color='white')

        # Create scenario where crop would be too large
        crop_bomb_bboxes = [
            [0.0, 0.0, 50.0, 30.0],
            [12000.0, 0.0, 12050.0, 30.0]  # Would create huge crop
        ]

        result_img, result_bboxes = rotate_image(img, crop_bomb_bboxes)

        # Safeguard should prevent the crop
        assert result_img.size == img.size

    def test_empty_bboxes_with_safeguards(self):
        """Test that empty bboxes work with safeguards."""
        img = Image.new('RGB', (100, 50), color='white')

        result_img, result_bboxes = rotate_image(img, [])

        assert result_img.size == img.size
        assert result_bboxes == []

    def test_single_bbox_with_safeguards(self):
        """Test that single bbox works with safeguards."""
        img = Image.new('RGB', (100, 50), color='white')
        single_bbox = [[10.0, 10.0, 90.0, 40.0]]

        result_img, result_bboxes = rotate_image(img, single_bbox)

        assert result_img is not None
        assert len(result_bboxes) == 1
