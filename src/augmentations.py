"""
This module contains functions for applying geometric and image-level augmentations.
"""

from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import random

def apply_rotation(
    image: Image.Image, 
    bboxes: List[Dict[str, Any]], 
    angle: float
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Rotates an image and its bounding boxes.

    Args:
        image: The source PIL Image.
        bboxes: A list of bounding box dictionaries.
        angle: The rotation angle in degrees.

    Returns:
        A tuple containing the rotated image and the transformed bounding boxes.
    """
    # Rotate the image, expanding the canvas to fit
    rotated_image = image.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Get the transformation matrix
    w, h = image.size
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust the matrix to account for the new canvas size
    new_w, new_h = rotated_image.size
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Transform bounding boxes
    transformed_bboxes = []
    for bbox in bboxes:
        # Get the four corners of the bounding box
        corners = np.array([
            [bbox['x0'], bbox['y0']],
            [bbox['x1'], bbox['y0']],
            [bbox['x1'], bbox['y1']],
            [bbox['x0'], bbox['y1']]
        ], dtype=np.float32)

        # Add a homogeneous coordinate
        corners_homogeneous = np.hstack([corners, np.ones((4, 1))])

        # Apply the transformation
        transformed_corners = (M @ corners_homogeneous.T).T

        # Find the new bounding box
        min_x = np.min(transformed_corners[:, 0])
        max_x = np.max(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_y = np.max(transformed_corners[:, 1])

        new_bbox = bbox.copy()
        new_bbox['x0'] = int(min_x)
        new_bbox['y0'] = int(min_y)
        new_bbox['x1'] = int(max_x)
        new_bbox['y1'] = int(max_y)
        transformed_bboxes.append(new_bbox)

    return rotated_image, transformed_bboxes

def apply_perspective_warp(
    image: Image.Image, 
    bboxes: List[Dict[str, Any]], 
    magnitude: float,
    dst_points: np.ndarray = None
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Applies a perspective warp to an image and its bounding boxes.

    Args:
        image: The source PIL Image.
        bboxes: A list of bounding box dictionaries.
        magnitude: The intensity of the warp (0.0 to 1.0), used if dst_points is None.
        dst_points: Optional. A numpy array of 4 destination points to use for the warp.

    Returns:
        A tuple containing the warped image and transformed bounding boxes.
    """
    w, h = image.size
    
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if dst_points is None:
        max_offset = int(min(w, h) * magnitude / 2)
        dst_points = np.float32([
            [random.randint(0, max_offset), random.randint(0, max_offset)],
            [w - random.randint(1, max_offset), random.randint(0, max_offset)],
            [w - random.randint(1, max_offset), h - random.randint(1, max_offset)],
            [random.randint(0, max_offset), h - random.randint(1, max_offset)]
        ])

    M = cv2.getPerspectiveTransform(src_pts, dst_points)

    warped_image_np = cv2.warpPerspective(np.array(image), M, (w, h))
    warped_image = Image.fromarray(warped_image_np)

    # Transform bounding boxes
    transformed_bboxes = []
    for bbox in bboxes:
        corners = np.float32([
            [[bbox['x0'], bbox['y0']]],
            [[bbox['x1'], bbox['y0']]],
            [[bbox['x1'], bbox['y1']]],
            [[bbox['x0'], bbox['y1']]]
        ])

        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        min_x = np.min(transformed_corners[:, :, 0])
        max_x = np.max(transformed_corners[:, :, 0])
        min_y = np.min(transformed_corners[:, :, 1])
        max_y = np.max(transformed_corners[:, :, 1])

        new_bbox = bbox.copy()
        new_bbox['x0'] = int(min_x)
        new_bbox['y0'] = int(min_y)
        new_bbox['x1'] = int(max_x)
        new_bbox['y1'] = int(max_y)
        transformed_bboxes.append(new_bbox)

    return warped_image, transformed_bboxes

def apply_elastic_distortion(
    image: Image.Image, 
    bboxes: List[Dict[str, Any]], 
    alpha: float, 
    sigma: float
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Applies elastic distortion to an image and its bounding boxes.

    Args:
        image: The source PIL Image.
        bboxes: A list of bounding box dictionaries.
        alpha: The scaling factor for the distortion.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        A tuple containing the distorted image and transformed bounding boxes.
    """
    # Convert PIL image to numpy array
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # Generate random displacement fields
    dx = np.random.rand(h, w) * 2 - 1
    dy = np.random.rand(h, w) * 2 - 1

    # Smooth the fields with a Gaussian filter
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

    # Create the mapping
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    # Apply the distortion to the full image
    distorted_img_np = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    distorted_image = Image.fromarray(distorted_img_np)

    # Update bounding boxes
    transformed_bboxes = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']
        
        # Crop the character from the original image
        char_img_np = img_np[y0:y1, x0:x1]

        # Apply the same distortion to the character snippet
        char_map_x = map_x[y0:y1, x0:x1]
        char_map_y = map_y[y0:y1, x0:x1]
        distorted_char_np = cv2.remap(char_img_np, char_map_x - x0, char_map_y - y0, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # Find the new bounding box by searching for non-transparent pixels
        alpha_channel = distorted_char_np[:, :, 3]
        coords = np.argwhere(alpha_channel > 0)
        
        if coords.size > 0:
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)

            new_bbox = bbox.copy()
            new_bbox['x0'] = x0 + min_x
            new_bbox['y0'] = y0 + min_y
            new_bbox['x1'] = x0 + max_x
            new_bbox['y1'] = y0 + max_y
            transformed_bboxes.append(new_bbox)
        else:
            # If the character disappears, append the original bbox
            transformed_bboxes.append(bbox)

    return distorted_image, transformed_bboxes
