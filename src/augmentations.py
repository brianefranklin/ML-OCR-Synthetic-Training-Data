import random
import logging
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# --- Helper Functions ---

def pil_to_cv2(pil_image):
    """Converts a Pillow image to an OpenCV image."""
    return np.array(pil_image)

def cv2_to_pil(cv2_image):
    """Converts an OpenCV image to a Pillow image."""
    return Image.fromarray(cv2_image)

# --- Simple Augmentations (from previous version) ---

def add_noise(image):
    """Adds random 'salt and pepper' noise to a Pillow image."""
    logging.debug("Applying add_noise augmentation")
    # Preserve RGBA mode if present
    original_mode = image.mode
    img_np = pil_to_cv2(image.convert('L')) # Convert to grayscale
    h, w = img_np.shape
    noise = np.zeros((h, w), np.uint8)
    cv2.randu(noise, 0, 255)

    salt = noise > 245
    pepper = noise < 10

    img_np[salt] = 255
    img_np[pepper] = 0

    result = cv2_to_pil(img_np).convert(original_mode)
    return result


def rotate_image(image, bboxes):
    """Rotates a Pillow image by a random small angle, adjusts bboxes, and crops the image."""
    if not bboxes:
        return image, []
    logging.debug("Applying rotate_image augmentation")
    angle = random.uniform(-3, 3)
    w, h = image.size
    center_x, center_y = w / 2, h / 2

    # Rotate image with expand=True to prevent cutoff
    # Use transparent fillcolor for RGBA, white for RGB
    fillcolor = (255, 255, 255, 0) if image.mode == 'RGBA' else 'white'
    rotated_image = image.rotate(angle, expand=True, fillcolor=fillcolor)
    new_w, new_h = rotated_image.size

    # Adjust the rotation matrix to account for the new dimensions and center
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        points = np.float32([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]])
        ones = np.ones(shape=(len(points), 1))
        points_ones = np.hstack([points, ones])

        transformed_points = M.dot(points_ones.T).T

        x_min = min(transformed_points[:, 0])
        y_min = min(transformed_points[:, 1])
        x_max = max(transformed_points[:, 0])
        y_max = max(transformed_points[:, 1])
        new_bboxes.append([x_min, y_min, x_max, y_max])

    # Add padding to the image to ensure the bounding box is within the image
    overall_bbox = [min(b[0] for b in new_bboxes), min(b[1] for b in new_bboxes), max(b[2] for b in new_bboxes), max(b[3] for b in new_bboxes)]

    # Safeguard: Check if bboxes extend unreasonably far from image boundaries
    MAX_DIMENSION = 20000  # Maximum reasonable dimension (20K pixels per side)
    MAX_PIXELS = 178_000_000  # PIL's decompression bomb limit
    # Allow bbox to extend at most 3x the image dimension from boundaries
    max_offset_x = new_w * 3
    max_offset_y = new_h * 3

    # Check if any bbox extends too far from rotated image boundaries
    if (overall_bbox[2] > new_w + max_offset_x or overall_bbox[3] > new_h + max_offset_y or
        overall_bbox[0] < -max_offset_x or overall_bbox[1] < -max_offset_y):
        logging.warning(f"rotate_image: Bboxes extend too far from image boundaries (bbox: {overall_bbox}, image: {new_w}x{new_h}). Skipping rotation.")
        return image, bboxes

    # Check overall bbox dimensions
    bbox_width = overall_bbox[2] - overall_bbox[0]
    bbox_height = overall_bbox[3] - overall_bbox[1]
    if bbox_width > MAX_DIMENSION or bbox_height > MAX_DIMENSION or (bbox_width * bbox_height) > MAX_PIXELS:
        logging.warning(f"rotate_image: Overall bbox size ({bbox_width:.0f}x{bbox_height:.0f}) exceeds safe limits. Skipping rotation.")
        return image, bboxes

    padding_x = max(0, -overall_bbox[0])
    padding_y = max(0, -overall_bbox[1])

    # Safeguard: Detect unreasonable padding that would create massive images
    MAX_PADDING = 10000  # Maximum reasonable padding in pixels
    if padding_x > MAX_PADDING or padding_y > MAX_PADDING:
        logging.warning(f"rotate_image: Excessive padding detected (x={padding_x:.0f}, y={padding_y:.0f}). Skipping rotation to prevent decompression bomb. Bbox: {overall_bbox}")
        return image, bboxes  # Return original image without rotation

    # Additional safeguard: Validate that resulting image size is reasonable
    proposed_width = new_w + int(padding_x)
    proposed_height = new_h + int(padding_y)

    if proposed_width > MAX_DIMENSION or proposed_height > MAX_DIMENSION or (proposed_width * proposed_height) > MAX_PIXELS:
        logging.warning(f"rotate_image: Proposed image size ({proposed_width}x{proposed_height} = {proposed_width*proposed_height} pixels) exceeds safe limits. Skipping rotation.")
        return image, bboxes  # Return original image without rotation

    # Create padded image matching original mode
    pad_color = (255, 255, 255, 0) if image.mode == 'RGBA' else 'white'
    padded_image = Image.new(image.mode, (proposed_width, proposed_height), color=pad_color)
    padded_image.paste(rotated_image, (int(padding_x), int(padding_y)))

    # Adjust bounding boxes for the padded image
    padded_bboxes = []
    for bbox in new_bboxes:
        padded_bboxes.append([bbox[0] + padding_x, bbox[1] + padding_y, bbox[2] + padding_x, bbox[3] + padding_y])

    # Crop the padded image
    overall_padded_bbox = [min(b[0] for b in padded_bboxes), min(b[1] for b in padded_bboxes), max(b[2] for b in padded_bboxes), max(b[3] for b in padded_bboxes)]

    # Validate crop box before cropping
    crop_width = overall_padded_bbox[2] - overall_padded_bbox[0]
    crop_height = overall_padded_bbox[3] - overall_padded_bbox[1]
    if crop_width > MAX_DIMENSION or crop_height > MAX_DIMENSION or (crop_width * crop_height) > MAX_PIXELS:
        logging.warning(f"rotate_image: Crop size ({crop_width:.0f}x{crop_height:.0f}) exceeds safe limits. Skipping rotation.")
        return image, bboxes

    cropped_image = padded_image.crop(overall_padded_bbox)

    # Adjust bounding boxes to be relative to the cropped image
    final_bboxes = []
    for bbox in padded_bboxes:
        final_bboxes.append([bbox[0] - overall_padded_bbox[0], bbox[1] - overall_padded_bbox[1], bbox[2] - overall_padded_bbox[0], bbox[3] - overall_padded_bbox[1]])

    return cropped_image, final_bboxes

def blur_image(image):
    """Applies a Gaussian blur to a Pillow image."""
    logging.debug("Applying blur_image augmentation")

    # Check if image is valid
    if image.size[0] == 0 or image.size[1] == 0:
        logging.warning("blur_image received empty image, skipping")
        return image

    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))

# --- Advanced Augmentations ---

def perspective_transform(image, bboxes):
    """Applies a random perspective warp to a Pillow image and its bounding boxes."""
    logging.debug("Applying perspective_transform augmentation")
    img_cv = pil_to_cv2(image)
    h, w, _ = img_cv.shape
    
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    margin_x = w * 0.1
    margin_y = h * 0.2
    
    dst_points = np.float32([
        [random.uniform(0, margin_x), random.uniform(0, margin_y)],
        [w - random.uniform(0, margin_x), random.uniform(0, margin_y)],
        [random.uniform(0, margin_x), h - random.uniform(0, margin_y)],
        [w - random.uniform(0, margin_x), h - random.uniform(0, margin_y)]
    ])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img_cv, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # Transform bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        points = np.float32([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]]).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, matrix)
        x_min = min(transformed_points[:, 0, 0])
        y_min = min(transformed_points[:, 0, 1])
        x_max = max(transformed_points[:, 0, 0])
        y_max = max(transformed_points[:, 0, 1])
        new_bboxes.append([x_min, y_min, x_max, y_max])

    return cv2_to_pil(warped_img), new_bboxes

def elastic_distortion(image, bboxes):
    """Applies elastic distortion to a Pillow image and its bounding boxes."""
    logging.debug("Applying elastic_distortion augmentation")
    img_cv = pil_to_cv2(image)
    alpha = random.uniform(15, 25) # Distortion intensity
    sigma = random.uniform(4, 5)   # Distortion scale
    
    shape = img_cv.shape
    dx = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (5, 5), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape[:2]) * 2 - 1), (5, 5), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    distorted_img = cv2.remap(img_cv, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Distort bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        points = np.float32([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]]])
        distorted_points = []
        for px, py in points:
            # Clamp coordinates to be within image bounds for map lookup
            clamp_x = int(max(0, min(px, shape[1] - 1)))
            clamp_y = int(max(0, min(py, shape[0] - 1)))
            
            new_x = px + dx[clamp_y, clamp_x]
            new_y = py + dy[clamp_y, clamp_x]
            distorted_points.append([new_x, new_y])
        
        distorted_points = np.array(distorted_points)
        x_min = min(distorted_points[:, 0])
        y_min = min(distorted_points[:, 1])
        x_max = max(distorted_points[:, 0])
        y_max = max(distorted_points[:, 1])
        new_bboxes.append([x_min, y_min, x_max, y_max])

    return cv2_to_pil(distorted_img), new_bboxes

def adjust_brightness_contrast(image):
    """Randomly adjusts brightness and contrast of a Pillow image."""
    logging.debug("Applying adjust_brightness_contrast augmentation")
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))
    
    return image

def erode_dilate(image):
    """Applies erosion or dilation to a Pillow image."""
    logging.debug("Applying erode_dilate augmentation")

    # Check if image is valid
    if image.size[0] == 0 or image.size[1] == 0:
        logging.warning("erode_dilate received empty image, skipping")
        return image

    # Preserve original mode
    original_mode = image.mode
    img_cv = pil_to_cv2(image.convert('L'))

    # Double-check OpenCV image is valid
    if img_cv.size == 0:
        logging.warning("erode_dilate: OpenCV image is empty, skipping")
        return image

    kernel = np.ones((2, 2), np.uint8)

    if random.random() > 0.5:
        # Erode
        result_cv = cv2.erode(img_cv, kernel, iterations=1)
    else:
        # Dilate
        result_cv = cv2.dilate(img_cv, kernel, iterations=1)

    return cv2_to_pil(result_cv).convert(original_mode)

def add_background(image, background_images):
    """Adds a random background from a list of images."""
    logging.debug("Applying add_background augmentation")
    if not background_images:
        return image

    # Check if image is valid
    if image.size[0] == 0 or image.size[1] == 0:
        logging.warning("add_background received empty image, skipping")
        return image

    bg_path = random.choice(background_images)
    try:
        bg_image = Image.open(bg_path).convert('RGB')
        bg_image = bg_image.resize(image.size)

        # Handle RGBA images - use alpha channel as mask
        if image.mode == 'RGBA':
            # Composite RGBA text onto RGB background using alpha channel
            bg_image.paste(image, (0, 0), image)
            return bg_image
        else:
            # Create a mask from the text (dark pixels = text, light pixels = background)
            # Mask should be 255 where text is (to show original text) and 0 where background is
            mask = image.convert('L').point(lambda x: 255 if x < 200 else 0, '1')

            # Composite the text onto the background
            bg_image.paste(image, (0, 0), mask)
            return bg_image
    except Exception as e:
        logging.error(f"Could not apply background {bg_path}: {e}")
        return image


def add_shadow(image):
    """Adds a soft shadow effect to the text."""
    logging.debug("Applying add_shadow augmentation")

    # Check if image is valid
    if image.size[0] == 0 or image.size[1] == 0:
        logging.warning("add_shadow received empty image, skipping")
        return image

    # Convert to RGB for shadow processing, preserve alpha if RGBA
    if image.mode == 'RGBA':
        alpha_channel = image.split()[3]
        rgb_image = Image.merge('RGB', image.split()[:3])
    else:
        rgb_image = image
        alpha_channel = None

    img_cv = pil_to_cv2(rgb_image)
    h, w, _ = img_cv.shape

    # Create a shadow by shifting the text
    shadow_offset_x = random.randint(-3, 3)
    shadow_offset_y = random.randint(2, 4)
    
    M = np.float32([[1, 0, shadow_offset_x], [0, 1, shadow_offset_y]])
    shadow = cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # Make shadow gray and blur it
    shadow = cv2.cvtColor(shadow, cv2.COLOR_BGR2GRAY)
    shadow = cv2.GaussianBlur(shadow, (5, 5), 0)
    shadow = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)

    # Combine original image with shadow
    img_with_shadow = np.minimum(img_cv, shadow)

    result = cv2_to_pil(img_with_shadow)

    # Restore alpha channel if original was RGBA
    if alpha_channel is not None:
        result = Image.merge('RGBA', (*result.split(), alpha_channel))

    return result


def cutout(image):
    """Randomly erases a rectangular region in a Pillow image."""
    logging.debug("Applying cutout augmentation")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    x1 = random.randint(0, width)
    y1 = random.randint(0, height)
    x2 = x1 + random.randint(10, 40)
    y2 = y1 + random.randint(10, 40)
    
    draw.rectangle([x1, y1, x2, y2], fill='white')
    return image

# --- Main Augmentation Pipeline ---

def apply_augmentations(image, char_bboxes, background_images):
    """
    Applies a random pipeline of augmentations to an image and its character bounding boxes.
    """
    # Start with a clean image and original bboxes
    augmented_image = image
    augmented_bboxes = char_bboxes
    augmentations_applied = {}

    # Core text-affecting augmentations
    if random.random() < 0.3:
        augmented_image, augmented_bboxes = perspective_transform(augmented_image, augmented_bboxes)
        augmentations_applied['perspective_transform'] = True
    if random.random() < 0.2:
        augmented_image, augmented_bboxes = elastic_distortion(augmented_image, augmented_bboxes)
        augmentations_applied['elastic_distortion'] = True
    if random.random() < 0.4:
        augmented_image, augmented_bboxes = rotate_image(augmented_image, augmented_bboxes)
        augmentations_applied['rotate'] = True
    if random.random() < 0.3:
        augmented_image = erode_dilate(augmented_image)
        augmentations_applied['erode_dilate'] = True
    if random.random() < 0.3:
        augmented_image = add_shadow(augmented_image)
        augmentations_applied['shadow'] = True

    # Background and lighting
    if random.random() < 0.6:
        augmented_image = add_background(augmented_image, background_images)
        augmentations_applied['background'] = True
    
    augmented_image = adjust_brightness_contrast(augmented_image)
    augmentations_applied['brightness_contrast'] = True

    # Post-processing noise and blur
    if random.random() < 0.5:
        augmented_image = blur_image(augmented_image)
        augmentations_applied['blur'] = True
    if random.random() < 0.2:
        augmented_image = add_noise(augmented_image)
        augmentations_applied['noise'] = True
    if random.random() < 0.15:
        augmented_image = cutout(augmented_image)
        augmentations_applied['cutout'] = True

    return augmented_image, augmented_bboxes, augmentations_applied