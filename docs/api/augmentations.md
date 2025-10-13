# API Reference: `augmentations.py`

This module contains functions for applying geometric distortions to the final image. These functions transform both the image and the corresponding bounding boxes to maintain accuracy.

### `apply_rotation(image, bboxes, angle)`
- **Description:** Rotates the image and bounding boxes.
- **Implementation:** Uses `cv2.getRotationMatrix2D` to create a transformation matrix and applies it to the image and bounding box corner coordinates.

### `apply_perspective_warp(image, bboxes, ...)`
- **Description:** Applies a perspective warp to simulate viewing the image from an angle.
- **Implementation:** Uses `cv2.getPerspectiveTransform` to create a matrix and `cv2.perspectiveTransform` to warp the bounding box coordinates.

### `apply_elastic_distortion(image, bboxes, ...)`
- **Description:** Creates a non-linear "wavy" distortion.
- **Implementation:** Creates a displacement field and uses `cv2.remap`. Bounding boxes are recalculated by isolating each character, applying the distortion, and finding the new pixel boundaries.

### `apply_grid_distortion(image, bboxes, ...)`
- **Description:** Creates a distortion based on a randomly perturbed grid.
- **Implementation:** Uses `cv2.remap` and the same robust bounding box recalculation method as elastic distortion.

### `apply_optical_distortion(image, bboxes, ...)`
- **Description:** Simulates lens distortion (barrel/pincushion).
- **Implementation:** Uses `cv2.undistort` and the same bounding box recalculation method as other non-linear distortions.
