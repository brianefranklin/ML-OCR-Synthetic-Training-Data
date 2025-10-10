"""
Background Image Manager for OCR Data Generation

Manages background images with:
- Loading from multiple directories with glob patterns and weights
- Validation (corruption, dimensions)
- Performance scoring (persisted to disk like font performance)
- Weighted selection
"""

import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import fnmatch


class BackgroundImageManager:
    """Manages background images with validation and scoring."""

    def __init__(self,
                 background_dirs: Optional[List[str]] = None,
                 pattern: str = "*.{png,jpg,jpeg}",
                 weights: Optional[Dict[str, float]] = None,
                 score_file: Optional[str] = None,
                 enable_persistence: bool = False):
        """
        Initialize background image manager.

        Args:
            background_dirs: List of directory paths containing background images
            pattern: Glob pattern for image files (supports brace expansion)
            weights: Dictionary mapping directory paths to selection weights
            score_file: Path to JSON file for persisting scores (default: .background_scores.json)
            enable_persistence: If False, scores are session-only (not saved/loaded from disk).
                              Default False for batch mode where context changes between jobs.
        """
        self.background_dirs = background_dirs or []
        self.pattern = pattern
        self.weights = weights or {}
        self.score_file = score_file or ".background_scores.json"
        self.enable_persistence = enable_persistence

        # Background image data: path -> score
        self.backgrounds: Dict[str, float] = {}

        # Load existing scores only if persistence is enabled
        if self.enable_persistence:
            self._load_scores()
        else:
            logging.debug("Background scoring persistence disabled - using session-only scoring")

        # Discover background images
        self._discover_backgrounds()

        logging.info(f"BackgroundImageManager initialized with {len(self.backgrounds)} images")

    def _expand_pattern(self, pattern: str) -> List[str]:
        """
        Expand brace patterns in glob strings.

        Examples:
            "*.{png,jpg}" -> ["*.png", "*.jpg"]
            "image_*.png" -> ["image_*.png"]
        """
        # Simple brace expansion for {a,b,c} patterns
        if '{' in pattern and '}' in pattern:
            start = pattern.index('{')
            end = pattern.index('}')
            prefix = pattern[:start]
            suffix = pattern[end+1:]
            options = pattern[start+1:end].split(',')
            return [f"{prefix}{opt.strip()}{suffix}" for opt in options]
        return [pattern]

    def _discover_backgrounds(self):
        """Discover background images from configured directories."""
        patterns = self._expand_pattern(self.pattern)

        for dir_path in self.background_dirs:
            path = Path(dir_path)
            if not path.exists():
                logging.warning(f"Background directory does not exist: {dir_path}")
                continue

            if not path.is_dir():
                logging.warning(f"Background path is not a directory: {dir_path}")
                continue

            # Apply each pattern
            found_count = 0
            for pattern in patterns:
                for img_path in path.glob(pattern):
                    if img_path.is_file():
                        img_str = str(img_path.absolute())
                        # Initialize score if not already tracked
                        if img_str not in self.backgrounds:
                            self.backgrounds[img_str] = 1.0  # Default score
                        found_count += 1

            logging.info(f"Found {found_count} background images in {dir_path}")

    def _load_scores(self):
        """Load background image scores from disk."""
        score_path = Path(self.score_file)
        if not score_path.exists():
            return

        try:
            with open(score_path, 'r') as f:
                scores = json.load(f)
                self.backgrounds = {path: float(score) for path, score in scores.items()}
                logging.info(f"Loaded {len(self.backgrounds)} background scores from {self.score_file}")
        except Exception as e:
            logging.warning(f"Failed to load background scores from {self.score_file}: {e}")

    def _save_scores(self):
        """Persist background image scores to disk (only if persistence is enabled)."""
        if not self.enable_persistence:
            logging.debug("Background scoring persistence disabled, skipping save")
            return

        try:
            with open(self.score_file, 'w') as f:
                json.dump(self.backgrounds, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save background scores to {self.score_file}: {e}")

    def get_directory_weight(self, img_path: str) -> float:
        """
        Get the weight for an image based on its directory.

        Args:
            img_path: Absolute path to image file

        Returns:
            Weight value (default 1.0 if no weight specified)
        """
        path = Path(img_path)

        # Check if any configured directory matches this image's parent
        for dir_path, weight in self.weights.items():
            dir_abs = Path(dir_path).absolute()
            try:
                # Check if image is under this directory
                path.relative_to(dir_abs)
                return weight
            except ValueError:
                continue

        return 1.0

    def select_background(self) -> Optional[str]:
        """
        Select a background image using weighted random selection.

        Selection weight = directory_weight * performance_score

        Returns:
            Path to selected background image, or None if no backgrounds available
        """
        if not self.backgrounds:
            logging.warning("No background images available")
            return None

        # Calculate combined weights
        paths = list(self.backgrounds.keys())
        weights = [
            self.get_directory_weight(path) * self.backgrounds[path]
            for path in paths
        ]

        # Handle case where all weights are zero or negative
        if all(w <= 0 for w in weights):
            logging.warning("All background images have zero or negative weight")
            return None

        # Select using weighted random choice
        selected = random.choices(paths, weights=weights, k=1)[0]
        return selected

    def validate_background(self,
                          bg_path: str,
                          canvas_size: Tuple[int, int],
                          text_bbox: Tuple[int, int, int, int]) -> Tuple[bool, str, float]:
        """
        Validate background image against canvas and text dimensions.

        Args:
            bg_path: Path to background image
            canvas_size: (width, height) of canvas
            text_bbox: (x_min, y_min, x_max, y_max) bounding box of text

        Returns:
            Tuple of (is_valid, reason, score_penalty)
            - is_valid: True if background can be used
            - reason: String describing validation result
            - score_penalty: Score penalty to apply (0.0 = no penalty, higher = more penalty)
        """
        canvas_width, canvas_height = canvas_size
        bbox_width = text_bbox[2] - text_bbox[0]
        bbox_height = text_bbox[3] - text_bbox[1]

        try:
            # Try to open and get dimensions
            with Image.open(bg_path) as img:
                bg_width, bg_height = img.size

                # Check if image is valid
                if bg_width <= 0 or bg_height <= 0:
                    return False, "Invalid image dimensions", 1.0

                # Severe penalty: smaller than bounding box
                if bg_width < bbox_width or bg_height < bbox_height:
                    reason = f"Background ({bg_width}x{bg_height}) smaller than text bbox ({bbox_width}x{bbox_height})"
                    return False, reason, 1.0  # Severe penalty

                # Moderate penalty: smaller than canvas
                if bg_width < canvas_width or bg_height < canvas_height:
                    reason = f"Background ({bg_width}x{bg_height}) smaller than canvas ({canvas_width}x{canvas_height})"
                    return False, reason, 0.5  # Moderate penalty

                # Success: large enough
                return True, f"Valid background ({bg_width}x{bg_height})", 0.0

        except Exception as e:
            logging.error(f"Failed to validate background {bg_path}: {e}")
            return False, f"Corrupt or unreadable: {e}", 1.0

    def load_and_crop_background(self,
                                bg_path: str,
                                canvas_size: Tuple[int, int]) -> Optional[Image.Image]:
        """
        Load background image and crop random region to canvas size.

        Args:
            bg_path: Path to background image
            canvas_size: (width, height) of desired canvas

        Returns:
            PIL Image cropped to canvas_size, or None if failed
        """
        try:
            with Image.open(bg_path) as img:
                # Convert to RGB (in case of RGBA, grayscale, etc.)
                img_rgb = img.convert('RGB')

                bg_width, bg_height = img_rgb.size
                canvas_width, canvas_height = canvas_size

                # If background exactly matches canvas, return it
                if bg_width == canvas_width and bg_height == canvas_height:
                    return img_rgb.copy()

                # Calculate random crop position
                max_x = bg_width - canvas_width
                max_y = bg_height - canvas_height

                # Random position
                crop_x = random.randint(0, max(0, max_x))
                crop_y = random.randint(0, max(0, max_y))

                # Crop region
                crop_box = (
                    crop_x,
                    crop_y,
                    crop_x + canvas_width,
                    crop_y + canvas_height
                )

                cropped = img_rgb.crop(crop_box)
                return cropped

        except Exception as e:
            logging.error(f"Failed to load and crop background {bg_path}: {e}")
            return None

    def update_score(self, bg_path: str, penalty: float):
        """
        Update the performance score for a background image.

        Args:
            bg_path: Path to background image
            penalty: Penalty to apply (0.0 = no penalty, higher = more penalty)
        """
        if bg_path not in self.backgrounds:
            logging.warning(f"Attempting to score unknown background: {bg_path}")
            return

        # Apply penalty (reduce score)
        current_score = self.backgrounds[bg_path]
        new_score = max(0.01, current_score - penalty)  # Keep minimum score above 0

        self.backgrounds[bg_path] = new_score

        logging.debug(f"Updated score for {Path(bg_path).name}: {current_score:.3f} -> {new_score:.3f} (penalty: {penalty})")

        # Periodically save scores (every 10th update)
        if random.random() < 0.1:
            self._save_scores()

    def finalize(self):
        """Save scores to disk (only if persistence enabled). Call this at end of generation run."""
        if self.enable_persistence:
            self._save_scores()
            logging.info(f"Background scores saved to {self.score_file}")
        else:
            logging.debug("Background scoring persistence disabled, session scores discarded")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about background images.

        Returns:
            Dictionary with statistics
        """
        if not self.backgrounds:
            return {
                'total_backgrounds': 0,
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0
            }

        scores = list(self.backgrounds.values())
        return {
            'total_backgrounds': len(self.backgrounds),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores)
        }
