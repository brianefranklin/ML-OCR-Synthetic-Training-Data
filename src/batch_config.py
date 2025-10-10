"""
Batch Configuration System for OCR Synthetic Data Generation

Supports YAML-based batch specifications with:
- Proportional image allocation
- Per-batch parameters (text direction, corpus, fonts, augmentations)
- Font weights and text length ranges
- Interleaved generation for balanced output
"""

import yaml
import numpy as np
import random
import logging
import fnmatch
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchSpecification:
    """Specification for a single batch of images."""

    name: str
    proportion: float
    text_direction: str = "left_to_right"
    corpus_file: Optional[str] = None  # Single file (backwards compatible)
    corpus_dir: Optional[str] = None   # Directory of corpus files
    corpus_pattern: Optional[str] = None  # Glob pattern for corpus files
    corpus_weights: Dict[str, float] = field(default_factory=dict)  # Weights for corpus file selection
    text_pattern: str = "*.txt"  # Pattern for files in corpus_dir
    font_filter: str = "*.{ttf,otf}"
    font_weights: Dict[str, float] = field(default_factory=dict)
    min_text_length: int = 5
    max_text_length: int = 25
    curve_type: str = "none"  # 'none', 'arc', 'sine', 'random'
    curve_intensity: float = 0.0  # 0.0-1.0
    overlap_intensity: float = 0.0  # 0.0-1.0, character overlap amount
    ink_bleed_intensity: float = 0.0  # 0.0-1.0, ink bleed effect strength
    effect_type: str = "none"  # 'none', 'raised', 'embossed', 'engraved'
    effect_depth: float = 0.5  # 0.0-1.0, 3D effect depth intensity
    light_azimuth: float = 135.0  # 0-360 degrees, light direction angle
    light_elevation: float = 45.0  # 0-90 degrees, light elevation angle
    text_color_mode: str = "uniform"  # 'uniform', 'per_glyph', 'gradient', 'random'
    color_palette: str = "realistic_dark"  # 'realistic_dark', 'vibrant', 'pastels', etc.
    custom_colors: Optional[List[Tuple[int, int, int]]] = None  # Optional list of custom RGB tuples
    background_color: Union[Tuple[int, int, int], str] = "auto"  # Background color RGB tuple or 'auto'
    background_dirs: Optional[List[str]] = None  # List of directories containing background images
    background_pattern: str = "*.{png,jpg,jpeg}"  # Glob pattern for background images
    background_weights: Dict[str, float] = field(default_factory=dict)  # Weights for background directory selection
    use_solid_background_fallback: bool = True  # Fallback to solid color if no valid backgrounds
    augmentation_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate batch specification."""
        if not 0 < self.proportion <= 1:
            raise ValueError(f"Batch '{self.name}': proportion must be between 0 and 1")

        valid_directions = ['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top', 'random']
        if self.text_direction not in valid_directions:
            raise ValueError(f"Batch '{self.name}': invalid text_direction '{self.text_direction}'")

    def matches_font(self, font_filename: str) -> bool:
        """Check if font matches this batch's filter."""
        # Support glob patterns
        patterns = self.font_filter.split(',')
        for pattern in patterns:
            pattern = pattern.strip()
            if fnmatch.fnmatch(font_filename.lower(), pattern.lower()):
                return True
        return False

    def get_font_weight(self, font_filename: str) -> float:
        """Get weight for a specific font (default 1.0)."""
        for pattern, weight in self.font_weights.items():
            if fnmatch.fnmatch(font_filename.lower(), pattern.lower()):
                return weight
        return 1.0


@dataclass
class BatchConfig:
    """Complete batch configuration."""

    total_images: int
    batches: List[BatchSpecification]
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.total_images <= 0:
            raise ValueError("total_images must be positive")

        if not self.batches:
            raise ValueError("At least one batch must be specified")

        # Normalize proportions if they don't sum to 1
        total_proportion = sum(b.proportion for b in self.batches)
        if abs(total_proportion - 1.0) > 0.01:
            logging.warning(f"Batch proportions sum to {total_proportion:.3f}, normalizing to 1.0")
            for batch in self.batches:
                batch.proportion /= total_proportion

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BatchConfig':
        """Load batch configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        total_images = data.get('total_images', 100)
        seed = data.get('seed')

        batches = []
        for batch_data in data.get('batches', []):
            batch = BatchSpecification(
                name=batch_data['name'],
                proportion=batch_data.get('proportion', 1.0),
                text_direction=batch_data.get('text_direction', 'left_to_right'),
                corpus_file=batch_data.get('corpus_file'),
                corpus_dir=batch_data.get('corpus_dir'),
                corpus_pattern=batch_data.get('corpus_pattern'),
                corpus_weights=batch_data.get('corpus_weights', {}),
                text_pattern=batch_data.get('text_pattern', '*.txt'),
                font_filter=batch_data.get('font_filter', '*.{ttf,otf}'),
                font_weights=batch_data.get('font_weights', {}),
                min_text_length=batch_data.get('min_text_length', 5),
                max_text_length=batch_data.get('max_text_length', 25),
                curve_type=batch_data.get('curve_type', 'none'),
                curve_intensity=batch_data.get('curve_intensity', 0.0),
                overlap_intensity=batch_data.get('overlap_intensity', 0.0),
                ink_bleed_intensity=batch_data.get('ink_bleed_intensity', 0.0),
                effect_type=batch_data.get('effect_type', 'none'),
                effect_depth=batch_data.get('effect_depth', 0.5),
                light_azimuth=batch_data.get('light_azimuth', 135.0),
                light_elevation=batch_data.get('light_elevation', 45.0),
                text_color_mode=batch_data.get('text_color_mode', 'uniform'),
                color_palette=batch_data.get('color_palette', 'realistic_dark'),
                custom_colors=batch_data.get('custom_colors'),
                background_color=batch_data.get('background_color', 'auto'),
                background_dirs=batch_data.get('background_dirs'),
                background_pattern=batch_data.get('background_pattern', '*.{png,jpg,jpeg}'),
                background_weights=batch_data.get('background_weights', {}),
                use_solid_background_fallback=batch_data.get('use_solid_background_fallback', True),
                augmentation_params=batch_data.get('augmentation_params', {})
            )
            batches.append(batch)

        return cls(total_images=total_images, batches=batches, seed=seed)


class BatchManager:
    """Manages batch generation with interleaved image creation."""

    def __init__(self, config: BatchConfig, all_fonts: List[str]):
        """
        Initialize batch manager.

        Args:
            config: Batch configuration
            all_fonts: List of all available font paths
        """
        self.config = config
        self.all_fonts = all_fonts
        self.batch_allocations = []

        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)

        self._allocate_images()

    def _allocate_images(self):
        """Allocate images to batches based on proportions."""
        total = self.config.total_images
        remaining = total

        # Calculate target counts
        for i, batch in enumerate(self.config.batches):
            if i == len(self.config.batches) - 1:
                # Last batch gets remaining images
                count = remaining
            else:
                count = round(batch.proportion * total)
                remaining -= count

            self.batch_allocations.append({
                'batch': batch,
                'target_count': count,
                'generated_count': 0,
                'fonts': self._get_batch_fonts(batch)
            })

        logging.info(f"Batch allocation for {total} images:")
        for alloc in self.batch_allocations:
            batch = alloc['batch']
            logging.info(f"  {batch.name}: {alloc['target_count']} images "
                        f"({batch.proportion*100:.1f}%, {len(alloc['fonts'])} fonts)")

    def _get_batch_fonts(self, batch: BatchSpecification) -> List[str]:
        """Get fonts that match batch filter."""
        matching_fonts = []
        for font_path in self.all_fonts:
            font_name = Path(font_path).name
            if batch.matches_font(font_name):
                matching_fonts.append(font_path)

        if not matching_fonts:
            logging.debug(f"Batch '{batch.name}': no fonts match filter '{batch.font_filter}', "
                         f"using all fonts")
            matching_fonts = self.all_fonts.copy()

        return matching_fonts

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get next image generation task (interleaved).

        Returns:
            Dict with generation parameters, or None if all batches complete
        """
        # Filter batches that still need images
        active_batches = [
            alloc for alloc in self.batch_allocations
            if alloc['generated_count'] < alloc['target_count']
        ]

        if not active_batches:
            return None

        # Round-robin selection with randomization
        # This ensures interleaving while respecting proportions
        alloc = random.choice(active_batches)
        batch = alloc['batch']

        # Select font with weighting
        fonts = alloc['fonts']
        if batch.font_weights:
            weights = [batch.get_font_weight(Path(f).name) for f in fonts]
            font_path = random.choices(fonts, weights=weights, k=1)[0]
        else:
            font_path = random.choice(fonts)

        # Determine text direction
        if batch.text_direction == 'random':
            direction = random.choice(['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'])
        else:
            direction = batch.text_direction

        # Determine curve parameters
        curve_type = batch.curve_type
        if curve_type == 'random':
            curve_type = random.choice(['none', 'arc', 'sine'])
        curve_intensity = batch.curve_intensity

        # Create task
        task = {
            'batch_name': batch.name,
            'font_path': font_path,
            'text_direction': direction,
            'corpus_file': batch.corpus_file,
            'corpus_dir': batch.corpus_dir,
            'corpus_pattern': batch.corpus_pattern,
            'corpus_weights': batch.corpus_weights,
            'text_pattern': batch.text_pattern,
            'min_text_length': batch.min_text_length,
            'max_text_length': batch.max_text_length,
            'curve_type': curve_type,
            'curve_intensity': curve_intensity,
            'overlap_intensity': batch.overlap_intensity,
            'ink_bleed_intensity': batch.ink_bleed_intensity,
            'effect_type': batch.effect_type,
            'effect_depth': batch.effect_depth,
            'light_azimuth': batch.light_azimuth,
            'light_elevation': batch.light_elevation,
            'text_color_mode': batch.text_color_mode,
            'color_palette': batch.color_palette,
            'custom_colors': batch.custom_colors,
            'background_color': batch.background_color,
            'background_dirs': batch.background_dirs,
            'background_pattern': batch.background_pattern,
            'background_weights': batch.background_weights,
            'use_solid_background_fallback': batch.use_solid_background_fallback,
            'augmentation_params': batch.augmentation_params,
            'progress': f"{alloc['generated_count']+1}/{alloc['target_count']}",
            'batch_index': self.batch_allocations.index(alloc)  # Track which batch this is
        }

        # Don't increment generated_count here - only increment on successful generation
        # This allows retry logic to work properly

        return task

    def mark_task_success(self, task: Dict[str, Any]) -> None:
        """
        Mark a task as successfully completed.

        Args:
            task: The task that was completed successfully
        """
        batch_index = task.get('batch_index')
        if batch_index is not None and 0 <= batch_index < len(self.batch_allocations):
            self.batch_allocations[batch_index]['generated_count'] += 1

    def get_progress_summary(self) -> str:
        """Get progress summary for all batches."""
        lines = ["Batch generation progress:"]
        for alloc in self.batch_allocations:
            batch = alloc['batch']
            count = alloc['generated_count']
            target = alloc['target_count']
            pct = (count / target * 100) if target > 0 else 0
            lines.append(f"  {batch.name}: {count}/{target} ({pct:.1f}%)")
        return "\n".join(lines)