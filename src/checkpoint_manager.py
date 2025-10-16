"""
Checkpoint manager for resume capability.

This module provides checkpoint management to track generation progress,
enabling resume functionality after interruptions. It stores progress data,
validates configuration consistency, and identifies completed images.
"""

import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Set, Optional


class CheckpointWarning(UserWarning):
    """Warning raised when checkpoint validation detects issues.

    This warning is raised when:
    - Configuration hash has changed between runs
    - Checkpoint data appears inconsistent

    Examples:
        >>> warnings.warn("Config changed", CheckpointWarning)
    """
    pass


class CheckpointManager:
    """Manages generation checkpoints for resume capability.

    This manager tracks generation progress by:
    - Saving checkpoint files with progress and config hash
    - Loading existing checkpoints and validating config consistency
    - Identifying which images have been completed
    - Warning when configuration changes between runs

    Args:
        output_dir: Directory where images and checkpoint are saved.
        config: Batch configuration dictionary.

    Examples:
        >>> config = {"total_images": 100, "specifications": [...]}
        >>> manager = CheckpointManager("./output", config)
        >>> manager.save_checkpoint(completed_images=50)
        >>> data = manager.load_checkpoint()
        >>> completed = manager.get_completed_indices()
    """

    CHECKPOINT_FILENAME = ".generation_checkpoint.json"

    def __init__(self, output_dir: str, config: Dict[str, Any]):
        """Initialize checkpoint manager.

        Args:
            output_dir: Path to output directory.
            config: Batch configuration dictionary.
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILENAME
        self._config_hash = self._compute_config_hash()
        self._completed_indices: Optional[Set[int]] = None

    def _compute_config_hash(self) -> str:
        """Compute deterministic hash of configuration.

        Returns:
            SHA256 hash of configuration as hex string.

        Examples:
            >>> manager = CheckpointManager("./output", {"total_images": 100})
            >>> hash1 = manager._compute_config_hash()
            >>> hash2 = manager._compute_config_hash()
            >>> assert hash1 == hash2  # Deterministic
        """
        # Convert config to JSON with sorted keys for determinism
        config_json = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_json.encode('utf-8')).hexdigest()

    def save_checkpoint(self, completed_images: int) -> None:
        """Save checkpoint with current progress.

        Args:
            completed_images: Number of images completed so far.

        Examples:
            >>> manager = CheckpointManager("./output", config)
            >>> manager.save_checkpoint(completed_images=50)
        """
        checkpoint_data = {
            "completed_images": completed_images,
            "total_images": self.config.get("total_images", 0),
            "config_hash": self._config_hash,
            "timestamp": datetime.now().isoformat()
        }

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint atomically by writing to temp file then renaming
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        temp_path.replace(self.checkpoint_path)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load existing checkpoint and validate config.

        Returns:
            Checkpoint data dictionary if exists, None otherwise.
            Contains: completed_images, total_images, config_hash, timestamp

        Warns:
            CheckpointWarning: If configuration hash has changed.

        Examples:
            >>> manager = CheckpointManager("./output", config)
            >>> data = manager.load_checkpoint()
            >>> if data:
            ...     print(f"Completed: {data['completed_images']}")
        """
        if not self.checkpoint_path.exists():
            return None

        with open(self.checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        # Check if config hash matches
        saved_hash = checkpoint_data.get("config_hash")
        if saved_hash and saved_hash != self._config_hash:
            warnings.warn(
                "Configuration has changed since last run. "
                "This may result in inconsistent output. "
                "Consider starting a new generation with a fresh output directory.",
                CheckpointWarning
            )

        return checkpoint_data

    def get_completed_indices(self) -> Set[int]:
        """Get set of image indices that have been completed.

        Returns:
            Set of integer indices for completed images.

        Examples:
            >>> manager = CheckpointManager("./output", config)
            >>> completed = manager.get_completed_indices()
            >>> if 42 in completed:
            ...     print("Image 42 already exists")
        """
        if self._completed_indices is not None:
            return self._completed_indices

        # Scan output directory for existing PNG files
        completed = set()

        if not self.output_dir.exists():
            self._completed_indices = completed
            return completed

        # Look for image_NNNNN.png files
        for image_file in self.output_dir.glob("image_*.png"):
            try:
                # Extract index from filename
                # Format: image_00042.png -> 42
                index_str = image_file.stem.split('_')[1]
                index = int(index_str)
                completed.add(index)
            except (IndexError, ValueError):
                # Skip files that don't match expected format
                continue

        self._completed_indices = completed
        return completed

    def should_skip_index(self, index: int) -> bool:
        """Check if an image index should be skipped (already completed).

        Args:
            index: Image index to check.

        Returns:
            True if image already exists, False otherwise.

        Examples:
            >>> manager = CheckpointManager("./output", config)
            >>> if manager.should_skip_index(42):
            ...     print("Skipping image 42, already exists")
        """
        completed = self.get_completed_indices()
        return index in completed

    def clear_cache(self) -> None:
        """Clear cached completed indices.

        Call this after generating new images to force re-scan.

        Examples:
            >>> manager = CheckpointManager("./output", config)
            >>> # ... generate some images ...
            >>> manager.clear_cache()
            >>> completed = manager.get_completed_indices()  # Re-scans
        """
        self._completed_indices = None
