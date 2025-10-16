"""
Batch configuration validation module.

This module provides validation for batch configurations before generation starts,
ensuring all required resources exist and configuration values are valid.
It follows a fail-fast approach, raising an error on the first validation failure.
"""

from pathlib import Path
from typing import Dict, Any, List


class ValidationError(Exception):
    """Exception raised when batch configuration validation fails.

    This exception is raised when the batch configuration is invalid,
    missing required resources, or contains invalid parameter values.

    Args:
        message: Clear description of the validation error.

    Examples:
        >>> raise ValidationError("Corpus file 'data.txt' not found")
    """
    pass


class BatchValidator:
    """Validates batch configurations before generation starts.

    This validator performs comprehensive checks on batch configurations,
    including resource existence, parameter ranges, and required fields.
    It follows a fail-fast approach, raising ValidationError on the first issue.

    Args:
        config: The batch configuration dictionary (from YAML).
        corpus_dir: Path to the corpus text files directory.
        font_dir: Path to the font files directory.
        background_dir: Path to the background images directory.

    Examples:
        >>> config = {"total_images": 100, "specifications": [...]}
        >>> validator = BatchValidator(config, "./corpus", "./fonts", "./bg")
        >>> validator.validate()  # Raises ValidationError if invalid
    """

    def __init__(
        self,
        config: Dict[str, Any],
        corpus_dir: str,
        font_dir: str,
        background_dir: str
    ):
        """Initialize the validator with configuration and directories.

        Args:
            config: Batch configuration dictionary.
            corpus_dir: Path to corpus text files.
            font_dir: Path to font files.
            background_dir: Path to background images.
        """
        self.config = config
        self.corpus_dir = Path(corpus_dir)
        self.font_dir = Path(font_dir)
        self.background_dir = Path(background_dir)

    def validate(self) -> None:
        """Validate the batch configuration.

        Performs all validation checks in order:
        1. Required fields present
        2. Valid parameter values
        3. Required directories exist
        4. Required files exist

        Raises:
            ValidationError: If any validation check fails.

        Examples:
            >>> validator = BatchValidator(config, "./corpus", "./fonts", "./bg")
            >>> validator.validate()  # Raises ValidationError if invalid
        """
        # Check required top-level fields
        self._validate_required_fields()

        # Check total_images value
        self._validate_total_images()

        # Check specifications list
        self._validate_specifications_exist()

        # Check directories exist
        self._validate_directories_exist()

        # Check fonts are available
        self._validate_fonts_exist()

        # Check each specification
        for i, spec in enumerate(self.config["specifications"]):
            self._validate_specification(spec, index=i)

    def _validate_required_fields(self) -> None:
        """Validate that required top-level fields are present.

        Raises:
            ValidationError: If required field is missing.
        """
        required_fields = ["total_images", "specifications"]

        for field in required_fields:
            if field not in self.config:
                raise ValidationError(
                    f"Required field '{field}' missing from batch configuration"
                )

    def _validate_total_images(self) -> None:
        """Validate that total_images is a positive integer.

        Raises:
            ValidationError: If total_images is invalid.
        """
        total_images = self.config.get("total_images", 0)

        if not isinstance(total_images, int):
            raise ValidationError(
                f"Field 'total_images' must be an integer, got {type(total_images).__name__}"
            )

        if total_images <= 0:
            raise ValidationError(
                f"Field 'total_images' must be positive, got {total_images}"
            )

    def _validate_specifications_exist(self) -> None:
        """Validate that specifications list is not empty.

        Raises:
            ValidationError: If specifications list is empty.
        """
        specs = self.config.get("specifications", [])

        if not isinstance(specs, list):
            raise ValidationError(
                f"Field 'specifications' must be a list, got {type(specs).__name__}"
            )

        if len(specs) == 0:
            raise ValidationError(
                "Field 'specifications' cannot be empty - at least one specification is required"
            )

    def _validate_directories_exist(self) -> None:
        """Validate that required directories exist.

        Raises:
            ValidationError: If any required directory is missing.
        """
        if not self.font_dir.exists():
            raise ValidationError(
                f"Font directory does not exist: {self.font_dir}"
            )

        if not self.font_dir.is_dir():
            raise ValidationError(
                f"Font path is not a directory: {self.font_dir}"
            )

        if not self.background_dir.exists():
            raise ValidationError(
                f"Background directory does not exist: {self.background_dir}"
            )

        if not self.background_dir.is_dir():
            raise ValidationError(
                f"Background path is not a directory: {self.background_dir}"
            )

        if not self.corpus_dir.exists():
            raise ValidationError(
                f"Corpus directory does not exist: {self.corpus_dir}"
            )

        if not self.corpus_dir.is_dir():
            raise ValidationError(
                f"Corpus path is not a directory: {self.corpus_dir}"
            )

    def _validate_fonts_exist(self) -> None:
        """Validate that font directory contains font files.

        Raises:
            ValidationError: If no font files are found.
        """
        font_files = list(self.font_dir.rglob("*.ttf"))

        if len(font_files) == 0:
            raise ValidationError(
                f"No font files (.ttf) found in font directory: {self.font_dir}"
            )

    def _validate_specification(self, spec: Dict[str, Any], index: int) -> None:
        """Validate a single specification.

        Args:
            spec: Specification dictionary to validate.
            index: Index of specification in list (for error messages).

        Raises:
            ValidationError: If specification is invalid.
        """
        spec_name = spec.get("name", f"specification[{index}]")

        # Validate corpus files
        self._validate_corpus_files(spec, spec_name)

        # Validate min/max pairs
        self._validate_min_max_pairs(spec, spec_name)

        # Validate config section if present
        if "config" in spec:
            config = spec["config"]

            # Validate canvas settings
            if "canvas" in config:
                self._validate_canvas_config(config["canvas"], spec_name)

            # Validate font settings
            if "font" in config:
                self._validate_font_config(config["font"], spec_name)

            # Validate direction settings
            if "direction" in config:
                self._validate_direction_config(config["direction"], spec_name)

    def _validate_min_max_pairs(self, spec: Dict[str, Any], spec_name: str) -> None:
        """Validate that min values are not greater than max values."""
        min_max_pairs = [
            ("min_text_length", "max_text_length"),
            ("glyph_overlap_intensity_min", "glyph_overlap_intensity_max"),
            ("ink_bleed_radius_min", "ink_bleed_radius_max"),
            ("rotation_angle_min", "rotation_angle_max"),
            ("perspective_warp_magnitude_min", "perspective_warp_magnitude_max"),
            ("elastic_distortion_alpha_min", "elastic_distortion_alpha_max"),
            ("elastic_distortion_sigma_min", "elastic_distortion_sigma_max"),
            ("grid_distortion_steps_min", "grid_distortion_steps_max"),
            ("grid_distortion_limit_min", "grid_distortion_limit_max"),
            ("optical_distortion_limit_min", "optical_distortion_limit_max"),
            ("noise_amount_min", "noise_amount_max"),
            ("blur_radius_min", "blur_radius_max"),
            ("brightness_factor_min", "brightness_factor_max"),
            ("contrast_factor_min", "contrast_factor_max"),
            ("erosion_dilation_kernel_min", "erosion_dilation_kernel_max"),
            ("cutout_width_min", "cutout_width_max"),
            ("cutout_height_min", "cutout_height_max"),
            ("arc_radius_min", "arc_radius_max"),
            ("sine_amplitude_min", "sine_amplitude_max"),
            ("sine_frequency_min", "sine_frequency_max"),
            ("sine_phase_min", "sine_phase_max"),
            ("per_glyph_palette_size_min", "per_glyph_palette_size_max"),
            ("font_size_min", "font_size_max"),
        ]

        for min_key, max_key in min_max_pairs:
            min_val = spec.get(min_key)
            max_val = spec.get(max_key)

            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    raise ValidationError(
                        f"Specification '{spec_name}': {min_key} ({min_val}) cannot be greater than "
                        f"{max_key} ({max_val})"
                    )

    def _validate_corpus_files(self, spec: Dict[str, Any], spec_name: str) -> None:
        """Validate that corpus files exist.

        Args:
            spec: Specification dictionary.
            spec_name: Name of specification for error messages.

        Raises:
            ValidationError: If corpus files are missing.
        """
        corpus_files = spec.get("corpus_files", [])

        if not corpus_files:
            return  # No corpus files specified, that's okay

        for pattern in corpus_files:
            # Check if any files match this pattern
            matching_files = list(self.corpus_dir.rglob(pattern))

            if len(matching_files) == 0:
                raise ValidationError(
                    f"Specification '{spec_name}': No corpus files found matching pattern '{pattern}' "
                    f"in directory {self.corpus_dir}"
                )

    def _validate_canvas_config(self, canvas: Dict[str, Any], spec_name: str) -> None:
        """Validate canvas configuration parameters.

        Args:
            canvas: Canvas configuration dictionary.
            spec_name: Name of specification for error messages.

        Raises:
            ValidationError: If canvas configuration is invalid.
        """
        # Validate padding values
        min_padding = canvas.get("min_padding")
        max_padding = canvas.get("max_padding")

        if min_padding is not None:
            if not isinstance(min_padding, (int, float)):
                raise ValidationError(
                    f"Specification '{spec_name}': min_padding must be a number"
                )

            if min_padding < 0:
                raise ValidationError(
                    f"Specification '{spec_name}': min_padding must be non-negative, got {min_padding}"
                )

        if max_padding is not None:
            if not isinstance(max_padding, (int, float)):
                raise ValidationError(
                    f"Specification '{spec_name}': max_padding must be a number"
                )

            if max_padding < 0:
                raise ValidationError(
                    f"Specification '{spec_name}': max_padding must be non-negative, got {max_padding}"
                )

        if min_padding is not None and max_padding is not None:
            if min_padding > max_padding:
                raise ValidationError(
                    f"Specification '{spec_name}': min_padding ({min_padding}) cannot be greater than "
                    f"max_padding ({max_padding})"
                )

    def _validate_font_config(self, font: Dict[str, Any], spec_name: str) -> None:
        """Validate font configuration parameters.

        Args:
            font: Font configuration dictionary.
            spec_name: Name of specification for error messages.

        Raises:
            ValidationError: If font configuration is invalid.
        """
        size_range = font.get("size_range")

        if size_range is not None:
            if not isinstance(size_range, list) or len(size_range) != 2:
                raise ValidationError(
                    f"Specification '{spec_name}': font size_range must be a list of 2 numbers"
                )

            min_size, max_size = size_range

            if not isinstance(min_size, (int, float)) or not isinstance(max_size, (int, float)):
                raise ValidationError(
                    f"Specification '{spec_name}': font size_range values must be numbers"
                )

            if min_size <= 0:
                raise ValidationError(
                    f"Specification '{spec_name}': font min size must be positive, got {min_size}"
                )

            if max_size <= 0:
                raise ValidationError(
                    f"Specification '{spec_name}': font max size must be positive, got {max_size}"
                )

            if min_size > max_size:
                raise ValidationError(
                    f"Specification '{spec_name}': font min size ({min_size}) cannot be greater than "
                    f"max size ({max_size})"
                )

    def _validate_direction_config(self, direction: Dict[str, float], spec_name: str) -> None:
        """Validate direction configuration parameters.

        Args:
            direction: Direction weights dictionary.
            spec_name: Name of specification for error messages.

        Raises:
            ValidationError: If direction configuration is invalid.
        """
        if not isinstance(direction, dict):
            raise ValidationError(
                f"Specification '{spec_name}': direction must be a dictionary"
            )

        for dir_name, weight in direction.items():
            if not isinstance(weight, (int, float)):
                raise ValidationError(
                    f"Specification '{spec_name}': direction weight for '{dir_name}' must be a number"
                )

            if weight < 0:
                raise ValidationError(
                    f"Specification '{spec_name}': direction weight for '{dir_name}' cannot be negative, "
                    f"got {weight}"
                )
