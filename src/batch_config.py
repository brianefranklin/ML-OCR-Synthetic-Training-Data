"""Defines the data structures for batch configurations.

This module contains the dataclasses for representing batch configurations
and a manager class for orchestrating interleaved generation from those batches.
"""

import yaml
from dataclasses import dataclass, field, fields
from typing import List, Type, TypeVar, Dict, Any, Optional, Tuple
from src.distributions import DistributionType

# A generic type for the from_yaml method, allowing it to be used in subclasses.
T = TypeVar('T')

# Valid values for configuration fields
VALID_DISTRIBUTIONS = {"uniform", "normal", "exponential", "beta", "lognormal", "truncated_normal"}
VALID_CURVE_TYPES = {"none", "arc", "sine"}
VALID_TEXT_DIRECTIONS = {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}

@dataclass
class BatchSpecification:
    """Represents the configuration for a single batch.

    This dataclass holds all the parameters that define a specific "flavor" of
    image to be generated as part of a larger batch.

    Attributes:
        name (str): A unique identifier for the batch (e.g., "ancient_rtl_curved").
        proportion (float): The proportion of the total images this batch should represent.
        text_direction (str): The direction of the text rendering (e.g., 'left_to_right').
        corpus_file (str): The filename of the corpus file to use for this batch.
        font_filter (Optional[str]): A glob pattern to filter the fonts to be used.
        min_text_length (int): The minimum length of text to generate.
        max_text_length (int): The maximum length of text to generate.
        glyph_overlap_intensity_min (float): The minimum intensity for glyph overlap.
        glyph_overlap_intensity_max (float): The maximum intensity for glyph overlap.
        glyph_overlap_intensity_distribution (str): Distribution type (default: "half_normal").
        ink_bleed_radius_min (float): The minimum radius for ink bleed.
        ink_bleed_radius_max (float): The maximum radius for ink bleed.
        ink_bleed_radius_distribution (str): Distribution type (default: "half_normal").
        rotation_angle_min (float): The minimum angle for rotation.
        rotation_angle_max (float): The maximum angle for rotation.
        rotation_angle_distribution (str): Distribution type (default: "normal").
        perspective_warp_magnitude_min (float): The minimum magnitude for perspective warp.
        perspective_warp_magnitude_max (float): The maximum magnitude for perspective warp.
        perspective_warp_magnitude_distribution (str): Distribution type (default: "half_normal").
        elastic_distortion_alpha_min (float): The minimum alpha for elastic distortion.
        elastic_distortion_alpha_max (float): The maximum alpha for elastic distortion.
        elastic_distortion_alpha_distribution (str): Distribution type (default: "half_normal").
        elastic_distortion_sigma_min (float): The minimum sigma for elastic distortion.
        elastic_distortion_sigma_max (float): The maximum sigma for elastic distortion.
        elastic_distortion_sigma_distribution (str): Distribution type (default: "half_normal").
        grid_distortion_steps_min (int): The minimum number of steps for grid distortion.
        grid_distortion_steps_max (int): The maximum number of steps for grid distortion.
        grid_distortion_steps_distribution (str): Distribution type (default: "uniform").
        grid_distortion_limit_min (int): The minimum limit for grid distortion.
        grid_distortion_limit_max (int): The maximum limit for grid distortion.
        grid_distortion_limit_distribution (str): Distribution type (default: "half_normal").
        optical_distortion_limit_min (float): The minimum limit for optical distortion.
        optical_distortion_limit_max (float): The maximum limit for optical distortion.
        optical_distortion_limit_distribution (str): Distribution type (default: "half_normal").
        noise_amount_min (float): The minimum amount of noise.
        noise_amount_max (float): The maximum amount of noise.
        noise_amount_distribution (str): Distribution type (default: "half_normal").
        blur_radius_min (float): The minimum radius for blur.
        blur_radius_max (float): The maximum radius for blur.
        blur_radius_distribution (str): Distribution type (default: "half_normal").
        brightness_factor_min (float): The minimum brightness factor.
        brightness_factor_max (float): The maximum brightness factor.
        brightness_factor_distribution (str): Distribution type (default: "normal").
        contrast_factor_min (float): The minimum contrast factor.
        contrast_factor_max (float): The maximum contrast factor.
        contrast_factor_distribution (str): Distribution type (default: "normal").
        erosion_dilation_kernel_min (int): The minimum kernel size for erosion/dilation.
        erosion_dilation_kernel_max (int): The maximum kernel size for erosion/dilation.
        erosion_dilation_kernel_distribution (str): Distribution type (default: "uniform").
        cutout_width_min (int): The minimum width for cutout.
        cutout_width_max (int): The maximum width for cutout.
        cutout_width_distribution (str): Distribution type (default: "uniform").
        cutout_height_min (int): The minimum height for cutout.
        cutout_height_max (int): The maximum height for cutout.
        cutout_height_distribution (str): Distribution type (default: "uniform").
        curve_type (str): The type of curve to apply ("none", "arc", "sine").
        arc_radius_min (float): The minimum radius for arc curves (0.0 = straight line).
        arc_radius_max (float): The maximum radius for arc curves.
        arc_radius_distribution (str): Distribution type (default: "half_normal").
        arc_concave (bool): Whether the arc curves concavely (True) or convexly (False).
        sine_amplitude_min (float): The minimum amplitude for sine wave curves (0.0 = straight line).
        sine_amplitude_max (float): The maximum amplitude for sine wave curves.
        sine_amplitude_distribution (str): Distribution type (default: "half_normal").
        sine_frequency_min (float): The minimum frequency for sine wave curves.
        sine_frequency_max (float): The maximum frequency for sine wave curves.
        sine_frequency_distribution (str): Distribution type (default: "uniform").
        sine_phase_min (float): The minimum phase offset for sine wave curves.
        sine_phase_max (float): The maximum phase offset for sine wave curves.
        sine_phase_distribution (str): Distribution type (default: "uniform").

    Note on Distribution Types:
        - "uniform": Equal probability across entire range (default for discrete parameters)
        - "normal": Bell curve centered at midpoint (for parameters with natural center like rotation)
        - "exponential": Strong bias toward minimum with exponential decay (for degradation effects usually absent)
    """
    name: str
    proportion: float
    text_direction: str
    corpus_file: str
    font_filter: Optional[str] = None
    min_text_length: int = 10
    max_text_length: int = 50
    glyph_overlap_intensity_min: float = 0.0
    glyph_overlap_intensity_max: float = 0.0
    glyph_overlap_intensity_distribution: DistributionType = "exponential"
    ink_bleed_radius_min: float = 0.0
    ink_bleed_radius_max: float = 0.0
    ink_bleed_radius_distribution: DistributionType = "exponential"
    rotation_angle_min: float = 0.0
    rotation_angle_max: float = 0.0
    rotation_angle_distribution: DistributionType = "normal"
    perspective_warp_magnitude_min: float = 0.0
    perspective_warp_magnitude_max: float = 0.0
    perspective_warp_magnitude_distribution: DistributionType = "exponential"
    elastic_distortion_alpha_min: float = 0.0
    elastic_distortion_alpha_max: float = 0.0
    elastic_distortion_alpha_distribution: DistributionType = "exponential"
    elastic_distortion_sigma_min: float = 0.0
    elastic_distortion_sigma_max: float = 0.0
    elastic_distortion_sigma_distribution: DistributionType = "exponential"
    grid_distortion_steps_min: int = 2
    grid_distortion_steps_max: int = 2
    grid_distortion_steps_distribution: DistributionType = "uniform"
    grid_distortion_limit_min: int = 0
    grid_distortion_limit_max: int = 0
    grid_distortion_limit_distribution: DistributionType = "exponential"
    optical_distortion_limit_min: float = 0.0
    optical_distortion_limit_max: float = 0.0
    optical_distortion_limit_distribution: DistributionType = "exponential"
    noise_amount_min: float = 0.0
    noise_amount_max: float = 0.0
    noise_amount_distribution: DistributionType = "exponential"
    blur_radius_min: float = 0.0
    blur_radius_max: float = 0.0
    blur_radius_distribution: DistributionType = "exponential"
    brightness_factor_min: float = 1.0
    brightness_factor_max: float = 1.0
    brightness_factor_distribution: DistributionType = "normal"
    contrast_factor_min: float = 1.0
    contrast_factor_max: float = 1.0
    contrast_factor_distribution: DistributionType = "normal"
    erosion_dilation_kernel_min: int = 1
    erosion_dilation_kernel_max: int = 1
    erosion_dilation_kernel_distribution: DistributionType = "uniform"
    cutout_width_min: int = 0
    cutout_width_max: int = 0
    cutout_width_distribution: DistributionType = "uniform"
    cutout_height_min: int = 0
    cutout_height_max: int = 0
    cutout_height_distribution: DistributionType = "uniform"
    # Curve parameters - always present for consistent ML feature vectors
    curve_type: str = "none"
    arc_radius_min: float = 0.0
    arc_radius_max: float = 0.0
    arc_radius_distribution: DistributionType = "exponential"
    arc_concave: bool = True
    sine_amplitude_min: float = 0.0
    sine_amplitude_max: float = 0.0
    sine_amplitude_distribution: DistributionType = "exponential"
    sine_frequency_min: float = 0.0
    sine_frequency_max: float = 0.0
    sine_frequency_distribution: DistributionType = "uniform"
    sine_phase_min: float = 0.0
    sine_phase_max: float = 0.0
    sine_phase_distribution: DistributionType = "uniform"

@dataclass
class BatchConfig:
    """Represents the entire batch configuration.

    Attributes:
        total_images (int): The total number of images to generate for the entire run.
        specifications (List[BatchSpecification]): A list of individual batch specifications.
    """
    total_images: int
    specifications: List[BatchSpecification] = field(default_factory=list)

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: str) -> T:
        """Loads and parses a YAML file into a BatchConfig object.

        Args:
            yaml_path (str): The path to the YAML configuration file.

        Returns:
            An instance of the class populated with data from the YAML file.

        Raises:
            ValueError: If validation fails for any configuration values.
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data: Dict[str, Any] = yaml.safe_load(f)

        spec_data: List[Dict[str, Any]] = config_data.get('specifications', [])
        specifications = [BatchSpecification(**spec) for spec in spec_data]

        config = cls(
            total_images=config_data.get('total_images', 0),
            specifications=specifications
        )

        # Validate the configuration
        config.validate()

        return config

    def validate(self) -> None:
        """Validates the batch configuration.

        Raises:
            ValueError: If any validation check fails.
        """
        # Validate that proportions sum to 1.0 (within tolerance)
        total_proportion = sum(spec.proportion for spec in self.specifications)
        if abs(total_proportion - 1.0) > 0.001:
            raise ValueError(
                f"Proportions do not sum to 1.0 (sum = {total_proportion:.4f})"
            )

        # Validate each specification
        for spec in self.specifications:
            self._validate_specification(spec)

    @staticmethod
    def _validate_specification(spec: BatchSpecification) -> None:
        """Validates a single batch specification.

        Args:
            spec: The BatchSpecification to validate.

        Raises:
            ValueError: If any validation check fails.
        """
        errors = []

        # Validate distribution fields
        for field_info in fields(spec):
            if field_info.name.endswith('_distribution'):
                value = getattr(spec, field_info.name)
                if value not in VALID_DISTRIBUTIONS:
                    errors.append(
                        f"Invalid distribution '{value}' for field '{field_info.name}'. "
                        f"Must be one of: {VALID_DISTRIBUTIONS}"
                    )

        # Validate curve_type
        if spec.curve_type not in VALID_CURVE_TYPES:
            errors.append(
                f"Invalid curve_type '{spec.curve_type}'. "
                f"Must be one of: {VALID_CURVE_TYPES}"
            )

        # Validate text_direction
        if spec.text_direction not in VALID_TEXT_DIRECTIONS:
            errors.append(
                f"Invalid text_direction '{spec.text_direction}'. "
                f"Must be one of: {VALID_TEXT_DIRECTIONS}"
            )

        # Validate curve_type consistency with curve parameters
        if spec.curve_type == "none":
            # Check if arc parameters are non-zero
            if spec.arc_radius_min != 0.0 or spec.arc_radius_max != 0.0:
                errors.append(
                    f"curve_type is 'none' but arc_radius parameters are non-zero "
                    f"(min={spec.arc_radius_min}, max={spec.arc_radius_max})"
                )
            # Check if sine parameters are non-zero
            if spec.sine_amplitude_min != 0.0 or spec.sine_amplitude_max != 0.0:
                errors.append(
                    f"curve_type is 'none' but sine_amplitude parameters are non-zero "
                    f"(min={spec.sine_amplitude_min}, max={spec.sine_amplitude_max})"
                )

        # If there are errors, raise them all at once
        if errors:
            raise ValueError(
                f"Validation errors in specification '{spec.name}':\n" +
                "\n".join(f"  - {error}" for error in errors)
            )

class BatchManager:
    """Manages the allocation and interleaving of images from different batches.

    This class takes a BatchConfig and determines how many images to generate for
    each specification. It then provides a single, interleaved list of tasks to
    ensure a balanced and diverse dataset is generated from the start.
    """
    def __init__(self, config: BatchConfig):
        """Initializes the BatchManager with a BatchConfig.

        Args:
            config (BatchConfig): The batch configuration object.
        """
        self.config = config
        self._allocation: Dict[str, int] = {}
        self._allocate_images()

    def _allocate_images(self):
        """Allocates the total number of images to batches based on their proportions.
        
        This method handles the rounding of proportions and ensures that the sum
        of allocated images exactly matches the `total_images` requested.
        """
        total_images = self.config.total_images
        specs = self.config.specifications

        # Initial allocation based on proportions, ignoring any fractional remainders.
        allocated_counts = {spec.name: int(spec.proportion * total_images) for spec in specs}

        # Calculate how many images are left over due to rounding down.
        remainder = total_images - sum(allocated_counts.values())

        # Distribute the remainder one by one to the specs.
        # To make the distribution deterministic, we sort the specs by name before
        # distributing the remainder. This ensures that given the same config,
        # the allocation is always the same.
        for spec in sorted(specs, key=lambda s: s.name):
            if remainder <= 0:
                break
            allocated_counts[spec.name] += 1
            remainder -= 1

        self._allocation = allocated_counts

    def get_allocation(self) -> Dict[str, int]:
        """Returns the calculated allocation of images per batch specification.

        Returns:
            A dictionary mapping specification names to the number of images.
        """
        return self._allocation

    def task_list(self) -> List[BatchSpecification]:
        """Generates a full, interleaved list of which specification to use for
        each image to be generated.

        This method produces a list of BatchSpecification objects, where the order
        is interleaved to ensure that images from different specifications are
        generated in a mixed sequence, rather than all at once.

        Returns:
            A list of BatchSpecification objects, with a length equal to total_images.
        """
        tasks: List[BatchSpecification] = []
        remaining_counts = self._allocation.copy()
        
        # Sort specs by name for deterministic interleaving.
        sorted_specs = sorted(self.config.specifications, key=lambda s: s.name)

        # Loop until all images are allocated into the task list.
        while len(tasks) < self.config.total_images:
            # In each iteration, add one task from each spec that still has remaining images.
            for spec in sorted_specs:
                if remaining_counts[spec.name] > 0:
                    tasks.append(spec)
                    remaining_counts[spec.name] -= 1
        
        return tasks