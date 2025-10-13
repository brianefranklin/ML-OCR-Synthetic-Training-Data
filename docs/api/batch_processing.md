# API Reference: Batch Processing

This section describes the components responsible for managing and orchestrating the generation of large batches of images.

## `batch_config.py`

### `BatchSpecification`
- **Description:** A dataclass that holds the configuration for a single, specific type of image to be generated. It defines the allowable ranges for all randomizable parameters.
- **Key Attributes:**
    - `name` (str): A unique identifier for the batch.
    - `proportion` (float): The proportion of the total images this batch should represent.
    - `text_direction` (str): The direction of the text rendering.
    - `corpus_file` (str): The filename of the corpus file to use.
    - `font_filter` (Optional[str]): A glob pattern to filter the fonts to be used for this batch (e.g., `"*Bold.ttf"`).
    - `min_text_length` (int): The minimum length of text to generate.
    - `max_text_length` (int): The maximum length of text to generate.
    - `glyph_overlap_intensity_min` (float): The minimum intensity for glyph overlap.
    - `glyph_overlap_intensity_max` (float): The maximum intensity for glyph overlap.
    - `ink_bleed_radius_min` (float): The minimum radius for ink bleed.
    - `ink_bleed_radius_max` (float): The maximum radius for ink bleed.
    - `rotation_angle_min` (float): The minimum angle for rotation.
    - `rotation_angle_max` (float): The maximum angle for rotation.
    - `perspective_warp_magnitude_min` (float): The minimum magnitude for perspective warp.
    - `perspective_warp_magnitude_max` (float): The maximum magnitude for perspective warp.
    - `elastic_distortion_alpha_min` (float): The minimum alpha for elastic distortion.
    - `elastic_distortion_alpha_max` (float): The maximum alpha for elastic distortion.
    - `elastic_distortion_sigma_min` (float): The minimum sigma for elastic distortion.
    - `elastic_distortion_sigma_max` (float): The maximum sigma for elastic distortion.
    - `grid_distortion_steps_min` (int): The minimum number of steps for grid distortion.
    - `grid_distortion_steps_max` (int): The maximum number of steps for grid distortion.
    - `grid_distortion_limit_min` (int): The minimum limit for grid distortion.
    - `grid_distortion_limit_max` (int): The maximum limit for grid distortion.
    - `optical_distortion_limit_min` (float): The minimum limit for optical distortion.
    - `optical_distortion_limit_max` (float): The maximum limit for optical distortion.
    - `noise_amount_min` (float): The minimum amount of noise.
    - `noise_amount_max` (float): The maximum amount of noise.
    - `blur_radius_min` (float): The minimum radius for blur.
    - `blur_radius_max` (float): The maximum radius for blur.
    - `brightness_factor_min` (float): The minimum brightness factor.
    - `brightness_factor_max` (float): The maximum brightness factor.
    - `contrast_factor_min` (float): The minimum contrast factor.
    - `contrast_factor_max` (float): The maximum contrast factor.
    - `erosion_dilation_kernel_min` (int): The minimum kernel size for erosion/dilation.
    - `erosion_dilation_kernel_max` (int): The maximum kernel size for erosion/dilation.
    - `cutout_width_min` (int): The minimum width for cutout.
    - `cutout_width_max` (int): The maximum width for cutout.
    - `cutout_height_min` (int): The minimum height for cutout.
    - `cutout_height_max` (int): The maximum height for cutout.
    - `curve_type` (str): The type of curve to apply ("none", "arc", "sine"). Default: "none".
    - `arc_radius_min` (float): The minimum radius for arc curves (0.0 = straight line). Default: 0.0.
    - `arc_radius_max` (float): The maximum radius for arc curves. Default: 0.0.
    - `arc_concave` (bool): Whether the arc curves concavely (True) or convexly (False). Default: True.
    - `sine_amplitude_min` (float): The minimum amplitude for sine wave curves (0.0 = straight line). Default: 0.0.
    - `sine_amplitude_max` (float): The maximum amplitude for sine wave curves. Default: 0.0.
    - `sine_frequency_min` (float): The minimum frequency for sine wave curves. Default: 0.0.
    - `sine_frequency_max` (float): The maximum frequency for sine wave curves. Default: 0.0.
    - `sine_phase_min` (float): The minimum phase offset for sine wave curves (radians). Default: 0.0.
    - `sine_phase_max` (float): The maximum phase offset for sine wave curves (radians). Default: 0.0.

**Note on Curve Parameters:** All curve parameters are always present in `BatchSpecification` and in generated plans, even when `curve_type="none"`. This ensures consistent feature vectors for machine learning analysis. Zero values (`arc_radius=0.0`, `sine_amplitude=0.0`) explicitly represent straight-line text. See [Curved Text Rendering](../conceptual/curved_text.md) for details.

### `BatchConfig`
- **Description:** A dataclass that represents the entire batch job, containing the `total_images` to be generated and a list of `BatchSpecification` objects.

### `BatchManager`
- **Description:** Manages the interleaved generation of images from different batches based on their proportions.
- **Key Methods:**
    - `_allocate_images()`: Internally calculates how many images to generate for each specification.
    - `task_list()`: Returns a full, interleaved list of which `BatchSpecification` to use for each image in the batch.

## `generation_orchestrator.py`

### `GenerationOrchestrator`
- **Description:** A high-level class that wires all the other components together.
- **Key Methods:**
    - `__init__(...)`: Initializes all the necessary managers (`BatchManager`, `FontHealthManager`, `BackgroundImageManager`, `CorpusManager`).
    - `create_task_list(...)`: The main entry point for creating the full list of `GenerationTask` objects that will be executed by the main script.
