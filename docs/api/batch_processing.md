# API Reference: Batch Processing

This section describes the components responsible for managing and orchestrating the generation of large batches of images.

## `batch_config.py`

### `BatchSpecification`
- **Description:** A dataclass that holds the configuration for a single, specific type of image to be generated. It defines the allowable ranges for all randomizable parameters and their probability distributions.
- **Key Attributes:**

#### Basic Configuration
- `name` (str): A unique identifier for the batch.
- `proportion` (float): The proportion of the total images this batch should represent.
- `text_direction` (str): The direction of the text rendering.
- `corpus_file` (str): The filename of the corpus file to use.
- `font_filter` (Optional[str]): A glob pattern to filter the fonts to be used for this batch (e.g., `"*Bold.ttf"`).
- `min_text_length` (int): The minimum length of text to generate.
- `max_text_length` (int): The maximum length of text to generate.

#### Parameter Ranges and Distributions

For each randomizable parameter, there are three related fields:
1. `<param>_min` - minimum value
2. `<param>_max` - maximum value
3. `<param>_distribution` - probability distribution type (`"uniform"`, `"normal"`, or `"exponential"`)

##### Text Effects
- `glyph_overlap_intensity_min` (float), `glyph_overlap_intensity_max` (float), `glyph_overlap_intensity_distribution` (str): Glyph overlap parameters. Default distribution: `"exponential"`.
- `ink_bleed_radius_min` (float), `ink_bleed_radius_max` (float), `ink_bleed_radius_distribution` (str): Ink bleed effect parameters. Default distribution: `"exponential"`.

##### Geometric Transformations
- `rotation_angle_min` (float), `rotation_angle_max` (float), `rotation_angle_distribution` (str): Rotation angle in degrees. Default distribution: `"normal"` (centered at 0°).
- `perspective_warp_magnitude_min` (float), `perspective_warp_magnitude_max` (float), `perspective_warp_magnitude_distribution` (str): Perspective warp strength. Default distribution: `"exponential"`.

##### Non-Linear Distortions
- `elastic_distortion_alpha_min` (float), `elastic_distortion_alpha_max` (float), `elastic_distortion_alpha_distribution` (str): Elastic distortion alpha parameter. Default distribution: `"exponential"`.
- `elastic_distortion_sigma_min` (float), `elastic_distortion_sigma_max` (float), `elastic_distortion_sigma_distribution` (str): Elastic distortion sigma parameter. Default distribution: `"exponential"`.
- `grid_distortion_steps_min` (int), `grid_distortion_steps_max` (int), `grid_distortion_steps_distribution` (str): Grid distortion step count. Default distribution: `"uniform"`.
- `grid_distortion_limit_min` (int), `grid_distortion_limit_max` (int), `grid_distortion_limit_distribution` (str): Grid distortion magnitude limit. Default distribution: `"exponential"`.
- `optical_distortion_limit_min` (float), `optical_distortion_limit_max` (float), `optical_distortion_limit_distribution` (str): Optical distortion limit. Default distribution: `"exponential"`.

##### Image Quality Effects
- `noise_amount_min` (float), `noise_amount_max` (float), `noise_amount_distribution` (str): Noise intensity. Default distribution: `"exponential"`.
- `blur_radius_min` (float), `blur_radius_max` (float), `blur_radius_distribution` (str): Gaussian blur radius. Default distribution: `"exponential"`.
- `brightness_factor_min` (float), `brightness_factor_max` (float), `brightness_factor_distribution` (str): Brightness multiplier (1.0 = normal). Default distribution: `"normal"` (centered at 1.0).
- `contrast_factor_min` (float), `contrast_factor_max` (float), `contrast_factor_distribution` (str): Contrast multiplier (1.0 = normal). Default distribution: `"normal"` (centered at 1.0).
- `erosion_dilation_kernel_min` (int), `erosion_dilation_kernel_max` (int), `erosion_dilation_kernel_distribution` (str): Kernel size for morphological operations. Default distribution: `"uniform"`.
- `cutout_width_min` (int), `cutout_width_max` (int), `cutout_width_distribution` (str): Cutout width in pixels. Default distribution: `"uniform"`.
- `cutout_height_min` (int), `cutout_height_max` (int), `cutout_height_distribution` (str): Cutout height in pixels. Default distribution: `"uniform"`.

##### Curve Parameters
- `curve_type` (str): The type of curve to apply (`"none"`, `"arc"`, `"sine"`). Default: `"none"`.
- `arc_radius_min` (float), `arc_radius_max` (float), `arc_radius_distribution` (str): Arc curve radius (0.0 = straight). Default distribution: `"exponential"`.
- `arc_concave` (bool): Whether arc curves concavely (True) or convexly (False). Default: True.
- `sine_amplitude_min` (float), `sine_amplitude_max` (float), `sine_amplitude_distribution` (str): Sine wave amplitude (0.0 = straight). Default distribution: `"exponential"`.
- `sine_frequency_min` (float), `sine_frequency_max` (float), `sine_frequency_distribution` (str): Sine wave frequency. Default distribution: `"uniform"`.
- `sine_phase_min` (float), `sine_phase_max` (float), `sine_phase_distribution` (str): Sine wave phase offset (radians). Default distribution: `"uniform"`.

**Note on Curve Parameters:** All curve parameters are always present in `BatchSpecification` and in generated plans, even when `curve_type="none"`. This ensures consistent feature vectors for machine learning analysis. Zero values (`arc_radius=0.0`, `sine_amplitude=0.0`) explicitly represent straight-line text. See [Curved Text Rendering](../conceptual/curved_text.md) for details.

**Note on Distribution Types:** Distribution types control the probability of sampling values within the specified range:
- `"uniform"`: Equal probability across entire range (default for discrete parameters)
- `"normal"`: Bell curve centered at midpoint (for parameters with natural center like rotation)
- `"exponential"`: Biased toward minimum with exponential decay (for effects usually absent in real-world data)

See [Statistical Distributions](../conceptual/distributions.md) for detailed guidance on choosing distributions.

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
