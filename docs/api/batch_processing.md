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
    - `font_filter` (Optional[str]): A glob pattern to filter the fonts to be used.
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
    - `font_filter` (Optional[str]): A glob pattern to filter the fonts to be used for this batch (e.g., `"*Bold.ttf"`).

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