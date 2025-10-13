# API Reference: Batch Processing

This section describes the components responsible for managing and orchestrating the generation of large batches of images.

## `batch_config.py`

### `BatchSpecification`
- **Description:** A dataclass that holds the configuration for a single, specific type of image to be generated (e.g., "10% of images should be right-to-left with a specific font filter").
- **Key Attributes:** `name`, `proportion`, `text_direction`, `corpus_file`, and all the effect/augmentation parameters.

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
