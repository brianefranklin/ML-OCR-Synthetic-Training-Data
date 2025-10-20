# Synthetic Training Data for OCR

This project is a powerful, language-agnostic, and highly configurable synthetic data generation tool for Optical Character Recognition (OCR). It is designed to create diverse and realistic training data for any writing system, with a focus on universality and performance.

## Key Features

*   **Universality First**: Generate data for any writing system. The tool makes no assumptions about character sets, directionality, or glyphs. It supports:
    *   Left-to-right (e.g., English)
    *   Right-to-left (e.g., Arabic, Hebrew)
    *   Top-to-bottom (e.g., Chinese, Japanese)
    *   Bottom-to-top
*   **Curved Text Rendering**: Simulate text on curved surfaces like bottles and labels by rendering text along circular arcs and sine waves.
*   **Advanced Augmentations**: A rich pipeline of augmentations to simulate real-world conditions:
    *   **Geometric**: Rotation, perspective warp, elastic distortion, grid distortion, and optical (lens) distortion.
    *   **Effects**: Ink bleed, drop shadow, noise, blur, brightness/contrast adjustments, erosion/dilation, and cutouts.
*   **Realistic Parameter Sampling**: Uses 6 different statistical distributions (`uniform`, `normal`, `exponential`, `beta`, `lognormal`, `truncated_normal`) to sample generation parameters, creating more realistic and effective training data.
*   **High Performance**: Optimized for speed with features like:
    *   Parallel image generation and I/O.
    *   Vectorized image processing with NumPy.
    *   Efficient, low-memory corpus handling for massive text datasets.
*   **Reproducibility**: A "plan-then-execute" architecture ensures that every generated image is perfectly reproducible from its corresponding JSON label file.
*   **Highly Configurable**: Use YAML files to define complex batch generation jobs with multiple specifications, each with its own set of parameters and distributions.

## Getting Started

### Prerequisites

*   Python 3.8+
*   See `requirements.txt` for Python dependencies.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running a Generation Job

To run the data generator, you need to provide a batch configuration file and paths to your fonts, backgrounds, and text corpora.

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config batch_config.yaml \
    --output-dir ./output \
    --font-dir ./data.nosync/fonts \
    --background-dir ./data.nosync/backgrounds \
    --corpus-dir ./data.nosync/corpus_text/ltr
```

For more details on running generation jobs, see the [How-To: Run a Generation Job](./docs/how-to/run_generation.md) guide.

## Performance

The data generator is designed for high performance. The benchmark results below show the speedup achieved by using parallel workers for a sample generation task.

| Configuration      | Mean Time (s) | Speedup |
| ------------------ | ------------- | ------- |
| Sequential         | 18.01         | 1.00x   |
| 4 Generation Workers | 7.33          | 2.46x   |
| 8 Generation Workers | 7.20          | 2.50x   |

For more details on performance and profiling, see the [How-To: Profile Generation Performance](./docs/how-to/profiling.md) guide.

## Documentation

This project is documentation-driven. The `docs` directory contains detailed information on all aspects of the project, including:

*   **Conceptual Guides**: High-level explanations of the architecture, curved text rendering, and statistical distributions.
*   **How-To Guides**: Step-by-step instructions for common tasks like adding new augmentations, running generation jobs, and profiling performance.
*   **API Reference**: Detailed documentation for all modules, classes, and functions.

## Contributing

We welcome contributions! Please see our [contributing guidelines](./GEMINI.md) for more information on our development workflow, coding standards, and how to submit pull requests.

## License

Copyright (c) 2025 Brian E. Franklin. All rights reserved.