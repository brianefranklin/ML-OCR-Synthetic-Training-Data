# How-To: Run a Generation Job

This guide explains how to run the main data generation script, `main.py`.

## Basic Usage

The script is executed from the command line and requires several arguments to specify the configuration and data locations.

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config [PATH_TO_YAML] \
    --output-dir [OUTPUT_DIRECTORY] \
    --font-dir [FONT_DIRECTORY] \
    --background-dir [BACKGROUND_DIRECTORY] \
    --corpus-dir [CORPUS_DIRECTORY]
```

### Argument Breakdown

- **`PYTHONPATH=.`**: This is a crucial environment variable setting. It tells the Python interpreter to include the current directory in its search path, which allows it to find the `src` module correctly.
- **`python3 src/main.py`**: This executes the main script.
- **`--batch-config`**: The path to your YAML configuration file. This file defines the total number of images and the specifications for each batch.
- **`--output-dir`**: The directory where the generated images and their corresponding JSON label files will be saved.
- **`--font-dir`**: The path to the directory containing all your `.ttf` font files.
- **`--background-dir`**: The path to the directory containing all your background images.
- **`--corpus-dir`**: The path to the directory containing your text corpus files.

### Example

```bash
PYTHONPATH=. python3 src/main.py \
    --batch-config batch_config.yaml \
    --output-dir ./output \
    --font-dir ./data.nosync/fonts \
    --background-dir ./data.nosync/backgrounds \
    --corpus-dir ./data.nosync/corpus_text/ltr
```

This command will start the generation process, displaying a progress bar in the console. Upon completion, the `output` directory will contain the generated images and their JSON labels.
