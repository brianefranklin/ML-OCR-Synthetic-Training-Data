# API Reference: Corpus Management

This section describes the component responsible for efficiently handling large text corpora.

## `corpus_manager.py`

### `CorpusManager`
- **Description:** A class designed for efficient, low-memory handling of massive text corpora.
- **Key Features:**
    - **Sequential Streaming:** Reads from corpus files in a round-robin fashion, loading only small chunks into memory at a time. This allows the generator to work with terabyte-scale text datasets without high memory usage.
    - **Efficient Text Extraction:** The `extract_text_segment(min_length, max_length)` method randomly selects a segment of text from the currently loaded buffer.
