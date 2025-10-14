# Parallel Image Generation

This guide explains how to use parallel processing features to speed up synthetic OCR data generation.

## Overview

The generation pipeline supports two types of parallelization:

1. **Parallel Image Generation** (`--generation-workers`): Parallelizes the CPU-intensive image rendering, effects, and augmentations
2. **Parallel I/O** (`--workers`): Parallelizes the I/O-bound operations of PNG encoding and disk writes

These can be used independently or together to optimize performance based on your hardware and workload characteristics.

## Quick Start

### Fully Sequential (Development/Debugging)

```bash
python3 -m src.main \
  --batch-config configs/batch.yaml \
  --output-dir ./output \
  --font-dir ./data.nosync/fonts \
  --background-dir ./data.nosync/backgrounds \
  --corpus-dir ./data.nosync/corpus \
  --generation-workers 0 \
  --workers 0
```

### Parallel Generation Only (Recommended for Multi-Core Systems)

```bash
python3 -m src.main \
  --batch-config configs/batch.yaml \
  --output-dir ./output \
  --font-dir ./data.nosync/fonts \
  --background-dir ./data.nosync/backgrounds \
  --corpus-dir ./data.nosync/corpus \
  --generation-workers 4 \
  --workers 0
```

### Parallel I/O Only (Default - Backward Compatible)

```bash
python3 -m src.main \
  --batch-config configs/batch.yaml \
  --output-dir ./output \
  --font-dir ./data.nosync/fonts \
  --background-dir ./data.nosync/backgrounds \
  --corpus-dir ./data.nosync/corpus \
  --workers 4 \
  --io-batch-size 10
```

### Both Parallel Generation and I/O (Maximum Performance)

```bash
python3 -m src.main \
  --batch-config configs/batch.yaml \
  --output-dir ./output \
  --font-dir ./data.nosync/fonts \
  --background-dir ./data.nosync/backgrounds \
  --corpus-dir ./data.nosync/corpus \
  --generation-workers 4 \
  --workers 4 \
  --io-batch-size 50
```

## Command-Line Arguments

### `--generation-workers`

Controls the number of parallel processes for image generation (the CPU-bound part):

- **`--generation-workers 0`** or **`--generation-workers 1`**: Sequential generation (single-threaded)
- **`--generation-workers N`** (where N > 1): Parallel generation with N worker processes
- **Default**: `--generation-workers 0` (sequential)

**When to use**: Always use parallel generation on multi-core systems for significant speedup. This parallelizes the most time-consuming part of the pipeline.

### `--workers`

Controls the number of parallel processes for I/O operations (saving images and JSON files):

- **`--workers 0`** or **`--workers 1`**: Sequential I/O (single-threaded)
- **`--workers N`** (where N > 1): Parallel I/O with N worker processes
- **Default**: `--workers 4` (for backward compatibility)

**When to use**: Use parallel I/O on systems with fast storage (SSDs) and large batch sizes. May provide minimal benefit on slower storage or small batches.

### `--io-batch-size`

Controls how many images are accumulated before parallel I/O save:

- **Range**: Any positive integer
- **Default**: `--io-batch-size 10`

**When to use**: Increase for larger batches in production (e.g., 50-100). Decrease for memory-constrained systems.

### Examples

```bash
# Recommended for development (4-core machine)
python3 -m src.main --batch-config configs/batch.yaml --generation-workers 4 --workers 0 ...

# Recommended for production (8-core server with fast SSD)
python3 -m src.main --batch-config configs/batch.yaml --generation-workers 8 --workers 8 --io-batch-size 100 ...

# Use sequential mode for debugging
python3 -m src.main --batch-config configs/batch.yaml --generation-workers 0 --workers 0 ...
```

## When to Use Each Mode

### Parallel Generation (`--generation-workers`)

**Use when**:
- Generating any batch on multi-core systems (almost always beneficial)
- The bottleneck is CPU time, not I/O
- Running on systems with 2+ CPU cores
- Production workloads requiring maximum throughput

**Avoid when**:
- Debugging generation issues (use sequential for easier troubleshooting)
- Running on single-core systems
- Memory is severely constrained

### Parallel I/O (`--workers`)

**Use when**:
- Generating large batches (500+ images) on fast storage (SSD/NVMe)
- Storage can handle multiple concurrent writes
- Production servers with high-performance I/O subsystems
- Testing shows clear I/O bottleneck

**Avoid when**:
- Generating small batches (< 100 images) - overhead may exceed benefits
- Using network-mounted storage (NFS, CIFS) - may cause contention
- Using slow HDDs - concurrent writes may actually slow down
- Debugging file system issues

### Combined Mode (Both)

**Use when**:
- Production environment with multi-core CPUs AND fast storage
- Generating very large batches (1000+ images)
- Both CPU and I/O show significant utilization

**Recommended configurations**:
- **Development** (4-core laptop): `--generation-workers 2 --workers 0`
- **Production** (8-core server, SSD): `--generation-workers 6 --workers 4 --io-batch-size 50`
- **High-end** (16+ cores, NVMe): `--generation-workers 12 --workers 8 --io-batch-size 100`

## How Parallelization Works

### Parallel Generation Mode

When `--generation-workers > 1`:

1. **Task Creation**: All generation tasks are created upfront (deterministic)
2. **Parallel Generation**: Worker processes generate images in parallel using multiprocessing.Pool
3. **Deterministic Seeding**: Each image is seeded by its index, ensuring reproducibility
4. **Result Collection**: Results are collected in order to maintain deterministic output
5. **Saving**: Images are saved either sequentially or in parallel (based on `--workers`)

This design ensures:
- **Significant CPU speedup**: 2-4x faster on multi-core systems
- **Deterministic output**: Same index always produces same image, regardless of execution order
- **Memory overhead**: All generated images held in memory before saving (use care with large batches)

### Parallel I/O Mode

When `--workers > 1`:

1. **Image Generation**: Images are generated (either sequentially or in parallel)
2. **Batching**: Generated images and labels are accumulated in batches (configurable via `--io-batch-size`)
3. **Parallel Saving**: When a batch is full, all image/label pairs are saved concurrently
4. **Worker Processes**: Each worker process handles saving one image and its JSON label

This design ensures:
- **Potential I/O speedup**: Benefit depends on storage performance
- **Memory efficiency**: Only one batch held in memory at a time
- **Backward compatibility**: Works with both sequential and parallel generation

## Performance Considerations

### Optimal Worker Counts

**For Generation Workers (`--generation-workers`)**:
- **Rule of thumb**: Use 50-75% of your physical CPU cores
- **Example**: 8-core CPU → use 4-6 generation workers
- **Why**: Leaves headroom for I/O and system processes
- **Measurement**: Monitor CPU utilization during generation

**For I/O Workers (`--workers`)**:
- **Fast storage (NVMe/SSD)**: Start with 4-8 workers
- **Slow storage (HDD)**: Use 0-2 workers (sequential may be faster)
- **Network storage**: Test carefully - often sequential is better
- **Measurement**: Monitor disk I/O wait time

**For I/O Batch Size (`--io-batch-size`)**:
- **Small batches** (< 100 images): 10-20
- **Medium batches** (100-1000 images): 50-100
- **Large batches** (1000+ images): 100-500
- **Memory constraint**: Reduce if running out of RAM

### Memory Usage

**Parallel Generation Mode**:
- Holds ALL generated images in memory before saving begins
- Memory usage = (number of images) × (average image size)
- For large batches, this can be significant (e.g., 1000 images × 1MB = 1GB)
- Mitigation: Use smaller batch sizes or save incrementally

**Parallel I/O Mode**:
- Holds only one batch in memory at a time
- Memory usage = (`--io-batch-size`) × (average image size)
- Much more memory-efficient for large batches

**Combined Mode**:
- Worst-case memory usage combines both
- Monitor RAM usage and adjust batch sizes accordingly

### File System Considerations

- **Network drives**: May not benefit from parallelism due to network bottlenecks
- **SSDs**: Benefit significantly from parallel writes
- **HDDs**: Limited benefit due to mechanical seek time

## Monitoring Performance

You can monitor the generation process with different logging levels:

```bash
# INFO level (default) - shows progress every 10 images
python3 -m src.main --log-level INFO --workers 4 ...

# DEBUG level - shows detailed information for each image
python3 -m src.main --log-level DEBUG --workers 4 ...
```

Log files are saved with timestamps in the `./logs` directory:

```
./logs/generation_20251014_094912.log
```

The console output shows the matching timestamp at startup:

```
============================================================
Starting OCR generation run: generation_20251014_094912
Log file: ./logs/generation_20251014_094912.log
============================================================
```

## Troubleshooting

### "Process pool terminated unexpectedly"

**Cause**: Worker process crashed, often due to memory exhaustion or file system errors.

**Solution**:
1. Reduce both worker counts: `--generation-workers 2 --workers 2`
2. Check available memory (especially for parallel generation)
3. Verify output directory is writable
4. Try sequential mode for debugging: `--generation-workers 0 --workers 0`
5. Check logs in `./logs/` for detailed error messages

### "Slower than expected performance"

**Cause**: Suboptimal parallelization settings for your hardware.

**Solution**:
1. Monitor CPU and I/O usage during generation
2. Try different generation worker counts (2, 4, 6, 8)
3. For I/O bottlenecks, try different I/O worker counts
4. Ensure output directory is on a fast drive (SSD)
5. Check for background processes consuming resources
6. Run benchmarks (see below) to find optimal settings

### "Different results between sequential and parallel"

**Cause**: This should NOT happen - both modes should produce identical output due to deterministic seeding.

**Solution**:
1. File a bug report with detailed reproduction steps
2. Both sequential and parallel generation use index-based seeding
3. Verify same batch config and random seed are used
4. Check that output files have identical checksums

### "Out of memory errors with parallel generation"

**Cause**: All images held in memory before saving when using `--generation-workers > 1`.

**Solution**:
1. Reduce batch size in your config
2. Use sequential generation with parallel I/O instead
3. Reduce `--generation-workers` count
4. Add more RAM to your system

### "Permission denied" or "File not found" errors

**Cause**: File system permissions or path issues.

**Solution**:
1. Verify output directory exists and is writable
2. Check that all input directories exist
3. Try sequential mode to see clearer error messages: `--generation-workers 0 --workers 0`
4. Check file permissions on output directory

## Implementation Details

For developers interested in the implementation:

**Parallel Generation**:
- **Worker Function**: `generate_image_from_task()` in `src/main.py:37`
- **Determinism**: Index-based seeding with `random.seed(index)` and `np.random.seed(index)`
- **Process Pool**: `multiprocessing.Pool` with `imap()` for progress tracking
- **Tests**: `tests/test_parallel_generation.py` - validates determinism and correctness

**Parallel I/O**:
- **Worker Function**: `save_image_and_label()` in `src/main.py:102`
- **Batching**: Configurable batch size via `--io-batch-size`
- **Serialization**: `NumpyEncoder` handles NumPy data types in JSON
- **Process Pool**: `multiprocessing.Pool` with configurable worker count
- **Tests**: `tests/test_parallel_io.py` - validates I/O operations

**Key Design Decisions**:
- Module-level worker functions (required for pickling in multiprocessing)
- Deterministic generation via index-based seeding
- Results collected in order to maintain reproducibility
- Proper pool cleanup with `close()` and `join()`

See the API reference and test files for more details on the implementation.

## Related Documentation

- [Running Generation](run_generation.md) - Basic usage guide
- [Comprehensive Testing](comprehensive_testing.md) - Testing strategies
- [Profiling](profiling.md) - Performance analysis
- [Architecture](../conceptual/architecture.md) - System overview
