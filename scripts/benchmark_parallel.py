#!/usr/bin/env python3
"""Benchmark script for parallel I/O performance testing.

This script runs the OCR data generator with different worker counts
and measures the execution time to quantify the performance improvements
from parallel I/O.

Usage:
    python3 scripts/benchmark_parallel.py

Results are saved to benchmark_results.json and a summary is printed.
"""

import subprocess
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Configuration
GENERATION_WORKER_COUNTS = [0, 2, 4, 6, 8]  # Generation worker configurations to test
IO_WORKER_COUNTS = [0, 4, 8]  # I/O worker configurations to test
NUM_RUNS = 2  # Number of runs per configuration for averaging
BATCH_CONFIG = "configs/benchmark_batch.yaml"  # Small config for faster benchmarking
FONT_DIR = "data.nosync/fonts"
BACKGROUND_DIR = "data.nosync/backgrounds"
CORPUS_DIR = "data.nosync/corpus_text"


def run_generation(generation_workers: int, io_workers: int, output_dir: str) -> float:
    """Run generation with specified worker counts and measure time.

    Args:
        generation_workers: Number of generation worker processes (0 = sequential)
        io_workers: Number of I/O worker processes (0 = sequential)
        output_dir: Directory for output files

    Returns:
        Execution time in seconds
    """
    cmd = [
        "python3", "-m", "src.main",
        "--batch-config", BATCH_CONFIG,
        "--output-dir", output_dir,
        "--font-dir", FONT_DIR,
        "--background-dir", BACKGROUND_DIR,
        "--corpus-dir", CORPUS_DIR,
        "--generation-workers", str(generation_workers),
        "--workers", str(io_workers),
        "--io-batch-size", "20",
        "--log-level", "WARNING"  # Suppress most output for cleaner benchmarking
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"Error running generation with gen={generation_workers}, io={io_workers} workers:")
        print(result.stderr)
        raise RuntimeError(f"Generation failed with return code {result.returncode}")

    return end_time - start_time


def benchmark_configuration(generation_workers: int, io_workers: int, num_runs: int) -> Dict[str, Any]:
    """Benchmark a specific worker configuration.

    Args:
        generation_workers: Number of generation worker processes
        io_workers: Number of I/O worker processes
        num_runs: Number of times to run for averaging

    Returns:
        Dictionary with timing statistics
    """
    config_name = f"gen={generation_workers}, io={io_workers}"
    print(f"\nBenchmarking {config_name}...")
    times: List[float] = []

    for run in range(num_runs):
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
            elapsed = run_generation(generation_workers, io_workers, temp_dir)
            times.append(elapsed)
            print(f"{elapsed:.2f}s")

    return {
        "generation_workers": generation_workers,
        "io_workers": io_workers,
        "config_name": config_name,
        "runs": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times)
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary of benchmark results.

    Args:
        results: List of benchmark result dictionaries
    """
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Configuration':<30} {'Mean (s)':<12} {'Median (s)':<12} {'StdDev (s)':<12} {'Speedup':<10}")
    print("-" * 90)

    # Get baseline (sequential) time
    baseline = next((r for r in results if r["generation_workers"] == 0 and r["io_workers"] == 0), results[0])
    baseline_mean = baseline["mean"]

    for result in results:
        speedup = baseline_mean / result["mean"]

        print(f"{result['config_name']:<30} {result['mean']:<12.2f} {result['median']:<12.2f} "
              f"{result['stdev']:<12.2f} {speedup:<10.2f}x")

    print("=" * 90)

    # Find best configuration
    best = min(results, key=lambda r: r["mean"])
    print(f"\nBest configuration: {best['config_name']}")
    print(f"  Mean time: {best['mean']:.2f}s")
    print(f"  Speedup: {baseline_mean / best['mean']:.2f}x over sequential")

    # Print recommendation
    print("\nRecommendation:")
    if best["generation_workers"] == 0:
        print("  Sequential generation is fastest for this workload.")
        print("  This may indicate a small batch or single-core system.")
    else:
        print(f"  Use --generation-workers {best['generation_workers']} --workers {best['io_workers']} for optimal performance.")
        print(f"  This provides a {baseline_mean / best['mean']:.2f}x speedup over fully sequential mode.")

    # Analyze impact of parallel generation vs parallel I/O
    gen_only = next((r for r in results if r["generation_workers"] > 0 and r["io_workers"] == 0), None)
    io_only = next((r for r in results if r["generation_workers"] == 0 and r["io_workers"] > 0), None)

    print("\nAnalysis:")
    if gen_only:
        gen_speedup = baseline_mean / gen_only["mean"]
        print(f"  Parallel generation impact: {gen_speedup:.2f}x speedup")

    if io_only:
        io_speedup = baseline_mean / io_only["mean"]
        print(f"  Parallel I/O impact: {io_speedup:.2f}x speedup")
        if io_speedup < 1.1:
            print("  → Parallel I/O provides minimal benefit for this workload")


def main():
    """Main benchmark execution."""
    print("OCR Data Generator - Comprehensive Parallel Benchmark")
    print("=" * 90)
    print(f"Configuration:")
    print(f"  Batch config: {BATCH_CONFIG}")
    print(f"  Generation worker counts: {GENERATION_WORKER_COUNTS}")
    print(f"  I/O worker counts: {IO_WORKER_COUNTS}")
    print(f"  Runs per configuration: {NUM_RUNS}")
    print(f"  Font directory: {FONT_DIR}")
    print(f"  Background directory: {BACKGROUND_DIR}")
    print(f"  Corpus directory: {CORPUS_DIR}")

    # Verify batch config exists
    if not Path(BATCH_CONFIG).exists():
        print(f"\nError: Batch config not found: {BATCH_CONFIG}")
        print("Creating a benchmark configuration...")

        # Create benchmark config directory if needed
        config_dir = Path(BATCH_CONFIG).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Create a small benchmark config (50 images)
        benchmark_config = {
            "total_images": 50,
            "specifications": [
                {
                    "name": "benchmark_spec",
                    "count": 50,
                    "corpus_files": ["*.txt"],
                    "font_filter": {"extensions": [".ttf"]},
                    "config": {
                        "direction": {"ltr": 0.5, "rtl": 0.5},
                        "canvas": {
                            "min_padding": 10,
                            "max_padding": 50,
                            "placement": "uniform_random"
                        },
                        "font": {
                            "size_range": [20, 60]
                        },
                        "effects": {
                            "rotation": {"enabled": True, "range": [-5, 5]},
                            "noise": {"enabled": True, "range": [0.0, 0.01]},
                            "blur": {"enabled": True, "range": [0.0, 1.0]}
                        }
                    }
                }
            ]
        }

        import yaml
        with open(BATCH_CONFIG, 'w') as f:
            yaml.dump(benchmark_config, f, default_flow_style=False)

        print(f"Created benchmark config: {BATCH_CONFIG}")

    # Run benchmarks - test key configurations
    results: List[Dict[str, Any]] = []

    # Test configurations:
    # 1. Fully sequential (baseline)
    # 2. Parallel generation only (most important)
    # 3. Parallel I/O only (for comparison)
    # 4. Combined parallel generation and I/O

    configs_to_test = [
        (0, 0),  # Baseline: fully sequential
    ]

    # Add parallel generation configs
    for gen_workers in GENERATION_WORKER_COUNTS:
        if gen_workers > 0:
            configs_to_test.append((gen_workers, 0))  # Gen only

    # Add one parallel I/O config
    if len(IO_WORKER_COUNTS) > 1:
        configs_to_test.append((0, IO_WORKER_COUNTS[1]))  # I/O only

    # Add combined configs (best generation workers with different I/O)
    if len(GENERATION_WORKER_COUNTS) > 2:
        best_gen = GENERATION_WORKER_COUNTS[len(GENERATION_WORKER_COUNTS) // 2]  # Middle value
        for io_workers in IO_WORKER_COUNTS:
            if io_workers > 0:
                configs_to_test.append((best_gen, io_workers))

    for gen_workers, io_workers in configs_to_test:
        try:
            result = benchmark_configuration(gen_workers, io_workers, NUM_RUNS)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Skipping gen={gen_workers}, io={io_workers}")
            continue

    if not results:
        print("\nNo successful benchmark runs. Check your configuration.")
        return

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
