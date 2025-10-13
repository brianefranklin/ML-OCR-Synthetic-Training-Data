"""
Validation tests to verify that distribution implementations match intended behavior.

These tests generate larger sample sets and analyze the distribution shapes to ensure
they match the real-world frequency patterns we're trying to model.
"""

import random
import numpy as np
import pytest
from src.distributions import sample_parameter


def test_exponential_arc_radius_real_world_pattern():
    """
    Validates that exponential for arc_radius creates realistic pattern:
    - Most text should be straight (arc_radius close to 0)
    - Curved text should be rare but present
    - Heavily curved text should be very rare

    Target pattern: ~60-70% straight, ~20-30% moderately curved, ~5-10% heavily curved
    """
    random.seed(42)

    # Generate 10,000 samples to get stable statistics
    samples = [sample_parameter(0.0, 300.0, "exponential") for _ in range(10000)]

    # Count samples in different buckets
    straight = sum(1 for s in samples if s < 10.0)  # Nearly straight (radius < 10)
    slight_curve = sum(1 for s in samples if 10.0 <= s < 50.0)  # Slight curve
    moderate_curve = sum(1 for s in samples if 50.0 <= s < 100.0)  # Moderate curve
    strong_curve = sum(1 for s in samples if 100.0 <= s < 200.0)  # Strong curve
    extreme_curve = sum(1 for s in samples if s >= 200.0)  # Extreme curve

    total = len(samples)

    # Print distribution for manual inspection
    print(f"\nArc Radius Distribution (n={total}):")
    print(f"  Nearly straight (0-10):     {straight:5d} ({100*straight/total:5.1f}%)")
    print(f"  Slight curve (10-50):       {slight_curve:5d} ({100*slight_curve/total:5.1f}%)")
    print(f"  Moderate curve (50-100):    {moderate_curve:5d} ({100*moderate_curve/total:5.1f}%)")
    print(f"  Strong curve (100-200):     {strong_curve:5d} ({100*strong_curve/total:5.1f}%)")
    print(f"  Extreme curve (200-300):    {extreme_curve:5d} ({100*extreme_curve/total:5.1f}%)")

    # Statistical assertions for exponential distribution
    # At least 60% should be nearly straight (< 10) - MATCHES YOUR GOAL
    assert straight >= 6000, f"Expected >=60% nearly straight, got {100*straight/total:.1f}%"

    # Most samples should be in lower 20% (< 60)
    lower_fifth = sum(1 for s in samples if s < 60.0)
    assert lower_fifth >= 9500, f"Expected >=95% in lower 20%, got {100*lower_fifth/total:.1f}%"

    # Extreme curves should be extremely rare (<< 1%)
    assert extreme_curve <= 100, f"Expected <<1% extreme curves, got {100*extreme_curve/total:.1f}%"

    # Mean should be very close to min (around 10, which is 3.3% of range)
    mean = np.mean(samples)
    print(f"  Mean: {mean:.1f} (should be ~3% of range, strongly biased)")
    assert 5 <= mean <= 15, f"Mean {mean:.1f} not in expected range [5, 15]"


def test_exponential_blur_real_world_pattern():
    """
    Validates that exponential for blur creates realistic pattern:
    - Most images should be sharp (blur ≈ 0)
    - Some images slightly blurred
    - Heavy blur should be rare
    """
    random.seed(123)

    samples = [sample_parameter(0.0, 5.0, "exponential") for _ in range(10000)]

    sharp = sum(1 for s in samples if s < 0.5)  # Nearly sharp
    slight_blur = sum(1 for s in samples if 0.5 <= s < 1.5)
    moderate_blur = sum(1 for s in samples if 1.5 <= s < 3.0)
    heavy_blur = sum(1 for s in samples if s >= 3.0)

    total = len(samples)

    print(f"\nBlur Radius Distribution (n={total}):")
    print(f"  Sharp (0-0.5):         {sharp:5d} ({100*sharp/total:5.1f}%)")
    print(f"  Slight blur (0.5-1.5): {slight_blur:5d} ({100*slight_blur/total:5.1f}%)")
    print(f"  Moderate blur (1.5-3): {moderate_blur:5d} ({100*moderate_blur/total:5.1f}%)")
    print(f"  Heavy blur (3-5):      {heavy_blur:5d} ({100*heavy_blur/total:5.1f}%)")

    # Most images should be sharp with exponential distribution
    assert sharp >= 9000, f"Expected >=90% sharp, got {100*sharp/total:.1f}%"

    # Heavy blur should be extremely rare
    assert heavy_blur <= 10, f"Expected <<1% heavy blur, got {100*heavy_blur/total:.1f}%"

    mean = np.mean(samples)
    print(f"  Mean: {mean:.2f} (should be ~0.15-0.20, strongly biased toward 0)")
    assert 0.1 <= mean <= 0.25, f"Mean {mean:.2f} not in expected range [0.1, 0.25]"


def test_normal_rotation_real_world_pattern():
    """
    Validates that normal distribution for rotation creates realistic pattern:
    - Most images should be nearly upright (rotation ≈ 0°)
    - Moderate rotations should be common
    - Extreme rotations should be rare but present
    """
    random.seed(456)

    samples = [sample_parameter(-15.0, 15.0, "normal") for _ in range(10000)]

    upright = sum(1 for s in samples if abs(s) < 2.5)  # Nearly upright (±2.5°)
    slight_tilt = sum(1 for s in samples if 2.5 <= abs(s) < 5.0)
    moderate_tilt = sum(1 for s in samples if 5.0 <= abs(s) < 10.0)
    strong_tilt = sum(1 for s in samples if abs(s) >= 10.0)

    total = len(samples)

    print(f"\nRotation Angle Distribution (n={total}):")
    print(f"  Nearly upright (±2.5°):  {upright:5d} ({100*upright/total:5.1f}%)")
    print(f"  Slight tilt (2.5-5°):    {slight_tilt:5d} ({100*slight_tilt/total:5.1f}%)")
    print(f"  Moderate tilt (5-10°):   {moderate_tilt:5d} ({100*moderate_tilt/total:5.1f}%)")
    print(f"  Strong tilt (10-15°):    {strong_tilt:5d} ({100*strong_tilt/total:5.1f}%)")

    # Should be centered around 0
    mean = np.mean(samples)
    print(f"  Mean: {mean:.2f}° (should be ≈0°)")
    assert abs(mean) < 1.0, f"Mean {mean:.2f}° not centered near 0°"

    # About 40% should be nearly upright (within 1 sigma)
    # sigma = 30/6 = 5, so 1 sigma = 5°, but we're checking ±2.5° ≈ 0.5 sigma
    # This should be roughly 38-40% of samples
    assert 3000 <= upright <= 4500, f"Expected ~35-45% nearly upright, got {100*upright/total:.1f}%"

    # Strong tilts should be rare (beyond 2 sigma = 10°)
    assert strong_tilt <= 500, f"Expected <=5% strong tilt, got {100*strong_tilt/total:.1f}%"


def test_compare_distributions_side_by_side():
    """
    Compares all three distributions side-by-side for the same parameter range
    to visualize the differences in sampling behavior.
    """
    random.seed(789)

    # Generate 10,000 samples for each distribution type
    uniform_samples = [sample_parameter(0.0, 100.0, "uniform") for _ in range(10000)]
    normal_samples = [sample_parameter(0.0, 100.0, "normal") for _ in range(10000)]
    exponential_samples = [sample_parameter(0.0, 100.0, "exponential") for _ in range(10000)]

    # Divide into 10 bins
    bins = [(i*10, (i+1)*10) for i in range(10)]

    print(f"\nDistribution Comparison (range 0-100, n=10000):")
    print(f"{'Bin':<12} {'Uniform':<12} {'Normal':<12} {'Exponential':<12}")
    print("-" * 48)

    for bin_min, bin_max in bins:
        uniform_count = sum(1 for s in uniform_samples if bin_min <= s < bin_max)
        normal_count = sum(1 for s in normal_samples if bin_min <= s < bin_max)
        exponential_count = sum(1 for s in exponential_samples if bin_min <= s < bin_max)

        print(f"[{bin_min:3d}-{bin_max:3d})  "
              f"{uniform_count:5d} ({100*uniform_count/10000:4.1f}%)  "
              f"{normal_count:5d} ({100*normal_count/10000:4.1f}%)  "
              f"{exponential_count:5d} ({100*exponential_count/10000:4.1f}%)")

    # Calculate and display statistics
    print("\nSummary Statistics:")
    print(f"{'Statistic':<15} {'Uniform':<12} {'Normal':<12} {'Exponential':<12}")
    print("-" * 51)
    print(f"{'Mean':<15} {np.mean(uniform_samples):6.1f}       "
          f"{np.mean(normal_samples):6.1f}       "
          f"{np.mean(exponential_samples):6.1f}")
    print(f"{'Median':<15} {np.median(uniform_samples):6.1f}       "
          f"{np.median(normal_samples):6.1f}       "
          f"{np.median(exponential_samples):6.1f}")
    print(f"{'Std Dev':<15} {np.std(uniform_samples):6.1f}       "
          f"{np.std(normal_samples):6.1f}       "
          f"{np.std(exponential_samples):6.1f}")
    print(f"{'% in [0-30)':<15} {100*sum(1 for s in uniform_samples if s < 30)/10000:5.1f}%       "
          f"{100*sum(1 for s in normal_samples if s < 30)/10000:5.1f}%       "
          f"{100*sum(1 for s in exponential_samples if s < 30)/10000:5.1f}%")

    # Verify expected patterns
    # Uniform should be roughly flat (~1000 per bin)
    uniform_hist, _ = np.histogram(uniform_samples, bins=10, range=(0, 100))
    for count in uniform_hist:
        assert 800 <= count <= 1200, f"Uniform bin count {count} outside expected range"

    # Normal should be peaked in middle bins
    normal_hist, _ = np.histogram(normal_samples, bins=10, range=(0, 100))
    middle_bins = normal_hist[4:6]  # Bins 40-60
    assert all(count > 1200 for count in middle_bins), "Normal not peaked in middle"

    # Exponential should be strongly peaked in first bin
    exponential_hist, _ = np.histogram(exponential_samples, bins=10, range=(0, 100))
    assert exponential_hist[0] > 5000, \
        f"Exponential first bin ({exponential_hist[0]}) not strongly biased"


def test_exponential_percentage_distribution():
    """
    Tests what percentage of samples fall into different ranges for exponential.
    This helps verify it matches the intended "most samples near minimum" behavior.
    """
    random.seed(999)

    # Test with range [0, 300] (typical for arc_radius)
    samples = [sample_parameter(0.0, 300.0, "exponential") for _ in range(10000)]

    # Calculate percentiles
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    values = [np.percentile(samples, p) for p in percentiles]

    print(f"\nExponential Percentiles (range 0-300):")
    print(f"{'Percentile':<12} {'Value':<12} {'Interpretation'}")
    print("-" * 60)

    interpretations = {
        10: "10% of samples below this value",
        50: "Median - half of samples below this",
        90: "90% of samples below this value",
        95: "95% of samples below this value",
        99: "99% of samples below this value (only 1% above)"
    }

    for p, v in zip(percentiles, values):
        interp = interpretations.get(p, "")
        print(f"{p:3d}%         {v:6.1f}       {interp}")

    # Key assertions about the distribution shape
    p50 = np.percentile(samples, 50)  # Median
    p90 = np.percentile(samples, 90)
    p95 = np.percentile(samples, 95)

    # Median should be very close to minimum (< 10)
    assert p50 < 15, f"Median {p50:.1f} should be < 15 (strongly biased)"

    # 90% of samples should be in lower third of range
    assert p90 < 100, f"90th percentile {p90:.1f} should be < 100"

    # 95% of samples should be well below half of max
    assert p95 < 150, f"95th percentile {p95:.1f} should be < 150"

    print(f"\n✓ Exponential is strongly biased toward minimum")
    print(f"  - 50% of samples < {p50:.1f} ({100*p50/300:.0f}% of range)")
    print(f"  - 90% of samples < {p90:.1f} ({100*p90/300:.0f}% of range)")
    print(f"  - 95% of samples < {p95:.1f} ({100*p95/300:.0f}% of range)")


def test_realistic_dataset_composition():
    """
    Simulates generating a real dataset to show the final composition
    with the default distributions.
    """
    random.seed(12345)

    # Simulate generating 1000 images with typical ranges
    n_images = 1000

    # Arc radius (exponential, 0-300)
    arc_radii = [sample_parameter(0.0, 300.0, "exponential") for _ in range(n_images)]
    straight_text = sum(1 for r in arc_radii if r < 10.0)
    slightly_curved = sum(1 for r in arc_radii if 10.0 <= r < 100.0)
    heavily_curved = sum(1 for r in arc_radii if r >= 100.0)

    # Blur radius (exponential, 0-5)
    blur_radii = [sample_parameter(0.0, 5.0, "exponential") for _ in range(n_images)]
    sharp = sum(1 for b in blur_radii if b < 0.5)
    blurred = sum(1 for b in blur_radii if b >= 0.5)

    # Rotation angle (normal, -15 to 15)
    rotations = [sample_parameter(-15.0, 15.0, "normal") for _ in range(n_images)]
    upright = sum(1 for r in rotations if abs(r) < 5.0)
    tilted = sum(1 for r in rotations if abs(r) >= 5.0)

    # Noise amount (exponential, 0-0.3)
    noise = [sample_parameter(0.0, 0.3, "exponential") for _ in range(n_images)]
    clean = sum(1 for n in noise if n < 0.05)
    noisy = sum(1 for n in noise if n >= 0.05)

    print(f"\nRealistic Dataset Composition (n={n_images}):")
    print(f"\n{'Category':<20} {'Count':<8} {'Percentage'}")
    print("-" * 45)
    print(f"\n{'Arc Radius (Curvature):'}")
    print(f"  {'Straight (<10)':<18} {straight_text:5d}    {100*straight_text/n_images:5.1f}%")
    print(f"  {'Slightly curved':<18} {slightly_curved:5d}    {100*slightly_curved/n_images:5.1f}%")
    print(f"  {'Heavily curved':<18} {heavily_curved:5d}    {100*heavily_curved/n_images:5.1f}%")

    print(f"\n{'Blur:'}")
    print(f"  {'Sharp (<0.5)':<18} {sharp:5d}    {100*sharp/n_images:5.1f}%")
    print(f"  {'Blurred (≥0.5)':<18} {blurred:5d}    {100*blurred/n_images:5.1f}%")

    print(f"\n{'Rotation:'}")
    print(f"  {'Upright (±5°)':<18} {upright:5d}    {100*upright/n_images:5.1f}%")
    print(f"  {'Tilted (>5°)':<18} {tilted:5d}    {100*tilted/n_images:5.1f}%")

    print(f"\n{'Noise:'}")
    print(f"  {'Clean (<0.05)':<18} {clean:5d}    {100*clean/n_images:5.1f}%")
    print(f"  {'Noisy (≥0.05)':<18} {noisy:5d}    {100*noisy/n_images:5.1f}%")

    # Verify the distributions create realistic proportions
    # With exponential distribution, most images should be "pristine"
    assert straight_text >= 600, f"Should have >=60% straight text, got {straight_text}"
    assert sharp >= 900, f"Should have >=90% sharp images, got {sharp}"
    assert clean >= 950, f"Should have >=95% clean images, got {clean}"

    # But some variety should still exist
    assert slightly_curved >= 300, f"Should have some curved text, got {slightly_curved}"
    assert blurred >= 10, f"Should have at least some blurred images, got {blurred}"
    assert noisy >= 1, f"Should have at least some noisy images, got {noisy}"

    print(f"\n✓ Exponential distribution creates realistic degradation dataset:")
    print(f"  - Vast majority of images are 'pristine' (straight, sharp, no noise)")
    print(f"  - Occasional degradation provides robustness training")
    print(f"  - Matches real-world: most OCR input is clean, with rare poor quality")
