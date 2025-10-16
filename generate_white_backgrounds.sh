#!/bin/bash

# This script generates 100 white background images with various sizes and common aspect ratios.

OUTPUT_DIR="data.nosync/backgrounds/solid_white"
COLOR_RGB="255 255 255"

echo "Generating 100 white backgrounds..."

# Aspect Ratio 1:1 (20 images)
for size in 10 32 64 128 256 512 1024 1536 2048 2560 3072 3547 48 96 192 384 768 1280 1920 2240; do
    python3 generate_background/solid_color_generator.py --width $size --height $size --rgb $COLOR_RGB --output_dir $OUTPUT_DIR --num_images 1
done

# Aspect Ratio 4:3 (20 images)
for height in 10 24 48 96 192 384 600 768 900 1080 1200 1440 1536 1800 2160 2400 2700 2880 3000 3072; do
    width=$((height * 4 / 3))
    python3 generate_background/solid_color_generator.py --width $width --height $height --rgb $COLOR_RGB --output_dir $OUTPUT_DIR --num_images 1
done

# Aspect Ratio 3:2 (20 images)
for height in 10 21 42 85 170 341 540 682 853 1024 1280 1440 1706 2048 2200 2400 2600 2800 2850 2854; do
    width=$((height * 3 / 2))
    python3 generate_background/solid_color_generator.py --width $width --height $height --rgb $COLOR_RGB --output_dir $OUTPUT_DIR --num_images 1
done

# Aspect Ratio 16:9 (20 images)
for height in 10 18 36 72 144 288 480 720 900 1080 1200 1440 1600 1800 2000 2160 2400 2500 2600 2611; do
    width=$((height * 16 / 9))
    python3 generate_background/solid_color_generator.py --width $width --height $height --rgb $COLOR_RGB --output_dir $OUTPUT_DIR --num_images 1
done

# Aspect Ratio 16:10 (20 images)
for height in 10 20 40 80 160 320 500 640 800 1024 1280 1440 1600 1800 2000 2240 2400 2600 2800 2804; do
    width=$((height * 16 / 10))
    python3 generate_background/solid_color_generator.py --width $width --height $height --rgb $COLOR_RGB --output_dir $OUTPUT_DIR --num_images 1
done

echo "Finished generating backgrounds."
