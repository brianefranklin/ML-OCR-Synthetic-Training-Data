#!/bin/bash

# This script runs a battery of tests for the synthetic data generation script.
# It generates 10 images for each text direction, with a unique font and text string for each image.

# Clear the output directory once at the beginning
python3 src/main.py --output-dir output --clear-output --force --num-images 0

# Define the text directions
declare -a directions=("left_to_right" "right_to_left" "top_to_bottom" "bottom_to_top")

# Get a list of all available fonts
mapfile -t fonts < <(ls data/fonts)

# Check if we have fonts
if [ ${#fonts[@]} -eq 0 ]; then
    echo "Error: No fonts found in data/fonts directory"
    exit 1
fi

# Loop through each text direction
for direction in "${directions[@]}"
do
    echo "Generating 10 images for text direction: $direction"

    # Select appropriate corpus file for the direction
    text_file="data/raw_text/corpus.txt"
    if [ "$direction" == "right_to_left" ]; then
        text_file="data/raw_text/arabic_corpus.txt"
    elif [ "$direction" == "top_to_bottom" ] || [ "$direction" == "bottom_to_top" ]; then
        text_file="data/raw_text/japanese_corpus.txt"
    fi

    # Check if corpus file exists
    if [ ! -f "$text_file" ]; then
        echo "Error: Corpus file $text_file not found. Using default corpus.txt"
        text_file="data/raw_text/corpus.txt"
    fi

    # Select appropriate fonts based on direction
    if [ "$direction" == "right_to_left" ]; then
        # Filter for Arabic/RTL fonts
        mapfile -t direction_fonts < <(ls data/fonts/ | grep -i -E "(arabic|amiri|naskh|kufi|scheherazade|noto.*arab)")
        if [ ${#direction_fonts[@]} -eq 0 ]; then
            echo "Warning: No Arabic fonts found, using all fonts"
            direction_fonts=("${fonts[@]}")
        fi
    elif [ "$direction" == "top_to_bottom" ] || [ "$direction" == "bottom_to_top" ]; then
        # Filter for CJK fonts - specific patterns only
        mapfile -t direction_fonts < <(ls data/fonts/ | grep -i -E "(cjk|^blackhan|^noto.*jp\[|^noto.*kr\[|^noto.*sc\[|^noto.*tc\[|^noto.*hans|^noto.*hant|^hiragino|^mincho|^gothic)")
        if [ ${#direction_fonts[@]} -eq 0 ]; then
            echo "Warning: No CJK fonts found, using all fonts"
            direction_fonts=("${fonts[@]}")
        fi
    else
        # For LTR, use all fonts
        direction_fonts=("${fonts[@]}")
    fi

    # Generate 10 images for the current direction
    for i in {1..10}
    do
        # Select a random font from appropriate set
        font=${direction_fonts[$RANDOM % ${#direction_fonts[@]}]}

        echo "- Generating image $i with font: $font"

        # Run the main script with the full corpus file
        python3 src/main.py \
            --text-file "$text_file" \
            --fonts-dir data/fonts \
            --output-dir output \
            --num-images 1 \
            --text-direction "$direction" \
            --font-name "$font"

        # Check if the command succeeded
        if [ $? -ne 0 ]; then
            echo "Warning: Failed to generate image $i for direction $direction"
        fi
    done
done

echo "Battery test finished. Check output directory for generated images."
