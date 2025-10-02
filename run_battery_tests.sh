#!/bin/bash

# This script runs a battery of tests for the synthetic data generation script.
# It generates images with different text directions and fonts.

# Clear the output directory once at the beginning
python3 src/main.py --output-dir output --clear-output --force --num-images 0

# Define the test combinations
declare -a combinations=(
    "left_to_right:AlegreyaSans-BlackItalic.ttf:5"
    "left_to_right:Lato-Regular.ttf:10"
    "right_to_left:NotoSansArabic[wdth,wght].ttf:15"
    "right_to_left:IBMPlexSansArabic-Regular.ttf:20"
    "top_to_bottom:NotoSerifCJKjp-Regular.otf:5"
    "top_to_bottom:NanumGothic.ttf:10"
    "top_to_bottom:NotoSansCJKtc-VF.otf:15"
    "bottom_to_top:NotoSerifCJKjp-Regular.otf:5"
    "left_to_right::20"
    "right_to_left::5"
    "top_to_bottom::10"
    "bottom_to_top::10"
)

# Loop through the combinations and run the script
for combo in "${combinations[@]}"
do
    IFS=':' read -r -a params <<< "$combo"
    direction="${params[0]}"
    font="${params[1]}"
    num_images="${params[2]}"

    echo "Running with text direction: $direction, font: $font, num_images: $num_images"

    command="python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images $num_images --text-direction $direction"

    if [ -n "$font" ]; then
        command="$command --font-name '$font'"
    fi

    eval $command
done

echo "Battery test finished. Check the output directory for the generated images."