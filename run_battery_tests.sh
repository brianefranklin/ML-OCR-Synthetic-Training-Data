#!/bin/bash

# Combination 1
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 5 --text-direction left_to_right --font-name AlegreyaSans-BlackItalic.ttf --clear-output --force

# Combination 2
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 10 --text-direction left_to_right --font-name Lato-Regular.ttf --clear-output --force

# Combination 3
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 15 --text-direction right_to_left --font-name 'NotoSansArabic[wdth,wght].ttf' --clear-output --force

# Combination 4
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 20 --text-direction right_to_left --font-name IBMPlexSansArabic-Regular.ttf --clear-output --force

# Combination 5
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 5 --text-direction top_to_bottom --font-name NotoSerifCJKjp-Regular.otf --clear-output --force

# Combination 6
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 10 --text-direction top_to_bottom --font-name NanumGothic.ttf --clear-output --force

# Combination 7
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 15 --text-direction top_to_bottom --font-name NotoSansCJKtc-VF.otf --clear-output --force

# Combination 8
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 20 --text-direction left_to_right --clear-output --force

# Combination 9
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 5 --text-direction right_to_left --clear-output --force

# Combination 10
python3 src/main.py --text-file requirements.txt --fonts-dir data/fonts --output-dir output --num-images 10 --text-direction top_to_bottom --clear-output --force
