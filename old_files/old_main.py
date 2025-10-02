import argparse
import os
import json
import random
import sys
import time
import logging
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import bidi.algorithm

# Import the new augmentation pipeline
from augmentations import apply_augmentations

def main():
    # --- Configuration Loading ---
    config = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Synthetic Data Foundry for OCR')
    parser.add_argument('--text-file', type=str, default=config.get('text_file'), help='Path to the text corpus file.')
    parser.add_argument('--fonts-dir', type=str, default=config.get('fonts_dir'), help='Path to the directory containing font files.')
    parser.add_argument('--output-dir', type=str, default=config.get('output_dir'), help='Path to the directory to save the generated images and labels.')
    parser.add_argument('--backgrounds-dir', type=str, default=config.get('backgrounds_dir'), help='Optional: Path to a directory of background images.')
    parser.add_argument('--num-images', type=int, default=config.get('num_images', 1000), help='Number of images to generate.')
    parser.add_argument('--max-execution-time', type=float, default=config.get('max_execution_time'), help='Optional: Maximum execution time in seconds.')
    parser.add_argument('--min-text-length', type=int, default=config.get('min_text_length', 1), help='Minimum length of text to generate.')
    parser.add_argument('--max-text-length', type=int, default=config.get('max_text_length', 100), help='Maximum length of text to generate.')
    parser.add_argument('--text-direction', type=str, default=config.get('text_direction', 'left_to_right'), choices=['left_to_right', 'top_to_bottom', 'right_to_left', 'bottom_to_top'], help='Direction of the text.')
    parser.add_argument('--log-level', type=str, default=config.get('log_level', 'INFO'), choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Set the logging level.')
    parser.add_argument('--log-file', type=str, default=config.get('log_file', 'generation.log'), help='Path to the log file.')
    parser.add_argument('--clear-output', action='store_true', help='If set, clears the output directory before generating new images.')
    parser.add_argument('--force', action='store_true', help='If set, bypasses the confirmation prompt when clearing the output directory.')
    parser.add_argument('--font-name', type=str, default=None, help='Name of the font file to use.')

    args = parser.parse_args()

    # --- Configure Logging ---
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=args.log_file,
                        filemode='w')
    # Add a handler to print to console as well
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, args.log_level.upper()))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("Script started.")

    # --- Clear Output Directory (if requested) ---
    if args.clear_output:
        if os.path.exists(args.output_dir):
            if not args.force:
                response = input(f"Are you sure you want to clear the output directory at {args.output_dir}? [y/N] ")
                if response.lower() != 'y':
                    logging.info("Aborting.")
                    return
            
            logging.info(f"Clearing output directory: {args.output_dir}")
            for filename in os.listdir(args.output_dir):
                file_path = os.path.join(args.output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        # You might want to handle subdirectories differently if they exist
                        # For now, this will not delete them.
                        pass
                except Exception as e:
                    logging.error(f'Failed to delete {file_path}. Reason: {e}')
        else:
            logging.info(f"Output directory {args.output_dir} does not exist. Nothing to clear.")


    # --- Validate Essential Arguments ---
    if not args.text_file:
        logging.error("Error: Text file not specified in config.json or command line.")
        sys.exit(1)
    if not args.fonts_dir or not os.path.isdir(args.fonts_dir):
        logging.error("Error: Fonts directory not specified or is not a valid directory.")
        sys.exit(1)
    if not args.output_dir:
        logging.error("Error: Output directory not specified in config.json or command line.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Load Assets ---
    logging.info("Loading assets...")
    # Load fonts
    font_files = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir) if f.endswith(('.ttf', '.otf'))]
    if not font_files:
        logging.error(f"No font files found in {args.fonts_dir}")
        sys.exit(1)
    logging.debug(f"Found {len(font_files)} font files.")

    # Load background images
    background_images = []
    if args.backgrounds_dir and os.path.exists(args.backgrounds_dir):
        background_images = [os.path.join(args.backgrounds_dir, f) for f in os.listdir(args.backgrounds_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Found {len(background_images)} background images.")

    # Load text corpus
    with open(args.text_file, 'r') as text_file:
        corpus = text_file.read()
    if not corpus:
        logging.error(f"No text found in {args.text_file}")
        sys.exit(1)
    logging.debug(f"Corpus length: {len(corpus)}")

    if args.max_text_length > len(corpus):
        args.max_text_length = len(corpus)


    # --- Generation Loop ---
    if args.num_images > 0:
        start_time = time.time()
        image_counter = 0
        labels_file = os.path.join(args.output_dir, 'labels.csv')
        with open(labels_file, 'w') as f:
            f.write('filename,text\n')

            logging.info(f"Generating up to {args.num_images} images...")
            for i in range(args.num_images):
                # --- Time Limit Check ---
                if args.max_execution_time and (time.time() - start_time) > args.max_execution_time:
                    logging.info(f"\nTime limit of {args.max_execution_time} seconds reached. Stopping generation.")
                    break

                # Select random elements
                text_line = ""
                while len(text_line) < args.min_text_length:
                    text_length = random.randint(args.min_text_length, args.max_text_length)
                    start_index = random.randint(0, len(corpus) - text_length)
                    text_line = corpus[start_index:start_index + text_length].replace('\n', ' ').strip()
                logging.debug(f"Selected text: {text_line}")
                if args.font_name:
                    font_path = os.path.join(args.fonts_dir, args.font_name)
                    if not os.path.exists(font_path):
                        logging.error(f"Error: Font file {args.font_name} not found in {args.fonts_dir}")
                        continue
                else:
                    font_path = random.choice(font_files)
                logging.debug(f"Selected font: {font_path}")

                try:
                    font = ImageFont.truetype(font_path, size=random.randint(28, 40))
                except Exception as e:
                    logging.error(f"Could not load font {font_path}: {e}")
                    continue

                # --- Create Base Image & Capture BBoxes ---
                char_bboxes = []
                x_offset = 20
                y_offset = 15

                transparent_img = Image.new('RGBA', (1, 1))
                draw = ImageDraw.Draw(transparent_img)

                if args.text_direction == 'left_to_right':
                    # Estimate image size first
                    total_text_bbox = draw.textbbox((0, 0), text_line, font=font)
                    img_width = (total_text_bbox[2] - total_text_bbox[0]) + 40
                    img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

                    image = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(image)

                    for char in text_line:
                        char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
                        draw.text((x_offset, y_offset), char, font=font, fill='black')
                        
                        # Store absolute bbox coordinates
                        char_bboxes.append(list(char_bbox))
                        
                        # Update x_offset for the next character
                        x_offset += draw.textlength(char, font=font)
                elif args.text_direction == 'right_to_left':
                    # Right-to-left-specific logic
                    right_to_left_text = bidi.algorithm.get_display(text_line)
                    total_text_bbox = draw.textbbox((0, 0), right_to_left_text, font=font)
                    img_width = (total_text_bbox[2] - total_text_bbox[0]) + 40
                    img_height = (total_text_bbox[3] - total_text_bbox[1]) + 30

                    image = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(image)

                    x_offset = img_width - 20 # Start from the right
                    for char in right_to_left_text:
                        char_width = draw.textlength(char, font=font)
                        x_offset -= char_width
                        
                        char_bbox = draw.textbbox((x_offset, y_offset), char, font=font)
                        draw.text((x_offset, y_offset), char, font=font, fill='black')
                        
                        char_bboxes.append(list(char_bbox))
                elif args.text_direction == 'bottom_to_top':
                    # Estimate image size for bottom-to-top text
                    char_widths = [draw.textbbox((0,0), char, font=font)[2] - draw.textbbox((0,0), char, font=font)[0] for char in text_line]
                    max_char_width = max(char_widths) if char_widths else 0
                    
                    _, _, _, total_height = draw.textbbox((0,0), '\n'.join(text_line), font=font)

                    img_width = max_char_width + 40
                    img_height = total_height + 30

                    image = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(image)

                    y_cursor = img_height - 15
                    for char in reversed(text_line):
                        char_width = draw.textbbox((0,0), char, font=font)[2] - draw.textbbox((0,0), char, font=font)[0]
                        char_height = draw.textbbox((0,0), char, font=font)[3] - draw.textbbox((0,0), char, font=font)[1]
                        x_cursor = (img_width - char_width) / 2
                        y_cursor -= char_height
                        draw.text((x_cursor, y_cursor), char, font=font, fill='black')
                        bbox = list(draw.textbbox((x_cursor, y_cursor), char, font=font))
                        char_bboxes.append(bbox)
                        logging.debug(f"char: {char}, bbox: {bbox}")
                    char_bboxes.reverse()

                else: # top_to_bottom
                    # Estimate image size for top-to-bottom text
                    char_widths = [draw.textbbox((0,0), char, font=font)[2] - draw.textbbox((0,0), char, font=font)[0] for char in text_line]
                    max_char_width = max(char_widths) if char_widths else 0

                    _, _, _, total_height = draw.textbbox((0,0), '\n'.join(text_line), font=font)

                    img_width = max_char_width + 40
                    img_height = total_height + 30

                    image = Image.new('RGB', (img_width, img_height), color='white')
                    draw = ImageDraw.Draw(image)

                    y_cursor = 15
                    for char in text_line:
                        char_width = draw.textbbox((0,0), char, font=font)[2] - draw.textbbox((0,0), char, font=font)[0]
                        char_height = draw.textbbox((0,0), char, font=font)[3] - draw.textbbox((0,0), char, font=font)[1]
                        x_cursor = (img_width - char_width) / 2
                        draw.text((x_cursor, y_cursor), char, font=font, fill='black')
                        bbox = list(draw.textbbox((x_cursor, y_cursor), char, font=font))
                        char_bboxes.append(bbox)
                        logging.debug(f"char: {char}, bbox: {bbox}")
                        y_cursor += char_height


                # --- Augmentation Step ---
                logging.debug("Applying augmentations...")
                augmented_image, augmented_bboxes = apply_augmentations(image, char_bboxes, background_images)

                # --- Save Image and Label ---
                image_filename = f'image_{i:05d}.png'
                image_path = os.path.join(args.output_dir, image_filename)
                augmented_image.save(image_path)
                logging.debug(f"Saved image to {image_path}")

                # Create the JSON structure for the label
                label_data = {
                    "text": text_line,
                    "bboxes": [[float(coord) for coord in bbox] for bbox in augmented_bboxes]
                }
                f.write(f'{image_filename},{json.dumps(label_data)}\n')
                image_counter += 1

        logging.info(f"Successfully generated {image_counter} images and a labels.csv file in {args.output_dir}")

if __name__ == '__main__':
    main()
    logging.info("Script finished.")
