# solid_color_generator.py
# Description: A Python script to generate an image with a solid, random color.
#
# Usage:
# python solid_color_generator.py --min_width 1920 --max_width 1920 --min_height 1080 --max_height 1080 --num_images 10 --output_dir ./backgrounds
#
# This will create 10 images, each 1920x1080, in a directory named 'backgrounds'.
#
# Requirements:
# You need to install the Pillow library to run this script.
# pip install Pillow

import random
import argparse
import os
import logging
from datetime import datetime
from PIL import Image

def generate_random_color():
    """Generates a random RGB color tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_solid_color_image(width, height):
    """
    Generates an image filled with a single random color.

    Args:
        width (int): The width of the image to generate.
        height (int): The height of the image to generate.

    Returns:
        PIL.Image.Image: The generated image object.
    """
    # Generate a single random color for the background
    background_color = generate_random_color()
    
    # Create a new image with the specified dimensions and the random color
    image = Image.new('RGB', (width, height), background_color)
    
    return image

def main():
    """
    Main function to parse arguments and run the image generation.
    """
    parser = argparse.ArgumentParser(description="Generate solid color background images.")
    parser.add_argument('--min_width', type=int, required=True, help="The minimum width of the output image.")
    parser.add_argument('--max_width', type=int, required=True, help="The maximum width of the output image.")
    parser.add_argument('--min_height', type=int, required=True, help="The minimum height of the output image.")
    parser.add_argument('--max_height', type=int, required=True, help="The maximum height of the output image.")
    parser.add_argument('--output_dir', type=str, default='.', help="The directory to save the output image(s).")
    parser.add_argument('--num_images', type=int, default=1, help="The number of images to generate.")
    parser.add_argument('--logfile', type=str, default=None, help="The file to write logs to. Defaults to a timestamped log file.")
    args = parser.parse_args()

    # --- Setup Logging ---
    log_filename = args.logfile
    if log_filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"solid_color_generator_{timestamp}.log"
    
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # --- Validate Arguments ---
    if args.min_width > args.max_width:
        logging.error("min_width cannot be greater than max_width.")
        return # Exit if validation fails
    if args.min_height > args.max_height:
        logging.error("min_height cannot be greater than max_height.")
        return # Exit if validation fails

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving images to directory: {os.path.abspath(args.output_dir)}")

    # Loop to generate the specified number of images
    for i in range(args.num_images):
        # Determine random dimensions for this image
        width = random.randint(args.min_width, args.max_width)
        height = random.randint(args.min_height, args.max_height)
        logging.info(f"Generating image {i+1}/{args.num_images} with dimensions: {width}x{height}")

        # Generate the image
        generated_image = generate_solid_color_image(width, height)

        # Save the image to a file with a unique name
        output_filename = f"solid_color_{width}x{height}_{i+1}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        generated_image.save(save_path)
        logging.info(f"Successfully created image: {save_path}")


if __name__ == '__main__':
    main()