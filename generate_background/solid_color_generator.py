# solid_color_generator.py
# Description: A Python script to generate an image with a solid color.
# Can generate images of a specific size and color, or multiple images of random sizes and colors.
#
# Usage for a specific size and color:
# python solid_color_generator.py --width 1920 --height 1080 --rgb 0 0 255 --output_dir ./backgrounds
#
# Usage for random sizes and colors:
# python solid_color_generator.py --min_width 1920 --max_width 1920 --min_height 1080 --max_height 1080 --num_images 10 --output_dir ./backgrounds
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

def generate_solid_color_image(width, height, color=None):
    """
    Generates an image filled with a single specified or random color.

    Args:
        width (int): The width of the image to generate.
        height (int): The height of the image to generate.
        color (tuple, optional): A tuple of (R, G, B) values. If not provided, a random color is generated.

    Returns:
        tuple[PIL.Image.Image, tuple[int, int, int]]: The generated image object and the RGB color used.
    """
    # Generate a single random color for the background if not provided
    background_color = color if color else generate_random_color()
    
    # Create a new image with the specified dimensions and the color
    image = Image.new('RGB', (width, height), background_color)
    
    return image, background_color

def main():
    """
    Main function to parse arguments and run the image generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate solid color background images. "
                    "You can specify a specific size and color, or generate multiple images with random sizes and/or colors.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    size_group = parser.add_argument_group('Size Specification (choose one method)')
    size_group.add_argument('--width', type=int, help="The width of the output image (for specific size).")
    size_group.add_argument('--height', type=int, help="The height of the output image (for specific size).")
    size_group.add_argument('--min_width', type=int, help="The minimum width for randomly sized images.")
    size_group.add_argument('--max_width', type=int, help="The maximum width for randomly sized images.")
    size_group.add_argument('--min_height', type=int, help="The minimum height for randomly sized images.")
    size_group.add_argument('--max_height', type=int, help="The maximum height for randomly sized images.")

    color_group = parser.add_argument_group('Color Specification')
    color_group.add_argument('--rgb', type=int, nargs=3, metavar=('R', 'G', 'B'), help="A specific RGB color tuple, e.g., --rgb 255 0 0 for red.\nIf not specified, a random color is used for each image.")

    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_dir', type=str, default='.', help="The directory to save the output image(s).")
    output_group.add_argument('--num_images', type=int, default=1, help="The number of images to generate.")
    output_group.add_argument('--logfile', type=str, default=None, help="The file to write logs to. Defaults to a timestamped log file.")
    
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
    specific_size = args.width is not None and args.height is not None
    random_size_params = [args.min_width, args.max_width, args.min_height, args.max_height]
    random_size = all(p is not None for p in random_size_params)

    if specific_size and any(p is not None for p in random_size_params):
        parser.error("Cannot mix specific size arguments (--width, --height) with random size arguments (--min_width, etc.).")
    
    if not specific_size and not random_size:
        parser.error("Either specify --width and --height for a specific size, or all of --min_width, --max_width, --min_height, --max_height for random sizes.")

    if random_size:
        if args.min_width > args.max_width:
            logging.error("min_width cannot be greater than max_width.")
            parser.error("min_width cannot be greater than max_width.")
        if args.min_height > args.max_height:
            logging.error("min_height cannot be greater than max_height.")
            parser.error("min_height cannot be greater than max_height.")

    color = None
    if args.rgb:
        r, g, b = args.rgb
        if not all(0 <= c <= 255 for c in args.rgb):
            parser.error("RGB values must be between 0 and 255.")
        color = (r, g, b)

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving images to directory: {os.path.abspath(args.output_dir)}")

    # Loop to generate the specified number of images
    for i in range(args.num_images):
        if specific_size:
            width, height = args.width, args.height
        else:  # random_size
            width = random.randint(args.min_width, args.max_width)
            height = random.randint(args.min_height, args.max_height)
        
        logging.info(f"Generating image {i+1}/{args.num_images} with dimensions: {width}x{height}")

        # Generate the image
        generated_image, used_color = generate_solid_color_image(width, height, color=color)

        # Save the image to a file with a unique name
        color_str = f"{used_color[0]}-{used_color[1]}-{used_color[2]}"
        output_filename = f"solid_color_{width}x{height}_{color_str}_{i+1}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        generated_image.save(save_path)
        logging.info(f"Successfully created image: {save_path}")


if __name__ == '__main__':
    main()
