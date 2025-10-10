# generative_art.py
# Description: A Python script to generate an image with random stripes and a geometric shape.
#
# Usage:
# python generative_art.py --width 800 --height 600
#
# This will create an 800x600 image named 'random_geometric_art.png'.
#
# Requirements:
# You need to install the Pillow library to run this script.
# pip install Pillow

import random
import argparse
import os
import math
import logging
from datetime import datetime
from PIL import Image, ImageDraw

def generate_random_color():
    """Generates a random RGB color tuple."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def generate_art(width, height):
    """
    Generates an image with a random striped background and a random geometric shape.

    Args:
        width (int): The width of the image to generate.
        height (int): The height of the image to generate.

    Returns:
        PIL.Image.Image: The generated image object.
    """
    # Create a new image with a white background
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # --- 1. Draw random stripes at a random angle ---
    # To do this, we create a larger temporary canvas, draw horizontal stripes on it,
    # rotate it, and then paste it onto our main image.

    # Calculate the diagonal of the image, this will be the size of our temp canvas
    diag = int(math.sqrt(width**2 + height**2))
    
    # Create the temporary canvas and a drawing context for it
    stripe_canvas = Image.new('RGB', (diag, diag), 'white')
    stripe_draw = ImageDraw.Draw(stripe_canvas)

    # Use a variable stripe height for more randomness
    y_pos = 0
    while y_pos < diag:
        stripe_color = generate_random_color()
        # Determine a random height for the current stripe
        stripe_height = random.randint(10, 50)
        # Define the bounding box for the stripe rectangle on the temp canvas
        box = (0, y_pos, diag, y_pos + stripe_height)
        stripe_draw.rectangle(box, fill=stripe_color)
        y_pos += stripe_height

    # Rotate the temporary canvas by a random angle
    angle = random.randint(0, 180)
    rotated_stripes = stripe_canvas.rotate(angle, resample=Image.BICUBIC, expand=False)

    # Calculate the coordinates to paste the rotated image so it's centered
    paste_x = (width - diag) // 2
    paste_y = (height - diag) // 2
    
    # Paste the rotated stripes onto the main image
    image.paste(rotated_stripes, (paste_x, paste_y))

    # --- 2. Draw between 1 and 5 random geometric shapes on top ---
    num_shapes = random.randint(1, 5)
    logging.info(f"Drawing {num_shapes} shapes on the canvas.")

    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'ellipse', 'triangle'])
        shape_color = generate_random_color()
        outline_color = generate_random_color() # Use a different color for the outline

        # Define a maximum size for the shape to ensure it's not overly large
        max_shape_width = width // 2
        max_shape_height = height // 2
        
        # Calculate random coordinates for the shape, ensuring it stays within bounds
        shape_x1 = random.randint(0, width - max_shape_width)
        shape_y1 = random.randint(0, height - max_shape_height)
        shape_x2 = shape_x1 + random.randint(50, max_shape_width)
        shape_y2 = shape_y1 + random.randint(50, max_shape_height)

        logging.info(f"  -> Drawing a {shape_type}.")

        if shape_type == 'rectangle':
            draw.rectangle(
                (shape_x1, shape_y1, shape_x2, shape_y2), 
                fill=shape_color, 
                outline=outline_color,
                width=5 # outline width
            )
        elif shape_type == 'ellipse':
            draw.ellipse(
                (shape_x1, shape_y1, shape_x2, shape_y2), 
                fill=shape_color, 
                outline=outline_color,
                width=5
            )
        elif shape_type == 'triangle':
            # For a triangle, we define three points
            point1 = (random.randint(shape_x1, shape_x2), random.randint(shape_y1, shape_y2))
            point2 = (random.randint(shape_x1, shape_x2), random.randint(shape_y1, shape_y2))
            point3 = (random.randint(shape_x1, shape_x2), random.randint(shape_y1, shape_y2))
            draw.polygon(
                [point1, point2, point3], 
                fill=shape_color, 
                outline=outline_color
            )
            # Note: outline width is not directly supported for polygons in the same way,
            # so we draw lines over it for a similar effect.
            draw.line([point1, point2, point3, point1], fill=outline_color, width=5)

    return image

def main():
    """
    Main function to parse arguments and run the image generation.
    """
    parser = argparse.ArgumentParser(description="Generate an image with random stripes and shapes.")
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
        log_filename = f"art_generator_{timestamp}.log"
    
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

    # Loop to generate the specified number of images
    for i in range(args.num_images):
        # Determine random dimensions for this image
        width = random.randint(args.min_width, args.max_width)
        height = random.randint(args.min_height, args.max_height)
        logging.info(f"Generating image {i+1}/{args.num_images} with dimensions: {width}x{height}")

        # Generate the image
        generated_image = generate_art(width, height)

        # Save the image to a file with a unique name
        output_filename = f"random_geometric_art_{i+1}.png"
        save_path = os.path.join(args.output_dir, output_filename)
        generated_image.save(save_path)
        logging.info(f"Successfully created image: {save_path}")


if __name__ == '__main__':
    main()

