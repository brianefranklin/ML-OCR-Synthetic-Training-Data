
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.main import OCRDataGenerator
from PIL import ImageFont, Image
import logging

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='font_test.log',
    filemode='w'
)


def find_bad_font():
    font_dir = "/home/vscode/workspace/data/fonts/"
    font_files = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf"):
                font_files.append(os.path.join(root, file))

    generator = OCRDataGenerator(font_files=[])
    text = "This is a test string"

    for font_path in font_files:
        logging.info(f"Testing font: {font_path}")
        try:
            font = generator.load_font(font_path, 30)
            image, char_boxes = generator.render_bottom_to_top(
                text=text,
                font=font,
                overlap_intensity=0.8,
                ink_bleed_intensity=0.0
            )
        except Exception as e:
            logging.error(f"Error with font {font_path}: {e}")
            print(f"Error with font {font_path}: {e}")

if __name__ == "__main__":
    find_bad_font()
