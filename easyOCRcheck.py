import easyocr
import os
import csv
import argparse
from tqdm import tqdm

# Requires easyocr and tqdm  

def process_images_in_directory(input_dir, output_csv, languages=['en']):
    """
    Processes all images in a given directory using EasyOCR and saves the
    detected text to a CSV file.

    Args:
        input_dir (str): The path to the directory containing images.
        output_csv (str): The path to the output CSV file.
        languages (list): A list of language codes for OCR (e.g., ['en', 'es']).
    """
    # --- 1. Initialize the EasyOCR Reader ---
    # This will download the model for the specified languages on the first run.
    # To use a GPU, set gpu=True, e.g., easyocr.Reader(languages, gpu=True)
    try:
        print(f"Initializing EasyOCR reader for languages: {languages}...")
        reader = easyocr.Reader(languages)
        print("EasyOCR reader initialized successfully.")
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        print("Please ensure you have a working PyTorch installation.")
        print("You can install the CPU version with: pip install torch torchvision torchaudio")
        return

    # --- 2. Find all valid image files in the directory ---
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    try:
        image_files = [f for f in os.listdir(input_dir)
                       if os.path.splitext(f)[1].lower() in supported_extensions]
        if not image_files:
            print(f"No supported image files found in '{input_dir}'.")
            return
        print(f"Found {len(image_files)} images to process.")
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # --- 3. Open the CSV file for writing ---
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            # Define CSV header
            csv_writer = csv.writer(csvfile)
            header = [
                'filename',
                'text',
                'confidence',
                'top_left_x', 'top_left_y',
                'bottom_right_x', 'bottom_right_y'
            ]
            csv_writer.writerow(header)

            # --- 4. Process each image and write results ---
            # Using tqdm for a progress bar
            for filename in tqdm(image_files, desc="Processing Images"):
                image_path = os.path.join(input_dir, filename)

                try:
                    # Perform OCR on the image
                    # detail=1 provides full details including coordinates and confidence
                    results = reader.readtext(image_path, detail=1)

                    if not results:
                        # Write a row even if no text is found
                        csv_writer.writerow([filename, 'NO_TEXT_FOUND', 0.0, 0, 0, 0, 0])
                        continue

                    # Each 'result' is a tuple: (bounding_box, text, confidence)
                    for (bbox, text, confidence) in results:
                        # Extract coordinates from the bounding box
                        # bbox is [[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]]
                        top_left = bbox[0]
                        bottom_right = bbox[2]

                        row = [
                            filename,
                            text,
                            f"{confidence:.4f}", # Format confidence to 4 decimal places
                            int(top_left[0]),
                            int(top_left[1]),
                            int(bottom_right[0]),
                            int(bottom_right[1])
                        ]
                        csv_writer.writerow(row)

                except Exception as e:
                    print(f"\nCould not process file '{filename}'. Error: {e}")
                    # Write an error row to the CSV
                    csv_writer.writerow([filename, f"ERROR: {e}", 0.0, 0, 0, 0, 0])

        print(f"\nProcessing complete. Results saved to '{output_csv}'.")

    except IOError as e:
        print(f"Error writing to CSV file '{output_csv}'. Error: {e}")


if __name__ == "__main__":
    # --- 5. Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Batch OCR processing for a directory of images.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing image files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ocr_results.csv",
        help="Path to the output CSV file (default: ocr_results.csv)."
    )
    parser.add_argument(
        '--lang',
        nargs='+',
        default=['en'],
        help="List of languages for OCR, e.g., --lang en es fr (default: en)."
    )

    args = parser.parse_args()

    # Run the main function
    process_images_in_directory(args.input_dir, args.output_csv, args.lang)
