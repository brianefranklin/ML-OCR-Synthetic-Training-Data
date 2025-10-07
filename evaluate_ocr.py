
import easyocr
import os
import csv
import json
from tqdm import tqdm
import argparse
from Levenshtein import distance as levenshtein_distance

def evaluate_ocr(input_dir, output_csv, languages=['en']):
    """
    Processes all images in a given directory using EasyOCR, compares with JSON truth data,
    and saves the evaluation to a CSV file.

    Args:
        input_dir (str): The path to the directory containing images and JSON files.
        output_csv (str): The path to the output CSV file.
        languages (list): A list of language codes for OCR.
    """
    try:
        print(f"Initializing EasyOCR reader for languages: {languages}...")
        reader = easyocr.Reader(languages)
        print("EasyOCR reader initialized successfully.")
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        return

    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return
    print(f"Found {len(json_files)} JSON files to process.")

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [
            'image_name',
            'true_text',
            'easyocr_text',
            'similarity_score'
        ]
        # Dynamically add generation parameters to the header
        first_json_path = os.path.join(input_dir, json_files[0])
        with open(first_json_path, 'r') as jf:
            first_data = json.load(jf)
            param_keys = list(first_data.get('generation_params', {}).keys())
            header.extend(param_keys)
        
        csv_writer.writerow(header)

        for json_filename in tqdm(json_files, desc="Processing Files"):
            json_path = os.path.join(input_dir, json_filename)
            image_filename = json_filename.replace('.json', '.png')
            image_path = os.path.join(input_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image file not found for {json_filename}")
                continue

            with open(json_path, 'r') as jf:
                truth_data = json.load(jf)
                true_text = truth_data.get('text', '')
                gen_params = truth_data.get('generation_params', {})

            try:
                results = reader.readtext(image_path, detail=0)
                easyocr_text = " ".join(results)

                if not easyocr_text:
                    similarity_score = 0.0
                else:
                    # Calculate Levenshtein distance and normalize to a similarity score
                    dist = levenshtein_distance(true_text, easyocr_text)
                    max_len = max(len(true_text), len(easyocr_text))
                    if max_len == 0:
                        similarity_score = 1.0
                    else:
                        similarity_score = 1 - (dist / max_len)


                row = [
                    image_filename,
                    true_text,
                    easyocr_text,
                    f"{similarity_score:.4f}"
                ]
                row.extend([gen_params.get(key, '') for key in param_keys])
                csv_writer.writerow(row)

            except Exception as e:
                print(f"\nCould not process file '{image_filename}'. Error: {e}")
                row = [
                    image_filename,
                    true_text,
                    f"ERROR: {e}",
                    0.0
                ]
                row.extend([gen_params.get(key, '') for key in param_keys])
                csv_writer.writerow(row)

    print(f"\nProcessing complete. Results saved to '{output_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy against ground truth.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing image and JSON files."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="evaluation.csv",
        help="Path to the output CSV file (default: evaluation.csv)."
    )
    parser.add_argument(
        '--lang',
        nargs='+',
        default=['en'],
        help="List of languages for OCR (default: en)."
    )

    args = parser.parse_args()
    evaluate_ocr(args.input_dir, args.output_csv, args.lang)
