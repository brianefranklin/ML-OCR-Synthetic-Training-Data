import easyocr
import os
import json
from tqdm import tqdm
import argparse
from Levenshtein import distance as levenshtein_distance
import multiprocessing
import torch # Import torch to check for GPU

# Global reader for worker processes, initialized in init_worker
reader = None

def init_worker(languages, use_gpu):
    """Initializer for worker processes, creates a global reader instance."""
    global reader
    # Each worker process creates its own reader.
    # The models should already be downloaded and cached by the main process,
    # so this will be a fast operation.
    reader = easyocr.Reader(languages, gpu=use_gpu)

def process_file(args):
    """
    Processes a single file using the global reader instance.
    Args:
        args (tuple): A tuple containing (json_filename, input_dir).
    """
    json_filename, input_dir = args
    json_path = os.path.join(input_dir, json_filename)
    image_filename = json_filename.replace('.json', '.png')
    image_path = os.path.join(input_dir, image_filename)

    if not os.path.exists(image_path):
        return None

    with open(json_path, 'r') as jf:
        truth_data_content = json.load(jf)
    
    true_text = truth_data_content.get('text', '')

    try:
        # The 'reader' is global within this worker process
        results = reader.readtext(image_path, detail=0)
        easyocr_text = " ".join(results)

        if not easyocr_text:
            similarity_score = 0.0
        else:
            dist = levenshtein_distance(true_text, easyocr_text)
            max_len = max(len(true_text), len(easyocr_text))
            if max_len == 0:
                similarity_score = 1.0
            else:
                similarity_score = 1 - (dist / max_len)

        return {
            'image_name': image_filename,
            'truth_data': truth_data_content,
            'test_data': {
                'easyocr_text': easyocr_text,
                'similarity_score': f"{similarity_score:.4f}",
                'raw_easyocr_results': results
            }
        }

    except Exception as e:
        return {
            'image_name': image_filename,
            'truth_data': truth_data_content,
            'test_data': {
                'easyocr_text': f"ERROR: {e}",
                'similarity_score': 0.0,
                'raw_easyocr_results': []
            }
        }

def evaluate_ocr(input_dir, output_file, languages=['en'], use_gpu=False, workers=None):
    """
    Processes images using EasyOCR, compares with truth data, and saves the evaluation
    to a JSON Lines file.
    """
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return
    print(f"Found {len(json_files)} JSON files to process.")

    # ===== FIX STARTS HERE =====
    # 1. Trigger model download in the main process BEFORE starting the pool.
    # This avoids the race condition where multiple workers try to download at once.
    print("Initializing EasyOCR and downloading models if necessary...")
    _ = easyocr.Reader(languages, gpu=use_gpu)
    print("Initialization complete.")
    # ===== FIX ENDS HERE =====


    if workers is None:
        workers = multiprocessing.cpu_count()
    
    pool_args = [(json_filename, input_dir) for json_filename in json_files]

    with open(output_file, 'w', encoding='utf-8') as f:
        # Create the pool ONCE outside the loop.
        with multiprocessing.Pool(processes=workers, initializer=init_worker, initargs=(languages, use_gpu)) as pool:
            # Use tqdm to create a progress bar for the results from the pool
            for result in tqdm(pool.imap_unordered(process_file, pool_args), total=len(json_files), desc="Processing Files"):
                if result:
                    f.write(json.dumps(result) + '\n')

    print(f"\nProcessing complete. Results saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy against ground truth.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to the directory containing image and JSON files."
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation.jsonl",
        help="Path to the output JSON Lines file (default: evaluation.jsonl)."
    )
    parser.add_argument(
        '--lang', nargs='+', default=['en'],
        help="List of languages for OCR (default: en)."
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help="Enable GPU for OCR processing."
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Number of worker processes to use (default: all available cores)."
    )
    
    args = parser.parse_args()
    
    # Check if GPU is requested and available
    use_gpu_flag = args.gpu and torch.cuda.is_available()
    if args.gpu and not use_gpu_flag:
        print("GPU was requested, but is not available. Falling back to CPU.")

    # The chunk_size argument has been removed as it's no longer needed with the fix.
    evaluate_ocr(args.input_dir, args.output_file, args.lang, use_gpu_flag, args.workers)
