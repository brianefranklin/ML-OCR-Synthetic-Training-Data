import easyocr
import os
import json
from tqdm import tqdm
import argparse
from Levenshtein import distance as levenshtein_distance
import multiprocessing
import torch
import datetime
import warnings # Import the warnings module
import random

# This will be a global variable within each worker process.
reader = None

def init_worker(languages, use_gpu):
    """
    Initializer for each worker process in the pool.
    Loads the EasyOCR model into the global 'reader' variable for that specific process.
    """
    global reader
    # Suppress the specific UserWarning from PyTorch's DataLoader about 'pin_memory'.
    # This warning is not relevant when running on CPU and just clutters the output.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*'pin_memory' argument is set as true*"
    )
    # Each worker loads the model from the cache into its own memory.
    reader = easyocr.Reader(languages, gpu=use_gpu)

def process_file(args):
    """
    Processes a single file using the reader instance specific to its worker process.
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
        # The 'reader' global is guaranteed to be initialized in this worker process.
        results = reader.readtext(image_path, detail=1)
        
        # Extract text and format raw results
        ocr_text_parts = []
        formatted_results = []
        for (bbox, text, conf) in results:
            ocr_text_parts.append(text)
            formatted_results.append({
                'box': [[int(coord) for coord in point] for point in bbox],
                'text': text,
                'confidence': float(conf)
            })
        
        ocr_text = " ".join(ocr_text_parts)

        dist = levenshtein_distance(true_text, ocr_text)
        max_len = max(len(true_text), len(ocr_text))
        if max_len == 0:
            # Handle case where both strings are empty
            similarity_score = 1.0
        else:
            similarity_score = 1 - (dist / max_len)

        return {
            'image_name': image_filename,
            'truth_data': truth_data_content,
            'test_data': {
                'model_name': 'EasyOCR',
                'ocr_text': ocr_text,
                'similarity_score': f"{similarity_score:.4f}",
                'raw_ocr_results': formatted_results
            }
        }

    except Exception as e:
        return {
            'image_name': image_filename,
            'truth_data': truth_data_content,
            'test_data': {
                'model_name': 'EasyOCR',
                'ocr_text': f"ERROR: {e}",
                'similarity_score': 0.0,
                'raw_ocr_results': []
            }
        }

def evaluate_ocr(input_dir, output_path, languages=['en'], use_gpu=False, workers=None, max_images=None, randomize=False):
    """
    Processes images using EasyOCR, compares with truth data, and saves the evaluation
    to a JSON Lines file.
    """
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if randomize:
        print("NOTE: Randomizing the order of input files.")
        random.shuffle(json_files)

    # For testing: limit the maximum number of images
    if max_images is not None:
        print(f"NOTE: Limiting run to a maximum of {max_images} images.")
        json_files = json_files[:max_images]

    if not json_files:
        print(f"No JSON files found in '{input_dir}'.")
        return
    print(f"Found {len(json_files)} JSON files to process.")

    if workers is None:
        workers = multiprocessing.cpu_count()
        
    print("\n" + "="*60)
    print("SETUP PHASE: PREPARING FOR OCR PROCESSING")
    print("Each worker process will now load the OCR model into memory.")
    print("="*60)
    
    pool_args = [(json_filename, input_dir) for json_filename in json_files]

    with open(output_path, 'w', encoding='utf-8') as f:
        # Use an initializer to ensure each worker has its own reader instance.
        initargs = (languages, use_gpu)
        with multiprocessing.Pool(processes=workers, initializer=init_worker, initargs=initargs) as pool:
            
            print(f"\nPARALLEL PROCESSING PHASE: Starting with {workers} worker(s).")
            print("="*60)
            
            for result in tqdm(pool.imap_unordered(process_file, pool_args), total=len(json_files), desc="Processing Files"):
                if result:
                    f.write(json.dumps(result) + '\n')

    print(f"\nProcessing complete. Results saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy against ground truth.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing image and JSON files.")
    
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output_file", type=str, help="Path to a specific output JSON Lines file.")
    output_group.add_argument("--output_dir", type=str, help="Path to an output directory for a timestamped log file.")

    parser.add_argument('--lang', nargs='+', default=['en'], help="List of languages for OCR (default: en).")
    parser.add_argument('--gpu', action='store_true', help="Enable GPU for OCR processing.")
    parser.add_argument('--workers', type=int, default=None, help="Number of worker processes to use (default: all available cores).")
    # New argument for limiting images for testing
    parser.add_argument('--max_images', type=int, default=None, help="For testing: limit the maximum number of images to process.")
    parser.add_argument('--randomize', action='store_true', help="Randomize the order of the input files.")
    
    args = parser.parse_args()

    output_path = ""
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_evaluation_{timestamp}.jsonl"
        output_path = os.path.join(args.output_dir, filename)
    elif args.output_file:
        output_path = args.output_file
    else:
        # Default to a timestamped file in the current directory if no output is specified.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ocr_evaluation_{timestamp}.jsonl"
        output_path = filename
    
    use_gpu_flag = args.gpu and torch.cuda.is_available()
    if args.gpu and not use_gpu_flag:
        print("GPU was requested, but is not available. Falling back to CPU.")

    evaluate_ocr(args.input_dir, output_path, args.lang, use_gpu_flag, args.workers, args.max_images, args.randomize)
