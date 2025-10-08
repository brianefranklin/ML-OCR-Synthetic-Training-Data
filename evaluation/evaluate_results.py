import json
import argparse
import numpy as np
import os
from collections import Counter, defaultdict
from itertools import combinations

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    'parent_key' and 'sep' are used to build the new keys.
    Example: {'a': {'b': 1}} becomes {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def analyze_parameter_correlations(results, poor_threshold=0.5, min_count=5):
    """
    Analyzes correlations between truth data parameters (including nested ones)
    and poor OCR scores.
    """
    print("\n" + "="*80)
    print(" " * 22 + "PARAMETER CORRELATION ANALYSIS")
    print("="*80)
    
    poor_results = [r for r in results if r['similarity_score_float'] < poor_threshold]

    if not poor_results:
        print("No results fell into the 'poor performance' category. No correlation analysis to run.")
        return

    # Keys to explicitly ignore in the correlation analysis
    keys_to_ignore = {
        'text', 'generation_params.text', 'image_file', 'canvas_size', 
        'text_placement', 'line_bbox', 'char_bboxes'
    }

    # --- 1. SINGLE PARAMETER ANALYSIS ---
    print("\n--- [ Single Parameter Impact on Performance ] ---\n")
    print(f"Analyzing parameters for the {len(results)} total results...")
    print(f"A parameter value is flagged if its average score is low and it appears at least {min_count} times.\n")

    param_stats = defaultdict(lambda: defaultdict(list))
    all_keys = set()
    for res in results:
        # Use the flattened data for analysis
        all_keys.update(res.get('flat_truth_data', {}).keys())
    
    # Remove keys we don't want to analyze from the set
    for key in keys_to_ignore:
        all_keys.discard(key)

    # Group scores by parameter and value using the flattened data
    for key in sorted(list(all_keys)):
        for res in results:
            if key in res.get('flat_truth_data', {}):
                value = res['flat_truth_data'][key]
                value_str = str(value) # Make booleans and other types groupable
                score = res['similarity_score_float']
                param_stats[key][value_str].append(score)

    found_single_param_issues = False
    for param, values in param_stats.items():
        significant_findings = []
        for value, scores in values.items():
            if len(scores) >= min_count:
                avg_score = np.mean(scores)
                if avg_score < poor_threshold:
                    significant_findings.append((avg_score, value, len(scores)))
        
        if significant_findings:
            found_single_param_issues = True
            print(f"[*] Parameter '{param}':")
            significant_findings.sort() # Sort by the worst average score first
            for avg_score, value, count in significant_findings:
                print(f"    - When value is '{value}', avg score is {avg_score:.3f} (from {count} examples)")
            print()

    if not found_single_param_issues:
        print("No single parameters were strongly correlated with poor performance.\n")

    # --- 2. PAIRED PARAMETER ANALYSIS ---
    print("\n--- [ Paired Parameter Analysis for Poor Performance ] ---\n")
    print(f"Finding common parameter pairs in the {len(poor_results)} worst-performing results...\n")

    pair_counts = Counter()
    for res in poor_results:
        # Create a list of (key, value) items from the flattened dict
        params = sorted(res.get('flat_truth_data', {}).items())
        
        for pair1, pair2 in combinations(params, 2):
            # Skip ignored keys
            if pair1[0] in keys_to_ignore or pair2[0] in keys_to_ignore:
                continue
            
            key = (f"{pair1[0]}: {pair1[1]}", f"{pair2[0]}: {pair2[1]}")
            pair_counts[key] += 1
    
    if not pair_counts:
        print("Could not find any parameter pairs to analyze.")
        return

    print("Most frequent parameter combinations in poor results:")
    for (pair, count) in pair_counts.most_common(10):
        if count > 1: # Only show pairs that appear more than once
             print(f"  - Occurrences: {count:3d} -> {pair[0]} AND {pair[1]}")


def analyze_ocr_results(input_file, top_n=10):
    """
    Analyzes a JSON Lines file from the OCR evaluation script and prints a
    detailed performance report.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    results = []
    error_count = 0

    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                score_str = data.get('test_data', {}).get('similarity_score', '0.0')
                if "ERROR" in data.get('test_data', {}).get('easyocr_text', ''):
                    error_count += 1
                    data['similarity_score_float'] = 0.0
                else:
                    data['similarity_score_float'] = float(score_str)
                
                # *** NEW: Flatten the truth data for detailed analysis ***
                if 'truth_data' in data:
                    data['flat_truth_data'] = flatten_dict(data['truth_data'])

                results.append(data)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Could not parse line, skipping. Error: {e}\nLine: {line.strip()}")

    if not results:
        print("No valid results were found in the file.")
        return

    scores = np.array([res['similarity_score_float'] for res in results])
    
    # --- PRINT Main REPORT ---
    print("\n" + "="*80)
    print(" " * 25 + "OCR PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    print(f"Analyzed results from: {os.path.basename(input_file)}\n")

    total_files = len(results)
    print("-" * 40)
    print("Overall Statistics")
    print("-" * 40)
    print(f"Total Files Processed: {total_files}")
    print(f"Files with Errors:     {error_count}")
    print(f"Average Similarity:    {np.mean(scores):.4f}")
    print(f"Median Similarity:     {np.median(scores):.4f}")
    print(f"Standard Deviation:    {np.std(scores):.4f}\n")

    perfect_scores = np.sum(scores >= 0.99)
    good_scores = np.sum((scores >= 0.8) & (scores < 0.99))
    medium_scores = np.sum((scores >= 0.5) & (scores < 0.8))
    poor_scores = np.sum(scores < 0.5)

    print("-" * 40)
    print("Performance Distribution")
    print("-" * 40)
    print(f"Perfect (>= 99%): {perfect_scores:5d} ({perfect_scores/total_files:7.2%})")
    print(f"Good (80% - 99%): {good_scores:5d} ({good_scores/total_files:7.2%})")
    print(f"Medium (50% - 80%): {medium_scores:5d} ({medium_scores/total_files:7.2%})")
    print(f"Poor (< 50%):     {poor_scores:5d} ({poor_scores/total_files:7.2%})\n")

    results.sort(key=lambda x: x['similarity_score_float'])
    worst_examples = results[:top_n]

    print("-" * 80)
    print(f"Top {len(worst_examples)} Worst Performing Examples")
    print("-" * 80)
    for i, ex in enumerate(worst_examples):
        score = ex['similarity_score_float']
        truth = ex['truth_data'].get('text', 'N/A')
        ocr_text = ex['test_data'].get('easyocr_text', 'N/A')
        print(f"{i+1}. Image: {ex['image_name']}")
        print(f"   Score: {score:.4f}")
        print(f"   Truth: '{truth}'")
        print(f"   OCR:   '{ocr_text}'\n")
    
    # --- RUN THE NEW ANALYSIS ---
    analyze_parameter_correlations(results)
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and summarize OCR evaluation results.")
    parser.add_argument("input_file", type=str, help="Path to the .jsonl file generated by the evaluation script.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of the worst-performing examples to display (default: 10).")
    args = parser.parse_args()
    analyze_ocr_results(args.input_file, args.top_n)

