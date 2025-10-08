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

def create_text_boxplot(scores, width=50):
    """
    Creates an enhanced, colorized, text-based box-and-whisker plot.
    """
    if len(scores) == 0:
        return ""

    # ANSI color codes for a more readable plot
    class Colors:
        RESET = '\033[0m'
        BLUE = '\033[94m'   # For the box
        GREEN = '\033[92m'  # For the median
        YELLOW = '\033[93m' # For the whiskers
        CYAN = '\033[96m'   # For labels and axis

    min_val, q1_val, median_val, q3_val, max_val = np.percentile(scores, [0, 25, 50, 75, 100])

    # Scale the stat values to the plot width (scores are 0-1)
    pos_min = int(min_val * width)
    pos_q1 = int(q1_val * width)
    pos_median = int(median_val * width)
    pos_q3 = int(q3_val * width)
    pos_max = int(max_val * width)

    # Build the plot line by drawing components in order of precedence
    plot_line = [' '] * (width + 1)
    
    # 1. Draw whisker
    for i in range(pos_min, pos_max + 1):
        plot_line[i] = Colors.YELLOW + '-' + Colors.RESET
        
    # 2. Draw box over the whisker
    for i in range(pos_q1, pos_q3 + 1):
        plot_line[i] = Colors.BLUE + '█' + Colors.RESET
        
    # 3. Draw median and min/max markers over the box/whisker
    plot_line[pos_median] = Colors.GREEN + '|' + Colors.RESET
    plot_line[pos_min] = Colors.YELLOW + '>' + Colors.RESET
    plot_line[pos_max] = Colors.YELLOW + '<' + Colors.RESET

    plot_str = "".join(plot_line)
    
    # Create colorized axis and labels for context
    axis_line = Colors.CYAN + '+' + '-' * width + '+' + Colors.RESET
    labels_line = Colors.CYAN + '0.0' + ' ' * (width // 2 - 2) + '0.5' + ' ' * (width - (width // 2) - 4) + '1.0' + Colors.RESET

    # Assemble the final multi-line string for printing
    output = [
        "Score Distribution Box Plot:",
        f"Min: {min_val:.3f}, Q1: {q1_val:.3f}, Median: {median_val:.3f}, Q3: {q3_val:.3f}, Max: {max_val:.3f}",
        labels_line,
        axis_line,
        plot_str,
        axis_line
    ]
    return "\n".join(output)

def analyze_parameter_correlations(results, poor_threshold=0.5, min_count=5, max_correlation_depth=3):
    """
    Analyzes correlations between truth data parameters (including nested ones)
    and poor OCR scores, for combinations of N parameters.
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
        all_keys.update(res.get('flat_truth_data', {}).keys())
    
    for key in keys_to_ignore:
        all_keys.discard(key)

    for key in sorted(list(all_keys)):
        for res in results:
            if key in res.get('flat_truth_data', {}):
                value = res['flat_truth_data'][key]
                value_str = str(value)
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
            significant_findings.sort()
            for avg_score, value, count in significant_findings:
                print(f"    - When value is '{value}', avg score is {avg_score:.3f} (from {count} examples)")
            print()

    if not found_single_param_issues:
        print("No single parameters were strongly correlated with poor performance.\n")

    # --- 2. MULTI-PARAMETER COMBINATION ANALYSIS ---
    # This loop will handle pairs (n=2), triplets (n=3), and so on.
    for n in range(2, max_correlation_depth + 1):
        level_name = {2: "Paired", 3: "Triple", 4: "Quadruple"}.get(n, f"{n}-Parameter")
        print(f"\n--- [ {level_name} Parameter Analysis for Poor Performance ] ---\n")
        print(f"Finding common {n}-parameter combinations in the {len(poor_results)} worst-performing results...\n")

        combo_counts = Counter()
        for res in poor_results:
            params = sorted(res.get('flat_truth_data', {}).items())
            
            # Generate all combinations of size 'n' from the parameters
            for combo in combinations(params, n):
                # Filter out any combinations that include an ignored key
                if any(p[0] in keys_to_ignore for p in combo):
                    continue
                
                # Create a canonical key for the combination
                key = tuple(f"{p[0]}: {p[1]}" for p in combo)
                combo_counts[key] += 1
        
        if not combo_counts:
            print(f"Could not find any {n}-parameter combinations to analyze.")
            continue

        print(f"Most frequent {n}-parameter combinations in poor results:")
        found_combos = False
        for combo, count in combo_counts.most_common(10):
            if count > 1:
                found_combos = True
                details = " AND ".join(combo)
                print(f"  - Occurrences: {count:3d} -> {details}")
        
        if not found_combos:
            print("No combinations occurred frequently enough to report.")


def analyze_ocr_results(input_file, top_n=10, max_corr=3):
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

    if total_files > 0:
        print(create_text_boxplot(scores))
        print("\n")

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
    analyze_parameter_correlations(results, max_correlation_depth=max_corr)
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and summarize OCR evaluation results.")
    parser.add_argument("input_file", type=str, help="Path to the .jsonl file generated by the evaluation script.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of the worst-performing examples to display (default: 10).")
    parser.add_argument(
        "--max_corr",
        type=int,
        default=3,
        help="The maximum number of parameters to check in combination for correlation (default: 3)."
    )
    args = parser.parse_args()
    analyze_ocr_results(args.input_file, args.top_n, args.max_corr)

