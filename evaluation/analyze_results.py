import json
import argparse
import numpy as np
import os
from collections import Counter, defaultdict
from itertools import combinations

# ANSI color codes for a more readable report, defined globally
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'

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

def make_value_hashable(value):
    """
    Recursively converts lists within a value to tuples to make them hashable.
    """
    if isinstance(value, list):
        return tuple(make_value_hashable(v) for v in value)
    return value

def create_text_boxplot(scores, width=50):
    """
    Creates an enhanced, colorized, text-based box-and-whisker plot.
    """
    if len(scores) == 0:
        return ""

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
        f"{Colors.BOLD}Score Distribution Box Plot:{Colors.RESET}",
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
    header_color = Colors.BOLD + Colors.MAGENTA
    subheader_color = Colors.CYAN
    highlight_color = Colors.YELLOW

    print("\n" + header_color + "="*80 + Colors.RESET)
    print(header_color + " " * 22 + "PARAMETER CORRELATION ANALYSIS" + Colors.RESET)
    print(header_color + "="*80 + Colors.RESET)
    
    # --- PRE-ANALYSIS: Find constant parameters and separate numerical/categorical ---
    param_values = defaultdict(set)
    for res in results:
        for key, value in res.get('flat_truth_data', {}).items():
            hashable_value = make_value_hashable(value)
            param_values[key].add(hashable_value)

    constant_keys = {key for key, values in param_values.items() if len(values) == 1}

    explicit_keys_to_ignore = {
        'text', 'generation_params.text', 'image_file', 'canvas_size', 
        'text_placement', 'line_bbox', 'char_bboxes'
    }
    keys_to_ignore = constant_keys.union(explicit_keys_to_ignore)

    if constant_keys:
        print(f"\n{subheader_color}--- [ Constant Parameter Detection ] ---{Colors.RESET}\n")
        print("The following parameters have the same value across all results and will be excluded from correlation analysis:\n")
        for key in sorted(list(constant_keys)):
            if key not in explicit_keys_to_ignore:
                value = next(iter(param_values[key]))
                print(f"  - {highlight_color}'{key}'{Colors.RESET}: (always '{value}')")
        print()
    
    poor_results = [r for r in results if r['similarity_score_float'] < poor_threshold]
    if not poor_results:
        print("No results fell into the 'poor performance' category. No correlation analysis to run.")
        return

    all_keys = {k for res in results for k in res.get('flat_truth_data', {})}
    variable_keys = sorted([k for k in all_keys if k not in keys_to_ignore])

    categorical_params_stats = defaultdict(lambda: defaultdict(list))
    numerical_params_values = {}
    for key in variable_keys:
        values = [res['flat_truth_data'][key] for res in results if key in res.get('flat_truth_data', {})]
        if all(isinstance(v, (int, float)) for v in values):
            numerical_params_values[key] = values
        else:
            for res in results:
                if key in res.get('flat_truth_data', {}):
                    value_str = str(res['flat_truth_data'][key])
                    score = res['similarity_score_float']
                    categorical_params_stats[key][value_str].append(score)

    # --- 1. CATEGORICAL PARAMETER ANALYSIS ---
    print(f"\n{subheader_color}--- [ Categorical Parameter Impact on Performance ] ---{Colors.RESET}\n")
    found_single_param_issues = False
    for param, values in categorical_params_stats.items():
        significant_findings = []
        for value, scores in values.items():
            if len(scores) >= min_count and np.mean(scores) < poor_threshold:
                significant_findings.append((np.mean(scores), value, len(scores)))
        if significant_findings:
            found_single_param_issues = True
            print(f"[*] Parameter '{highlight_color}{param}{Colors.RESET}':")
            significant_findings.sort()
            for avg_score, value, count in significant_findings:
                print(f"    - When value is '{value}', avg score is {Colors.RED}{avg_score:.3f}{Colors.RESET} (from {count} examples)")
            print()
    if not found_single_param_issues:
        print("No categorical parameters were strongly correlated with poor performance.\n")

    # --- 2. NUMERICAL PARAMETER RANGE ANALYSIS ---
    print(f"\n{subheader_color}--- [ Numerical Parameter Range Impact ] ---{Colors.RESET}\n")
    numerical_thresholds = {}
    found_numerical_issues = False
    skipped_low_variance_stats = {}
    for param, values in numerical_params_values.items():
        if len(set(values)) < 4 or len(values) < min_count * 2: continue
        
        q1_thresh, q3_thresh = np.percentile(values, [25, 75])
        
        # If Q1 and Q3 are the same, the data is too concentrated for this analysis.
        if q1_thresh == q3_thresh:
            value_counts = Counter(values)
            most_common_val, most_common_count = value_counts.most_common(1)[0]
            stats = {
                "unique_count": len(set(values)),
                "most_common_val": most_common_val,
                "most_common_count": most_common_count,
                "most_common_pct": (most_common_count / len(values)) * 100,
                "mean": np.mean(values),
                "min": np.min(values),
                "max": np.max(values)
            }
            skipped_low_variance_stats[param] = stats
            continue
            
        numerical_thresholds[param] = {'q1': q1_thresh, 'q3': q3_thresh}
        
        low_scores = [r['similarity_score_float'] for r in results if r.get('flat_truth_data', {}).get(param, q1_thresh + 1) <= q1_thresh]
        high_scores = [r['similarity_score_float'] for r in results if r.get('flat_truth_data', {}).get(param, q3_thresh - 1) >= q3_thresh]
        
        significant_findings = []
        if len(low_scores) >= min_count and np.mean(low_scores) < poor_threshold:
            found_numerical_issues = True
            significant_findings.append(f"    - When value <= {q1_thresh:.3f} (bottom 25%), avg score is {Colors.RED}{np.mean(low_scores):.3f}{Colors.RESET} (from {len(low_scores)} examples)")
        if len(high_scores) >= min_count and np.mean(high_scores) < poor_threshold:
            found_numerical_issues = True
            significant_findings.append(f"    - When value >= {q3_thresh:.3f} (top 25%), avg score is {Colors.RED}{np.mean(high_scores):.3f}{Colors.RESET} (from {len(high_scores)} examples)")
        
        if significant_findings:
            print(f"[*] Parameter '{highlight_color}{param}{Colors.RESET}':")
            for finding in significant_findings: print(finding)
            print()

    if not found_numerical_issues:
        print("No numerical parameter ranges were strongly correlated with poor performance.\n")

    if skipped_low_variance_stats:
        print(f"{subheader_color}--- [ Low Variance Numerical Parameters ] ---{Colors.RESET}\n")
        print("The following numerical parameters were skipped from range analysis because their values are heavily concentrated, making quartile analysis uninformative:\n")
        for param, stats in skipped_low_variance_stats.items():
            print(f"  - {highlight_color}{param}{Colors.RESET}")
            print(f"    -> Most common value '{Colors.BOLD}{stats['most_common_val']}{Colors.RESET}' occurs {stats['most_common_count']} times ({stats['most_common_pct']:.1f}% of total).")
            print(f"    -> Stats: {stats['unique_count']} unique values | Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}")
        print()

    # --- 3. MULTI-PARAMETER COMBINATION ANALYSIS ---
    for n in range(2, max_correlation_depth + 1):
        level_name = {2: "Paired", 3: "Triple", 4: "Quadruple"}.get(n, f"{n}-Parameter")
        print(f"\n{subheader_color}--- [ {level_name} Parameter Analysis for Poor Performance ] ---{Colors.RESET}\n")
        print(f"Finding common {n}-parameter combinations in the {len(poor_results)} worst-performing results...\n")

        combo_counts = Counter()
        for res in poor_results:
            conditions = []
            flat_data = res.get('flat_truth_data', {})
            for key, value in flat_data.items():
                if key in categorical_params_stats:
                    conditions.append(f"{key}: {value}")
                elif key in numerical_thresholds:
                    if value <= numerical_thresholds[key]['q1']:
                        conditions.append(f"{key} <= {numerical_thresholds[key]['q1']:.3f}")
                    elif value >= numerical_thresholds[key]['q3']:
                        conditions.append(f"{key} >= {numerical_thresholds[key]['q3']:.3f}")
            
            conditions.sort()
            for combo in combinations(conditions, n):
                combo_counts[combo] += 1
        
        if not combo_counts:
            print(f"Could not find any {n}-parameter combinations to analyze.")
            continue

        print(f"Most frequent {n}-parameter combinations in poor results:")
        found_combos = False
        for combo, count in combo_counts.most_common(10):
            if count > 1:
                found_combos = True
                print(f"  - Occurrences: {Colors.BOLD}{count:3d}{Colors.RESET}")
                for param_detail in combo:
                    print(f"    -> {highlight_color}{param_detail}{Colors.RESET}")
                print()
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
    
    header_color = Colors.BOLD + Colors.MAGENTA
    subheader_color = Colors.CYAN
    
    print("\n" + header_color + "="*80 + Colors.RESET)
    print(header_color + " " * 25 + "OCR PERFORMANCE ANALYSIS REPORT" + Colors.RESET)
    print(header_color + "="*80 + Colors.RESET)
    print(f"Analyzed results from: {os.path.basename(input_file)}\n")

    total_files = len(results)
    print(f"{subheader_color}{'-' * 40}{Colors.RESET}")
    print(f"{Colors.BOLD}Overall Statistics{Colors.RESET}")
    print(f"{subheader_color}{'-' * 40}{Colors.RESET}")
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

    print(f"{subheader_color}{'-' * 40}{Colors.RESET}")
    print(f"{Colors.BOLD}Performance Distribution{Colors.RESET}")
    print(f"{subheader_color}{'-' * 40}{Colors.RESET}")
    print(f"{Colors.GREEN}Perfect (>= 99%): {perfect_scores:5d} ({perfect_scores/total_files:7.2%}){Colors.RESET}")
    print(f"{Colors.CYAN}Good (80% - 99%): {good_scores:5d} ({good_scores/total_files:7.2%}){Colors.RESET}")
    print(f"{Colors.YELLOW}Medium (50% - 80%): {medium_scores:5d} ({medium_scores/total_files:7.2%}){Colors.RESET}")
    print(f"{Colors.RED}Poor (< 50%):     {poor_scores:5d} ({poor_scores/total_files:7.2%}){Colors.RESET}\n")

    results.sort(key=lambda x: x['similarity_score_float'])
    worst_examples = results[:top_n]

    print(f"{header_color}{'-' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}Top {len(worst_examples)} Worst Performing Examples{Colors.RESET}")
    print(f"{header_color}{'-' * 80}{Colors.RESET}")
    for i, ex in enumerate(worst_examples):
        score = ex['similarity_score_float']
        truth = ex['truth_data'].get('text', 'N/A')
        ocr_text = ex['test_data'].get('easyocr_text', 'N/A')
        print(f"{i+1}. {Colors.BOLD}Image: {ex['image_name']}{Colors.RESET}")
        print(f"   Score: {Colors.RED}{score:.4f}{Colors.RESET}")
        print(f"   {Colors.GREEN}Truth: '{truth}'{Colors.RESET}")
        print(f"   {Colors.RED}OCR:   '{ocr_text}'{Colors.RESET}\n")
    
    analyze_parameter_correlations(results, max_correlation_depth=max_corr)
    print(header_color + "="*80 + Colors.RESET)

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

