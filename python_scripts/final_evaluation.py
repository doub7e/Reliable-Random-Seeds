#!/usr/bin/env python3
"""
Script to analyze the evaluation results from gpt4o_request_evaluation.py.
This script processes the GPT-4o responses, calculates accuracy metrics, and
compares performance between different models without visualizations.

Example usage:
    # Analyze results using default model name and paths
    python python_scripts/final_evaluation.py
    
    # Analyze results from a specific model
    python python_scripts/final_evaluation.py --model_name "fine-tuned-model"
    
    # Analyze results from a specific result file
    python python_scripts/final_evaluation.py --result_file "path/to/evaluation_results.pkl"
    
    # Compare multiple models
    python python_scripts/final_evaluation.py --result_files eval/pretrained/evaluation_results.pkl eval/fine-tuned/evaluation_results.pkl --labels "Pretrained" "Fine-tuned"
"""

import os
import re
import pickle
import argparse
import sys
from collections import defaultdict


def extract_number_from_response(response):
    """
    Extract the number mentioned in the GPT-4o response.
    
    Args:
        response (str): The GPT-4o response text
        
    Returns:
        int or None: The extracted number, or None if not found
    """
    if not response:
        return None
        
    # Pattern to match common number formats in responses
    patterns = [
        r'there (?:are|is) (\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten) .+ in the image',
        r'the image shows (\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'I can see (\d+|one|two|three|four|five|six|seven|eight|nine|ten)',
        r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)',  # Fallback pattern
    ]
    
    word_to_number = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'zero': 0, 'no': 0
    }
    
    response_lower = response.lower()
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            number_text = match.group(1)
            
            # Convert word to number if needed
            if number_text in word_to_number:
                return word_to_number[number_text]
            
            # Handle digit case
            try:
                return int(number_text)
            except ValueError:
                continue
                
    # Check for special cases
    if "no" in response_lower and any(obj in response_lower for obj in ["object", "item"]):
        return 0
    if "cannot" in response_lower or "not able to" in response_lower:
        return None
    
    
    return None


def calculate_accuracy(results, verbose=False):
    """
    Calculate accuracy metrics from GPT-4o evaluation results.
    
    Args:
        results (dict): The results dictionary from GPT-4o evaluation
        verbose (bool): Whether to print detailed results
        
    Returns:
        dict: Dictionary containing overall, per-count, and per-object accuracy metrics
    """
    total_correct = 0
    total_examples = 0
    count_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    object_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Detailed results for analysis
    detailed_results = defaultdict(lambda: defaultdict(list))
    
    for key, responses in results.items():
        obj, expected_count = key
        expected_count = int(expected_count)
        
        if verbose:
            print(f"\nAnalyzing {obj}, expected count: {expected_count}")
        
        for i, response in enumerate(responses):
            extracted_count = extract_number_from_response(response)
            
            if verbose and extracted_count is not None:
                print(f"  Response {i+1}: Extracted {extracted_count}, Expected {expected_count}")
            elif verbose:
                print(f"  Response {i+1}: Could not extract count")
            
            is_correct = (extracted_count == expected_count)
            detailed_results[obj][expected_count].append(is_correct)
            
            if is_correct:
                total_correct += 1
                count_metrics[expected_count]['correct'] += 1
                object_metrics[obj]['correct'] += 1
            
            total_examples += 1
            count_metrics[expected_count]['total'] += 1
            object_metrics[obj]['total'] += 1
    
    # Calculate overall accuracy
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
    
    # Calculate per-count accuracy
    count_accuracy = {}
    for count, metrics in count_metrics.items():
        count_accuracy[count] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
    
    # Calculate per-object accuracy
    object_accuracy = {}
    for obj, metrics in object_metrics.items():
        object_accuracy[obj] = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'count_accuracy': count_accuracy,
        'object_accuracy': object_accuracy,
        'total_examples': total_examples,
        'total_correct': total_correct,
        'detailed_results': detailed_results
    }


def compare_models(results_files, labels=None):
    """
    Compare performance between different models.
    
    Args:
        results_files (list): List of paths to result files
        labels (list, optional): Labels for each model
    """
    if not labels:
        labels = [os.path.basename(os.path.dirname(f)) for f in results_files]
    
    all_metrics = []
    
    for i, file_path in enumerate(results_files):
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            
            metrics = calculate_accuracy(results)
            all_metrics.append(metrics)
            
            print(f"\n{labels[i]} Results:")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print(f"Total Examples: {metrics['total_examples']}")
            print(f"Total Correct: {metrics['total_correct']}")
            
            print(f"\n{labels[i]} Accuracy by Count:")
            for count, acc in sorted(metrics['count_accuracy'].items()):
                print(f"  Count {count}: {acc:.4f}")
                
            print(f"\n{labels[i]} Top 5 Objects by Accuracy:")
            sorted_objs = sorted(metrics['object_accuracy'].items(), key=lambda x: x[1], reverse=True)
            for obj, acc in sorted_objs[:5]:
                print(f"  {obj}: {acc:.4f}")
                
            if len(sorted_objs) > 5:
                print(f"\n{labels[i]} Bottom 5 Objects by Accuracy:")
                for obj, acc in sorted_objs[-5:]:
                    print(f"  {obj}: {acc:.4f}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            all_metrics.append(None)
    
    # If we have at least two valid results, create comparison table
    valid_metrics = [m for m in all_metrics if m is not None]
    valid_labels = [labels[i] for i, m in enumerate(all_metrics) if m is not None]
    
    if len(valid_metrics) >= 2:
        # Print detailed comparison
        print("\nModel Comparison:")
        
        # Overall accuracy comparison
        print("\nOverall Accuracy:")
        for i, (metrics, label) in enumerate(zip(valid_metrics, valid_labels)):
            print(f"  {label}: {metrics['overall_accuracy']:.4f}")
        
        # Accuracy by count comparison
        counts = sorted({count for m in valid_metrics for count in m['count_accuracy'].keys()})
        
        print("\nAccuracy by Count:")
        print(f"{'Model':<15} | {'Overall':<10} | " + " | ".join([f"Count {c}".ljust(10) for c in counts]))
        print("-" * (15 + 13 + 13 * len(counts)))
        
        for i, (metrics, label) in enumerate(zip(valid_metrics, valid_labels)):
            count_accs = [f"{metrics['count_accuracy'].get(count, 0):.4f}".ljust(10) for count in counts]
            print(f"{label:<15} | {metrics['overall_accuracy']:<10.4f} | " + " | ".join(count_accs))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GPT-4o evaluation results")
    
    # Model name parameter
    parser.add_argument(
        "--model_name",
        type=str,
        default="stable-diffusion-2-1_obj15_set4_seed100",
        help="Name of the model for directory naming (default: stable-diffusion-2-1_obj15_set4_seed100)"
    )
    
    # Single result file
    parser.add_argument(
        "--result_file", 
        type=str, 
        default=None,
        help="Path to a single result file to analyze (default: output/{model_name}/generated-images-for-evaluation/evaluation_results.pkl)"
    )
    
    # Multiple result files for comparison
    parser.add_argument(
        "--result_files", 
        type=str, 
        nargs='+',
        help="Paths to multiple result files for comparison"
    )
    
    parser.add_argument(
        "--labels", 
        type=str, 
        nargs='+',
        help="Labels for each model in comparison (must match number of result files)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed analysis information"
    )
    
    args = parser.parse_args()
    
    # Set default result_file path if not provided
    if args.result_file is None:
        args.result_file = f"output/{args.model_name}/generated-images-for-evaluation/evaluation_results.pkl"
    
    return args


def main():
    args = parse_arguments()
    
    # Handle single result file
    if args.result_file and not args.result_files:
        try:
            print(f"Analyzing results from: {args.result_file}")
            
            # Check if file exists
            if not os.path.exists(args.result_file):
                print(f"Error: Result file not found at {args.result_file}")
                print("Make sure you've evaluated images with gpt4o_request_evaluation.py first")
                print(f"Command: python python_scripts/gpt4o_request_evaluation.py --input_dir \"output/{args.model_name}/generated-images-for-evaluation/\"")
                sys.exit(1)
                
            with open(args.result_file, 'rb') as f:
                results = pickle.load(f)
            
            metrics = calculate_accuracy(results, verbose=args.verbose)
            
            print("\nSummary Statistics:")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print(f"Total Examples: {metrics['total_examples']}")
            print(f"Total Correct: {metrics['total_correct']}")
            
            print("\nAccuracy by Count:")
            for count, acc in sorted(metrics['count_accuracy'].items()):
                print(f"  Count {count}: {acc:.4f}")
            
            print("\nTop 5 Objects by Accuracy:")
            sorted_objs = sorted(metrics['object_accuracy'].items(), key=lambda x: x[1], reverse=True)
            for obj, acc in sorted_objs[:5]:
                print(f"  {obj}: {acc:.4f}")
                
            if len(sorted_objs) > 5:
                print("\nBottom 5 Objects by Accuracy:")
                for obj, acc in sorted_objs[-5:]:
                    print(f"  {obj}: {acc:.4f}")
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
            import traceback
            traceback.print_exc()
    
    # Handle multiple result files for comparison
    elif args.result_files:
        # Verify all files exist
        missing_files = [f for f in args.result_files if not os.path.exists(f)]
        if missing_files:
            print(f"Error: The following result files were not found:")
            for f in missing_files:
                print(f"  - {f}")
            print("Make sure you've evaluated all models with gpt4o_request_evaluation.py first")
            sys.exit(1)
            
        if args.labels and len(args.labels) != len(args.result_files):
            print("Error: Number of labels must match number of result files")
            return
            
        # Generate default labels based on directory names if not provided
        if not args.labels:
            args.labels = []
            for f in args.result_files:
                model_dir = os.path.dirname(f)
                model_name = os.path.basename(model_dir)
                args.labels.append(model_name)
                
        compare_models(args.result_files, args.labels)
    
    else:
        print("Error: Please specify either --result_file or --result_files")


if __name__ == "__main__":
    main()
