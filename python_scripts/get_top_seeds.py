#!/usr/bin/env python3
"""
Script to analyze the performance of different random seeds in image generation.
It finds the top-performing seeds based on model responses, performs chi-square tests,
and saves the top seeds in a pickle file.
"""

import numpy as np
from scipy import stats
import pickle
import argparse
import os


def chi_square_test(num_seeds, images_per_seed, ratios):
    """
    Perform a chi-square test to determine if seed affects image quality.
    
    Args:
        num_seeds (int): Number of different seeds used
        images_per_seed (int): Number of images generated per seed
        ratios (list): List of success ratios for each seed
        
    Returns:
        tuple: (chi2 statistic, p-value)
    """
    # Calculate observed frequencies
    observed = np.column_stack([
        np.round(np.array(ratios) * images_per_seed).astype(int),
        images_per_seed - np.round(np.array(ratios) * images_per_seed).astype(int)
    ])
    
    # Calculate expected frequencies
    total_high_quality = np.sum(observed[:, 0])
    total_images = num_seeds * images_per_seed
    expected_ratio = total_high_quality / total_images
    expected = np.column_stack([
        np.full(num_seeds, expected_ratio * images_per_seed),
        np.full(num_seeds, (1 - expected_ratio) * images_per_seed)
    ])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Print results
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p_value:.12f}")
    print(f"Degrees of freedom: {dof}")
    
    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        print(f"\nThe p-value ({p_value:.12f}) is less than the significance level ({alpha}).")
        print("We reject the null hypothesis.")
        print("There is strong evidence to suggest that the seed influences image quality.")
    else:
        print(f"\nThe p-value ({p_value:.12f}) is not less than the significance level ({alpha}).")
        print("We fail to reject the null hypothesis.")
        print("There is not enough evidence to conclude that the seed influences image quality.")
    
    return chi2, p_value


def analyze_responses(response_data, count):
    """
    Analyze the responses from the model to determine accuracy rates by seed.
    
    Args:
        response_data (dict): Dictionary of model responses
        count (int): The expected count number (2-6)
        
    Returns:
        tuple: (results_by_seed, ordered_seeds, accuracy_rates)
    """
    results_seed = {}  # Results by seed only
    
    correct_values = {
        2: ["two", "2"],
        3: ["three", "3"],
        4: ["four", "4"],
        5: ["five", "5"],
        6: ["six", "6"]
    }
    
    correct = correct_values[count]
    
    # Process each response
    for key, value in response_data.items():
        seed, obj = key
        
        if seed not in results_seed:
            results_seed[seed] = []
        
        # Check each response for correctness
        for res in value:
            if res is None:
                # Handle None responses
                results_seed[seed].append(0)
                continue
                
            # Check if either the word or digit form is in the response
            is_correct = any(c.lower() in res.lower() for c in correct)
            results_seed[seed].append(1 if is_correct else 0)
    
    # Calculate accuracy rates for each seed
    seed_accuracies = {}
    for seed, results in results_seed.items():
        accuracy = sum(results) / len(results) if results else 0
        seed_accuracies[seed] = accuracy
    
    # Sort seeds by accuracy (highest first)
    ranked_seeds = sorted(seed_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    ordered_seeds = [seed for seed, _ in ranked_seeds]
    accuracy_rates = [acc for _, acc in ranked_seeds]
    
    return results_seed, ordered_seeds, accuracy_rates


def process_count(count, input_dir, output_dir, top_n):
    """
    Process a single count value.
    
    Args:
        count (int): The count number to analyze (2-6)
        input_dir (str): Directory containing response data
        output_dir (str): Directory to save results
        top_n (int): Number of top seeds to save
        
    Returns:
        tuple: (top_seeds, chi2, p_value) - The top seeds and chi-square test results
    """
    # Construct the path to the response file
    responses_file = os.path.join(input_dir, f"responses_count{count}.pkl")
    
    print(f"\n{'='*40}")
    print(f"Processing Count {count}")
    print(f"{'='*40}")
    print(f"Loading responses from: {responses_file}")
    
    # Load the response data
    try:
        with open(responses_file, 'rb') as f:
            responses = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the response file at {responses_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading the response file: {e}")
        return None, None, None
    
    # Analyze the responses
    results_seed, ordered_seeds, accuracies = analyze_responses(responses, count)
    
    # Print information about the data
    print(f"Analyzed {len(responses)} seed-object pairs")
    print(f"Total seeds: {len(results_seed)}")
    
    if not results_seed:
        return None, None, None
    
    # Print the top N seeds
    print(f"\nTop {top_n} seeds for count {count}:")
    for i in range(min(top_n, len(ordered_seeds))):
        print(f"{i+1}. Seed {ordered_seeds[i]}: {accuracies[i]:.4f}")
    
    # Print the bottom N seeds
    print(f"\nBottom {top_n} seeds for count {count}:")
    for i in range(1, min(top_n+1, len(ordered_seeds))):
        idx = len(ordered_seeds) - i
        print(f"{i}. Seed {ordered_seeds[idx]}: {accuracies[idx]:.4f}")
    
    # Perform chi-square test
    print("\nPerforming chi-square test to assess seed influence:")
    first_seed = next(iter(results_seed))
    images_per_seed = len(results_seed[first_seed])
    chi2, p_value = chi_square_test(len(results_seed), images_per_seed, accuracies)
    
    # Return the top seeds and stats
    return ordered_seeds[:top_n], chi2, p_value


def main():
    """Main function to process command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze model responses to find top-performing seeds.")
    
    parser.add_argument("--counts", type=int, nargs='+', default=[2, 3, 4, 5, 6],
                        help="The count numbers to analyze (default: all counts 2-6)")
    
    parser.add_argument("--model_params", type=str, 
                        default="stable-diffusion-2-1_obj15_set4_seed100",
                        help="Model parameters string")
    
    parser.add_argument("--base_dir", type=str, default="output",
                        help="Base directory containing all data")
    
    parser.add_argument("--top_n", type=int, default=3,
                        help="Number of top/bottom seeds to print and save for each count")
    
    parser.add_argument("--include_stats", action="store_true",
                        help="Include chi-square statistics in the pickle file")
    
    args = parser.parse_args()
    
    # Set up input and output directories based on the model parameters
    input_dir = os.path.join(args.base_dir, args.model_params, "seed-mining")
    output_dir = os.path.join(args.base_dir, args.model_params, "seed-analysis")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each count
    results = {}
    for count in args.counts:
        if count < 2 or count > 6:
            print(f"Warning: Count {count} is outside the expected range (2-6). Skipping.")
            continue
            
        top_seeds, chi2, p_value = process_count(count, input_dir, output_dir, args.top_n)
        if top_seeds is not None:
            # Only save the top seeds
            if args.include_stats:
                results[count] = {
                    "seeds": top_seeds,
                    "chi2": float(chi2),
                    "p_value": float(p_value)
                }
            else:
                results[count] = top_seeds
    
    # Save the results as a pickle file
    if results:
        pickle_file = os.path.join(output_dir, 'top_seeds.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved top {args.top_n} seeds for each count to {pickle_file}")
        
        # Also create a text summary
        summary_file = os.path.join(output_dir, 'chi_square_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Chi-square Test Summary for {args.model_params}\n")
            f.write(f"{'='*60}\n\n")
            f.write("Count | Chi-Square | p-value | Significant?\n")
            f.write(f"{'-'*50}\n")
            
            for count in sorted(results.keys()):
                if args.include_stats:
                    chi2 = results[count]['chi2']
                    p_value = results[count]['p_value']
                    is_significant = "YES" if p_value < 0.05 else "NO"
                    f.write(f"{count:5d} | {chi2:10.4f} | {p_value:10.8f} | {is_significant}\n")
        
        print(f"Saved chi-square summary to {summary_file}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main() 
