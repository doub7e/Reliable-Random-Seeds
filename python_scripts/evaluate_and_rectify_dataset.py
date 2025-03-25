#!/usr/bin/env python3
"""
Script to evaluate a dataset created with top-performing seeds.
It analyzes the responses from CogVLM2 to determine how accurately
the objects are represented in the images and saves evaluation metrics.
If requested, it can also rectify the dataset by fixing counts in prompts.
The rectified dataset can be pushed to HuggingFace Hub.
"""

import numpy as np
import pickle
import argparse
import os
import json
import re
import fnmatch
from tqdm import tqdm
import shutil
from PIL import Image
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: 'datasets' package not installed. HuggingFace Hub integration will not be available.")
    DATASETS_AVAILABLE = False


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_number_from_response(response):
    """
    Extract the number mentioned in the CogVLM2 response.
    
    Args:
        response (str): The response from CogVLM2
        
    Returns:
        str: The extracted number as a word ('zero', 'one', 'two', etc.)
    """
    # Handle None responses
    if response is None:
        return None
        
    # Get the first sentence which usually contains the count
    first_sent = response.split('.')[0].lower()
    words = first_sent.split()
    
    if '0' in first_sent or ' no ' in first_sent or 'zero' in first_sent:
        return 'zero'
    elif 'numerous' in first_sent or 'number of' in first_sent or 'variety of' in first_sent or 'large pile of' in first_sent:
        return "numerous"
    elif 'multitude of' in first_sent or "are multiple" in first_sent or "collection" in first_sent:
        return "numerous"
    elif any(word in ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20'] for word in words):
        for word in words:
            if word in ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']:
                return word
    elif 'ten' in first_sent or '10' in first_sent:
        return 'ten'
    elif 'one' in first_sent or '1' in first_sent:
        return 'one'
    elif 'two' in first_sent or '2' in first_sent:
        return 'two'
    elif 'three' in first_sent or '3' in first_sent:
        return 'three'
    elif 'four' in first_sent or '4' in first_sent:
        return 'four'
    elif 'five' in first_sent or '5' in first_sent:
        return 'five'
    elif 'six' in first_sent or '6' in first_sent:
        return 'six'
    elif 'seven' in first_sent or '7' in first_sent:
        return 'seven'
    elif 'eight' in first_sent or '8' in first_sent:
        return 'eight'
    elif 'nine' in first_sent or '9' in first_sent:
        return 'nine'
    
    # If no number is found, return None
    return None


def analyze_responses(response_data, count):
    """
    Analyze the responses from the model to determine accuracy rates by seed.
    
    Args:
        response_data (dict): Dictionary of model responses
        count (int): The expected count number (2-6)
        
    Returns:
        tuple: (results_by_seed, ordered_seeds, accuracy_rates, seed_accuracies, obj_accuracies, overall_accuracy, response_details)
              All numeric values are converted to standard Python types to ensure JSON compatibility.
    """
    results_seed = {}  # Results by seed only
    results_obj = {}   # Results by object
    response_details = {}  # For storing detailed information about each response
    
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
        
        if obj not in results_obj:
            results_obj[obj] = []
        
        # Track responses for this key
        response_details[key] = []
        
        # Check each response for correctness
        for i, res in enumerate(value):
            detected_number = get_number_from_response(res)
            
            if res is None:
                # Handle None responses
                results_seed[seed].append(0)
                results_obj[obj].append(0)
                response_details[key].append({
                    "response": None,
                    "is_correct": False,
                    "detected_number": None,
                    "needs_rectification": True,
                    "index": i
                })
                continue
                
            # Check if either the word or digit form is in the response
            is_correct = any(c.lower() in res.lower() for c in correct)
            results_seed[seed].append(1 if is_correct else 0)
            results_obj[obj].append(1 if is_correct else 0)
            
            # Store detailed information
            response_details[key].append({
                "response": res,
                "is_correct": bool(is_correct),  # Convert to standard Python bool
                "detected_number": detected_number,
                "needs_rectification": bool(not is_correct and detected_number is not None),  # Convert to standard Python bool
                "index": i
            })
    
    # Calculate accuracy rates for each seed
    seed_accuracies = {}
    for seed, results in results_seed.items():
        accuracy = sum(results) / len(results) if results else 0
        seed_accuracies[seed] = float(accuracy)  # Convert to standard Python float
    
    # Calculate accuracy rates for each object
    obj_accuracies = {}
    for obj, results in results_obj.items():
        accuracy = sum(results) / len(results) if results else 0
        obj_accuracies[obj] = float(accuracy)  # Convert to standard Python float
    
    # Sort seeds by accuracy (highest first)
    ranked_seeds = sorted(seed_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    ordered_seeds = [seed for seed, _ in ranked_seeds]
    accuracy_rates = [float(acc) for _, acc in ranked_seeds]  # Convert to standard Python float
    
    # Calculate overall accuracy
    all_results = [r for results in results_seed.values() for r in results]
    overall_accuracy = float(sum(all_results) / len(all_results) if all_results else 0)  # Convert to standard Python float
    
    return results_seed, ordered_seeds, accuracy_rates, seed_accuracies, obj_accuracies, overall_accuracy, response_details


def process_count(count, input_dir, output_dir):
    """
    Process a single count value.
    
    Args:
        count (int): The count number to analyze (2-6)
        input_dir (str): Directory containing response data
        output_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation results including accuracy metrics and detailed response info
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
        return None
    except Exception as e:
        print(f"Error loading the response file: {e}")
        return None
    
    # Analyze the responses
    results_seed, ordered_seeds, accuracies, seed_accuracies, obj_accuracies, overall_accuracy, response_details = analyze_responses(responses, count)
    
    # Print information about the data
    print(f"Analyzed {len(responses)} seed-object pairs")
    print(f"Total seeds: {len(results_seed)}")
    print(f"Overall accuracy for count {count}: {overall_accuracy:.4f}")
    
    if not results_seed:
        return None
    
    # Print the top 5 seeds
    print(f"\nTop 5 seeds for count {count}:")
    for i in range(min(5, len(ordered_seeds))):
        print(f"{i+1}. Seed {ordered_seeds[i]}: {accuracies[i]:.4f}")
    
    # Print the bottom 5 seeds
    print(f"\nBottom 5 seeds for count {count}:")
    for i in range(1, min(6, len(ordered_seeds))):
        idx = len(ordered_seeds) - i
        print(f"{i}. Seed {ordered_seeds[idx]}: {accuracies[idx]:.4f}")
    
    # Print object performance
    print(f"\nObject accuracies for count {count}:")
    sorted_objs = sorted(obj_accuracies.items(), key=lambda x: x[1], reverse=True)
    for obj, acc in sorted_objs[:5]:
        print(f"{obj}: {acc:.4f}")
    
    # Convert all NumPy values in dictionaries to standard Python types
    seed_accuracies_py = {k: float(v) for k, v in seed_accuracies.items()}
    obj_accuracies_py = {k: float(v) for k, v in obj_accuracies.items()}
    
    # Compile evaluation results with standard Python types
    evaluation_results = {
        "count": count,
        "overall_accuracy": float(overall_accuracy),
        "total_seed_object_pairs": len(responses),
        "total_seeds": len(results_seed),
        "total_objects": len(obj_accuracies),
        "seed_accuracies": seed_accuracies_py,
        "object_accuracies": obj_accuracies_py,
        "response_details": response_details
    }
    
    return evaluation_results


def rectify_dataset(model_dir, all_results, rectified_dir):
    """
    Create a rectified version of the dataset by fixing count mismatches.
    
    Args:
        model_dir (str): Base directory for the model
        all_results (dict): Evaluation results for all counts
        rectified_dir (str): Directory to save the rectified dataset
        
    Returns:
        dict: Rectified metadata
    """
    # Load the original dataset metadata
    dataset_dir = os.path.join(model_dir, "dataset")
    metadata_file = os.path.join(dataset_dir, "metadata.pkl")
    
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset metadata: {e}")
        return None
    
    # Create a dictionary to map image paths to metadata indices
    image_path_to_index = {}
    for i, path in enumerate(metadata["image_path"]):
        image_path_to_index[os.path.basename(path)] = i
    
    # Create the rectified dataset directory
    os.makedirs(rectified_dir, exist_ok=True)
    
    # Initialize rectified metadata
    rectified_metadata = {
        "prompt": [],
        "count": [],
        "seed": [],
        "image_path": [],
        "original_prompt": [],
        "rectified": []  # Track which prompts were rectified
    }
    
    # Initialize counters
    total_images = 0
    rectified_count = 0
    
    # Process each count and fix prompts where needed
    print("\nRectifying dataset...")
    for count, results in all_results.items():
        response_details = results["response_details"]
        
        for key, details_list in tqdm(response_details.items(), desc=f"Processing count {count}"):
            seed, obj = key
            
            for details in details_list:
                # Find the original image filename pattern
                # Format: {index}_{count}_{obj_idx}_{setting_idx}.png
                pattern = f"*_{count}_*_*.png"
                matching_files = []
                
                # Find all files for this count, then filter by seed
                for filename in os.listdir(dataset_dir):
                    if not filename.endswith('.png'):
                        continue
                    
                    if fnmatch.fnmatch(filename, pattern):
                        # Extract metadata index for this file
                        index = image_path_to_index.get(filename)
                        if index is not None and metadata["seed"][index] == int(seed) and obj in metadata["prompt"][index]:
                            matching_files.append(filename)
                
                # Skip if no matching files (shouldn't happen with proper dataset)
                if not matching_files:
                    continue
                
                # Get the image details
                idx = details["index"] % len(matching_files)
                filename = matching_files[idx]
                orig_index = image_path_to_index[filename]
                
                orig_prompt = metadata["prompt"][orig_index]
                orig_image_path = metadata["image_path"][orig_index]
                orig_seed = metadata["seed"][orig_index]
                orig_count = metadata["count"][orig_index]
                
                # Determine if we need to rectify this prompt
                needs_rectification = details["needs_rectification"]
                detected_number = details["detected_number"]
                
                # Create new filename and path for rectified dataset
                new_filename = filename
                new_image_path = os.path.join(rectified_dir, new_filename)
                
                # Copy the image to the rectified dataset
                shutil.copy(orig_image_path, new_image_path)
                
                # Update the prompt if rectification is needed
                if needs_rectification and detected_number:
                    # Replace the count in the prompt with the detected number
                    number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
                    number_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                    
                    # Extract the first word (supposed to be the count)
                    prompt_parts = orig_prompt.split()
                    if prompt_parts[0] in number_words + number_digits:
                        # Replace the count word
                        new_prompt = detected_number + " " + " ".join(prompt_parts[1:])
                        rectified_count += 1
                    else:
                        # If prompt doesn't start with a number (shouldn't happen), keep original
                        new_prompt = orig_prompt
                else:
                    new_prompt = orig_prompt
                
                # Add to rectified metadata
                rectified_metadata["prompt"].append(new_prompt)
                rectified_metadata["count"].append(orig_count)  # Keep the original count category
                rectified_metadata["seed"].append(orig_seed)
                rectified_metadata["image_path"].append(new_image_path)
                rectified_metadata["original_prompt"].append(orig_prompt)
                rectified_metadata["rectified"].append(needs_rectification and detected_number is not None)
                
                total_images += 1
    
    # Save the rectified metadata
    rectified_metadata_file = os.path.join(rectified_dir, "metadata.pkl")
    with open(rectified_metadata_file, 'wb') as f:
        pickle.dump(rectified_metadata, f)
    
    print(f"\nRectification complete!")
    print(f"Total images processed: {total_images}")
    print(f"Prompts rectified: {rectified_count} ({rectified_count/total_images*100:.2f}%)")
    print(f"Rectified dataset saved to: {rectified_dir}")
    print(f"Rectified metadata saved to: {rectified_metadata_file}")
    
    return rectified_metadata


def push_to_huggingface(rectified_metadata, rectified_dir, dataset_name, private=False):
    """
    Push the rectified dataset to HuggingFace Hub.
    
    Args:
        rectified_metadata (dict): Metadata for the rectified dataset
        rectified_dir (str): Directory containing the rectified dataset
        dataset_name (str): Name of the dataset on HuggingFace Hub
        private (bool): Whether the dataset should be private
        
    Returns:
        bool: Success status
    """
    if not DATASETS_AVAILABLE:
        print("Error: 'datasets' package is not installed. Cannot push to HuggingFace Hub.")
        print("Please install it with: pip install datasets")
        return False
        
    print(f"\nPreparing dataset for HuggingFace Hub: {dataset_name}")
    
    try:
        # Create a new dictionary for HuggingFace
        hf_dict = {
            "prompt": rectified_metadata["prompt"],
            "count": [int(c) if isinstance(c, np.integer) else c for c in rectified_metadata["count"]],
            "seed": [int(s) if isinstance(s, np.integer) else s for s in rectified_metadata["seed"]],
            "original_prompt": rectified_metadata["original_prompt"],
            "rectified": [bool(r) if isinstance(r, np.bool_) else r for r in rectified_metadata["rectified"]],
            "image_filename": [os.path.basename(path) for path in rectified_metadata["image_path"]]
        }
        
        # Load the images as numpy arrays
        print("Loading images...")
        images = []
        for image_path in tqdm(rectified_metadata["image_path"], desc="Processing images"):
            try:
                img = Image.open(image_path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Use a blank image as a placeholder
                blank = np.zeros((512, 512, 3), dtype=np.uint8)
                images.append(blank)
        
        hf_dict["image"] = images
        
        # Create the HuggingFace dataset with reduced batch size
        print("Creating HuggingFace Dataset...")
        ds = Dataset.from_dict(hf_dict)
        
        # Push to HuggingFace Hub using cached credentials
        print(f"Pushing dataset to HuggingFace Hub: {dataset_name}")
        ds.push_to_hub(
            dataset_name,
            private=private
        )
        print(f"Successfully pushed dataset to HuggingFace Hub: {dataset_name}")
        return True
        
    except Exception as e:
        print(f"Error pushing to HuggingFace Hub: {e}")
        print("Please ensure you have valid cached credentials.")
        print("Run 'huggingface-cli login' to log in if necessary.")
        return False


def main():
    """Main function to process command line arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate dataset created with top-performing seeds.")
    
    parser.add_argument("--counts", type=int, nargs='+', default=[2, 3, 4, 5, 6],
                        help="The count numbers to analyze (default: all counts 2-6)")
    
    parser.add_argument("--model_params", type=str, default="stable-diffusion-2-1_obj15_set4_seed100",
                        help="Model parameters string")
    
    parser.add_argument("--base_dir", type=str, default="output",
                        help="Base directory containing all data")
    
    parser.add_argument("--output_subdir", type=str, default="dataset_evaluation",
                        help="Subdirectory name where responses are saved and results will be stored")
    
    parser.add_argument("--rectify", action="store_true",
                        help="Create a rectified version of the dataset by fixing count mismatches")
    
    parser.add_argument("--rectified_subdir", type=str, default="rectified_dataset",
                        help="Subdirectory name to save the rectified dataset")
    
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the rectified dataset to HuggingFace Hub")
    
    parser.add_argument("--dataset_name", type=str, default="HF_USERNAME/stable-diffusion-2-1_obj15_set4_seed100-rectified",
                        help="Name of the dataset on HuggingFace Hub (required if --push_to_hub is set)")
    
    parser.add_argument("--private", action="store_true", default=False,
                        help="Whether the HuggingFace dataset should be private")
    
    args = parser.parse_args()
    
    # Validation
    if args.push_to_hub and not args.rectify:
        print("Error: --push_to_hub requires --rectify. You need to rectify the dataset before pushing to HuggingFace Hub.")
        return
        
    if args.push_to_hub and not args.dataset_name:
        print("Error: --dataset_name is required when --push_to_hub is set")
        return
    
    # Set up input and output directories based on the model parameters
    model_dir = os.path.join(args.base_dir, args.model_params)
    input_output_dir = os.path.join(model_dir, args.output_subdir)
    rectified_dir = os.path.join(model_dir, args.rectified_subdir)
    
    # Ensure evaluation directory exists
    if not os.path.exists(input_output_dir):
        print(f"Error: Directory '{input_output_dir}' does not exist.")
        print("Please run cogvlm2_request_for_dataset.py first to generate responses.")
        exit(1)
    
    # Process each count
    all_results = {}
    overall_counts = []
    overall_accuracies = []
    
    for count in args.counts:
        if count < 2 or count > 6:
            print(f"Warning: Count {count} is outside the expected range (2-6). Skipping.")
            continue
            
        evaluation_results = process_count(count, input_output_dir, input_output_dir)
        if evaluation_results is not None:
            all_results[count] = evaluation_results
            overall_counts.append(count)
            overall_accuracies.append(evaluation_results["overall_accuracy"])
    
    if all_results:
        # Save the evaluation results to a json file
        evaluation_file = os.path.join(input_output_dir, 'dataset_evaluation.json')
        
        # Remove the response_details from the saved JSON to keep it manageable
        json_results = {}
        for count, result in all_results.items():
            json_results[count] = {k: v for k, v in result.items() if k != 'response_details'}
        
        with open(evaluation_file, 'w') as f:
            json.dump(json_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved evaluation results to {evaluation_file}")
        
        # Create a summary text file
        summary_file = os.path.join(input_output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Dataset Evaluation Summary for {args.model_params}\n")
            f.write(f"{'='*60}\n\n")
            
            # Write count-specific statistics
            f.write("Count | Overall Accuracy | Seeds | Objects\n")
            f.write(f"{'-'*50}\n")
            
            for count in sorted(all_results.keys()):
                res = all_results[count]
                f.write(f"{count:5d} | {res['overall_accuracy']:16.4f} | {res['total_seeds']:5d} | {res['total_objects']:7d}\n")
            
            # Calculate and write overall metrics
            avg_accuracy = sum(overall_accuracies) / len(overall_accuracies) if overall_accuracies else 0
            f.write(f"\nAverage accuracy across all counts: {avg_accuracy:.4f}\n")
            
            # Write object performance analysis
            f.write("\nObject Performance Analysis:\n")
            f.write(f"{'-'*60}\n")
            
            # Combine object accuracies across all counts
            all_objs = {}
            for count, res in all_results.items():
                for obj, acc in res["object_accuracies"].items():
                    if obj not in all_objs:
                        all_objs[obj] = {"accuracies": [], "total": 0}
                    all_objs[obj]["accuracies"].append(acc)
                    all_objs[obj]["total"] += 1
            
            # Calculate average accuracy for each object
            for obj in all_objs:
                all_objs[obj]["avg"] = sum(all_objs[obj]["accuracies"]) / all_objs[obj]["total"]
            
            # Sort and write top and bottom objects
            sorted_objs = sorted(all_objs.items(), key=lambda x: x[1]["avg"], reverse=True)
            
            f.write("\nTop 10 most accurately recognized objects:\n")
            for i, (obj, data) in enumerate(sorted_objs[:10], 1):
                f.write(f"{i:2d}. {obj:15s}: {data['avg']:.4f} (across {data['total']} counts)\n")
            
            f.write("\nBottom 10 least accurately recognized objects:\n")
            for i, (obj, data) in enumerate(sorted_objs[-10:], 1):
                f.write(f"{i:2d}. {obj:15s}: {data['avg']:.4f} (across {data['total']} counts)\n")
        
        print(f"Saved evaluation summary to {summary_file}")
        
        # Rectify dataset if requested
        rectified_metadata = None
        if args.rectify:
            rectified_metadata = rectify_dataset(model_dir, all_results, rectified_dir)
            
            # Push to HuggingFace Hub if requested
            if args.push_to_hub and rectified_metadata:
                push_to_huggingface(
                    rectified_metadata, 
                    rectified_dir, 
                    args.dataset_name, 
                    private=args.private
                )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
