import base64
import os
import pickle
import argparse
import sys
from tqdm import tqdm

from openai import OpenAI


BASE_URL = "YOUR_API_URL"
API_SECRET_KEY = "YOUR_API_SECRET_KEY"
client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)


def encode_image(image_path):
    """
    Encodes an image file into a base64 string.
    Args:
        image_path (str): The path to the image file.

    This function opens the specified image file, reads its content, and encodes it into a base64 string.
    The base64 encoding is used to send images over HTTP as text.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def simple_image_chat(objects=None, img_path=None):
    
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Answer in one sentence: How many {objects} are in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                            # "resize": 512
                            "detail": "low"
                       }
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    

    print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate images using GPT-4o to count objects")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing images to evaluate (must contain generation_metadata.pkl)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (default: 'evaluation_results.pkl')")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check for metadata file
    metadata_path = os.path.join(args.input_dir, "generation_metadata.pkl")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}", file=sys.stderr)
        print("Make sure you've generated images using generate_images_for_evaluation.py", file=sys.stderr)
        sys.exit(1)
    
    # Load metadata
    try:
        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Check metadata keys - handle both old and new formats
        if "objects" in metadata and "counts" in metadata and "filenames" in metadata:
            # We have the expected keys format
            required_keys = ["objects", "counts", "filenames"]
        elif "prompts" in metadata and "objects" in metadata and "counts" in metadata:
            # Newer format with prompts
            required_keys = ["prompts", "objects", "counts", "filenames"]
            # No need to transform, the code checks keys separately
        else:
            print(f"Error: Metadata file has unexpected format", file=sys.stderr)
            print(f"Found keys: {list(metadata.keys())}", file=sys.stderr)
            sys.exit(1)
            
        print(f"Found metadata for {len(metadata['filenames'] if 'filenames' in metadata else metadata['objects'])} images")
    except Exception as e:
        print(f"Error loading metadata: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output filename
    output_name = args.output_name or "evaluation_results.pkl"
    output_file = os.path.join(args.input_dir, output_name)
    
    # Initialize results dictionary
    results = {}
    
    # Get the list of filenames from metadata
    filenames = metadata.get('filenames', [])
    
    # Print evaluation summary
    print(f"Evaluating {len(filenames)} images")
    print(f"Results will be saved to {output_file}")
    
    # Process images with progress bar
    try:
        with tqdm(total=len(filenames), desc="Evaluating images") as pbar:
            for i, filename in enumerate(filenames):
                try:
                    # Get object and count from metadata
                    object_name = metadata['objects'][i]
                    count = metadata['counts'][i]
                    
                    # Initialize results dictionary entry if needed
                    if (object_name, str(count)) not in results:
                        results[(object_name, str(count))] = []
                    
                    # Process the image
                    file_path = os.path.join(args.input_dir, filename)
                    if not os.path.exists(file_path):
                        print(f"\nWarning: Image file not found: {file_path}")
                        continue
                        
                    resp = simple_image_chat(objects=object_name, img_path=file_path)
                    results[(object_name, str(count))].append(resp)
                    
                    # Update progress bar with current file
                    pbar.set_description(f"Processing: {filename}")
                    
                except Exception as e:
                    print(f"\nError processing {filename}: {str(e)}", file=sys.stderr)
                    # Continue with next image
                
                # Update progress bar
                pbar.update(1)
                
                # Save intermediate results every 10 images
                if (i + 1) % 10 == 0:
                    with open(output_file, 'wb') as f:
                        pickle.dump(results, f)
                    print(f"\nSaved intermediate results ({i+1}/{len(filenames)} images processed)")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
    
    # Save final results
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nEvaluation complete!")
    print(f"Processed {len(filenames)} images")
    print(f"Results saved to {output_file}")
    print(f"Next steps: Analyze results with final_evaluation.py")


