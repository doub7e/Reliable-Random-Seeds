"""
This script is designed to mimic the OpenAI API interface with CogVLM2 Chat
It demonstrates how to integrate image and text-based input to generate a response.
Currently, the model can only handle a single image.
Therefore, do not use this script to process multiple images in one conversation. (includes images from history)
And it only works on the chat model, not the base model.
"""
import requests
import json
import base64
import os
import pickle
import argparse

base_url = "http://127.0.0.1:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
    """
    This function sends a request to the chat API to generate a response based on the given messages.

    Args:
        model (str): The name of the model to use for generating the response.
        messages (list): A list of message dictionaries representing the conversation history.
        temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
        max_tokens (int): The maximum length of the generated response.
        top_p (float): Controls diversity of response by filtering less likely options.
        use_stream (bool): Determines whether to use a streaming response or a single response.

    The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
    It then handles the response, either as a stream (for ongoing responses) or a single message.
    """

    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
            return content
    else:
        print("Error:", response.status_code)
        return None


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


def simple_image_chat(objects=None, use_stream=True, img_path=None):
    """
    Facilitates a simple chat interaction involving an image.

    Args:
        use_stream (bool): Specifies whether to use streaming for chat responses.
        img_path (str): Path to the image file to be included in the chat.

    This function encodes the specified image and constructs a predefined conversation involving the image.
    It then calls `create_chat_completion` to generate a response from the model.
    The conversation includes asking about the content of the image and a follow-up question.
    """
    
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": f"Answer in one sentence: How many {objects} are in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
    ]
    
    resp = create_chat_completion("cogvlm2", messages=messages, use_stream=use_stream)
    return resp


if __name__ == "__main__":
    # Add command line arguments to match generate_images_for_seed_mining.py
    parser = argparse.ArgumentParser(description="Process generated images with CogVLM2")
    parser.add_argument("--model_params", type=str, 
                        default="stable-diffusion-2-1_obj15_set4_seed100",
                        help="Model parameters (e.g., 'stable-diffusion-2-1_obj15_set4_seed100')")
    
    parser.add_argument("--base_dir", type=str, default="output",
                        help="Base directory containing all data")
    
    args = parser.parse_args()
    
    # Set up directories based on the model parameters
    input_dir = os.path.join(args.base_dir, args.model_params, "generated_images")
    output_dir = os.path.join(args.base_dir, args.model_params, "seed-mining")
    
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        print("Please run generate_images_for_seed_mining.py first or specify the correct directory.")
        exit(1)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process each count subdirectory (2-6)
    for count_dir in sorted(os.listdir(input_dir)):
        count_path = os.path.join(input_dir, count_dir)
        
        # Skip if not a directory or not a count directory
        if not os.path.isdir(count_path) or not count_dir.isdigit():
            continue
            
        print(f"Processing count: {count_dir}")
        results = {}
        
        # Process each category-seed directory
        for category_seed in sorted(os.listdir(count_path)):
            imgs_dir = os.path.join(count_path, category_seed)
            
            # Skip if not a directory
            if not os.path.isdir(imgs_dir):
                # Throw an error if not a directory
                raise ValueError(f"Error: '{imgs_dir}' is not a directory. The directory structure may be incorrect.")
                
            try:
                # Extract category and seed from directory name
                category, seed = category_seed.split('-')
                results[(seed, category)] = []
                
                print(f"---------------- {category} (seed {seed}) ----------------")
                
                # Process each image in the directory
                image_files = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.png'))]
                for file in sorted(image_files):
                    file_path = os.path.join(imgs_dir, file)
                    
                    try:
                        # Query the model
                        resp = simple_image_chat(objects=category, use_stream=False, img_path=file_path)
                        if resp:
                            results[(seed, category)].append(resp)
                        else:
                            print(f"Warning: No response for {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        results[(seed, category)].append(f"ERROR: {str(e)}")
            except Exception as e:
                print(f"Error processing directory {category_seed}: {e}")
        
        # Save results for this count
        output_file = os.path.join(output_dir, f"responses_count{count_dir}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved results to {output_file}")
