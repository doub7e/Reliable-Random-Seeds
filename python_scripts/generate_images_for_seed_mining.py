import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel


def load_objects_from_file(file_path):
    """Load object prompts from a file in the prompt_dataset folder."""
    objects = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(", ")
                if len(parts) >= 3:
                    _, single, plural = parts
                    objects.append(plural)
    return objects


def load_settings_from_file(file_path):
    """Load settings/backgrounds from a file in the prompt_dataset folder."""
    settings = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                settings.append(line)
    return settings


def save_images(image_list, main_folder, folder_name):
    """Save generated images to the specified folder."""
    # Create the main folder if it doesn't exist
    if not os.path.exists(main_folder):
        os.makedirs(main_folder)
    
    # Create the subfolder
    subfolder_path = os.path.join(main_folder, folder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    
    # Save each image in the list to the subfolder
    for i, img in enumerate(image_list):
        img_path = os.path.join(subfolder_path, f'{i}.jpg')
        print(img_path)
        img.save(img_path)


def parse_arguments():
    """Parse command line arguments."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Generate images for seed mining with Stable Diffusion")
    
    parser.add_argument(
        "--pipeline", 
        type=str, 
        default="stabilityai/stable-diffusion-2-1",
        help="Diffusion model pipeline to use (default: stabilityai/stable-diffusion-2-1)"
    )
    
    parser.add_argument(
        "--objects_file", 
        type=str, 
        default=os.path.join(base_dir, "prompt_dataset", "objects_train.txt"),
        help="Path to the file containing object prompts"
    )
    
    parser.add_argument(
        "--settings_file", 
        type=str, 
        default=os.path.join(base_dir, "prompt_dataset", "backgrounds_train.txt"),
        help="Path to the file containing setting/background prompts"
    )
    
    parser.add_argument(
        "--output_dir_base", 
        type=str, 
        default="output",
        help="Base directory for saving generated images"
    )
    
    parser.add_argument(
        "--num_objects", 
        type=int, 
        default=15,
        help="Number of objects to use from the objects file (default: 15)"
    )
    
    parser.add_argument(
        "--num_settings", 
        type=int, 
        default=4,
        help="Number of settings to use from the settings file (default: 4)"
    )
    
    parser.add_argument(
        "--seed_range", 
        type=int, 
        default=100,
        help="Number of random seeds to use (default: 100)"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Extract model name from pipeline path for the output directory
    model_name = args.pipeline.split('/')[-1]
    
    # Build model parameter directory name
    model_params_dir = f"{model_name}_obj{args.num_objects}_set{args.num_settings}_seed{args.seed_range}"
    
    # Build complete output directory path:
    # output/{model_params_dir}/generated_images/{count}/{obj}-{seed}
    output_dir = os.path.join(
        args.output_dir_base,
        model_params_dir,
        "generated_images"
    )
    
    print(f"Images will be saved to: {output_dir}")
    
    # Load model
    pipe = StableDiffusionPipeline.from_pretrained(args.pipeline, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    # Load objects and settings
    objects = load_objects_from_file(args.objects_file)[:args.num_objects]
    settings = load_settings_from_file(args.settings_file)[:args.num_settings]
    
    nb_to_word = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    
    # Generate images
    for count in [2, 3, 4, 5, 6]:
        for seed in range(args.seed_range):
            for obj in objects:
                output_path = os.path.join(output_dir, f"{count}/{obj}-{seed}")
                if os.path.exists(output_path):
                    print(f"{output_path} exists, skipping")
                    continue
                    
                imgs = []
                for sett in settings:
                    prompt = f"{nb_to_word[count]} {obj}, {sett}"
                    
                    ref_image, ref_latents = pipe.__call__(
                        prompt,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        return_dict=False
                    )
                    ref_image = ref_image[0]
                    imgs.append(ref_image)
                    
                save_images(imgs, output_dir, f"{count}/{obj}-{seed}")


if __name__ == "__main__":
    main()
