# Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds

---

[![Spotlight](https://img.shields.io/badge/ICLR%202025-Spotlight-blue)](https://iclr.cc/Conferences/2025) 
[![arXiv](https://img.shields.io/badge/arXiv-2411.18810-b31b1b.svg)](https://arxiv.org/abs/2411.18810) 

[Shuangqi Li](mailto:shuangqi.li@epfl.ch)¹ · [Hieu Le](mailto:hle@cs.stonybrook.edu)¹ · [Jingyi Xu](mailto:jingyixu@cs.stonybrook.edu)² · [Mathieu Salzmann](mailto:mathieu.salzmann@epfl.ch)¹

¹EPFL, Switzerland 
²Stony Brook University, USA

## Overview

This repository contains the official implementation of our ICLR 2025 paper "Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds." 

### Abstract

Text-to-image diffusion models have demonstrated remarkable capability in generating realistic images from arbitrary text prompts. However, they often produce inconsistent results for compositional prompts such as "two dogs" or "a penguin on the right of a bowl". Understanding these inconsistencies is crucial for reliable image generation. 

In this paper, we highlight the significant role of initial noise in these inconsistencies, where certain noise patterns are more reliable for compositional prompts than others. Our analyses reveal that different initial random seeds tend to guide the model to place objects in distinct image areas, potentially adhering to specific patterns of camera angles and image composition associated with the seed. 

To improve the model's compositional ability, we propose a method for mining these reliable cases, resulting in a curated training set of generated images without requiring any manual annotation. By fine-tuning text-to-image models on these generated images, we significantly enhance their compositional capabilities. For numerical composition, we observe relative increases of 29.3% and 19.5% for Stable Diffusion and PixArt-α, respectively. Spatial composition sees even larger gains, with 60.7% for Stable Diffusion and 21.1% for PixArt-α.

## Requirements

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/doub7e/Reliable-Random-Seeds.git
cd Reliable-Random-Seeds

# Create and activate a conda environment
conda create -n reliable-seeds python=3.8
conda activate reliable-seeds

# Install dependencies
pip install -r requirements.txt
```
Warning: If you encounter any issues with the environment, please refer to [diffusers](https://github.com/huggingface/diffusers/tree/v0.30.0/examples/text_to_image) and [CogVLM2](https://github.com/THUDM/CogVLM2/blob/main/basic_demo/requirements.txt) for more details.

### Additional Requirements

- **CogVLM2**: Our seed mining process uses CogVLM2 for seed mining. You might need:
  - At least 48GB of VRAM for running the CogVLM2 server, or 16 GB for the Int4-quantized version.
  - Alternatively, you can modify the code to use another vision-language model / API.

- **GPT-4o API**: To reproduce our results in the final model evaluation, you'll need access to the OpenAI API with GPT-4o

## Comp90: Our Dataset for Text-to-Image Composition

Our paper introduces **Comp90**, a carefully curated dataset for compositional text-to-image seed-mining and evaluation:

- **Scale**: 90 object categories (foods, animals, everyday items) × 12 background settings
- **Division**: 60 categories/8 settings (training) and 30 categories/4 settings (testing)
- **Prompt Types**:
  - **Numerical**: 2-6 objects per prompt (e.g., "three apples in an old European town")
  - **Spatial**: 4 relations (on top of, left of, right of, under) with GPT-4o filtered compositions

This dataset allows systematic seed-mining and evaluation of text-to-image models' compositional capabilities across both numerical and spatial dimensions.

### Dataset Preparation

The `prompt_dataset` directory should contain:
- `objects_train.txt`: Object names used for training
- `backgrounds_train.txt`: Background settings for training
- `objects_eval.txt`: Object names used for evaluation 
- `backgrounds_eval.txt`: Background settings for evaluation

## Usage

### Complete Workflow

To reproduce our experiments with the default settings, you can use the provided script:

```bash
# First, run the CogVLM2 server in a separate terminal or as a background process
nohup python python_scripts/cogvlm2_server.py &

# Then, run the script
bash run_all.sh
```

This script will:
1. Generate images for seed mining
2. Evaluate images with CogVLM2
3. Find top-performing seeds
4. Generate a dataset with these seeds
5. Evaluate and optionally rectify the dataset
6. Finetune a text-to-image model
7. Evaluate the finetuned model

### Step-by-Step Replication

If you prefer to run each step separately, follow this workflow:

#### 0. Huggingface login

```bash
huggingface-cli login --token=hf_XXXX
```

#### 1. Generate Images for Seed Mining

```bash
python python_scripts/generate_images_for_seed_mining.py \
  --pipeline "stabilityai/stable-diffusion-2-1" \
  --num_objects 15 \
  --num_settings 4 \
  --seed_range 100
```

#### 2. Evaluate Images with CogVLM2

First, start the CogVLM2 server:
```bash
# Run in a separate terminal or as a background process
nohup python python_scripts/cogvlm2_server.py &
```

Then, evaluate the images:
```bash
python python_scripts/cogvlm2_request_for_seed_mining.py \
  --input_dir "output/stable-diffusion-2-1_obj15_set4_seed100/generated_images"
```

#### 3. Find Top-Performing Seeds

```bash
python python_scripts/get_top_seeds.py \
  --model_params "stable-diffusion-2-1_obj15_set4_seed100"
```
Note: You can change the `top_n` parameter to get more or fewer seeds. A comparison is provided in Table 10 in our paper.

#### 4. Generate Dataset with Top Seeds

```bash
python python_scripts/generate_dataset_with_top_seeds.py \
  --model_params "stable-diffusion-2-1_obj15_set4_seed100" \
  --base_model "stabilityai/stable-diffusion-2-1" \
  --push_to_hub \
  --dataset_name "HF_USERNAME/stable-diffusion-2-1_obj15_set4_seed100"
```

#### 5. (Optional) Evaluate and Rectify Dataset

```bash
# Request evaluations
python python_scripts/cogvlm2_request_for_dataset_eval.py \
  --model_params "stable-diffusion-2-1_obj15_set4_seed100"

# Rectify dataset based on evaluations
python python_scripts/evaluate_and_rectify_dataset.py \
  --model_params "stable-diffusion-2-1_obj15_set4_seed100" \
  --rectify \
  --push_to_hub \
  --dataset_name "HF_USERNAME/stable-diffusion-2-1_obj15_set4_seed100-rectified"
```

#### 6. Finetune Text-to-Image Model

```bash
# Using accelerate for distributed training
accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 python_scripts/finetune_text_to_image.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --dataset_name="HF_USERNAME/stable-diffusion-2-1_obj15_set4_seed100-rectified" \
  --caption_column="prompt" \
  --use_ema \
  --resolution=768 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --checkpointing_steps=1000 \
  --resume_from_checkpoint="latest" \
  --learning_rate=1e-06 \
  --scale_lr \
  --max_grad_norm=1 \
  --output_dir="output/stable-diffusion-2-1_obj15_set4_seed100/finetuned_models/d1d2mu1u2-attn1+attn2-qk-true/" \
  --trainable_layer="d1d2mu1u2" \
  --trainable_attn="attn1+attn2" \
  --trainable_module="qk"
```
Note: Alternative fine-tuning approaches, such as Low-Rank Adaptation (LoRA), remain unexplored in this project and could potentially yield superior results, particularly in terms of image quality and training efficiency.
Note: Comparisons between different fine-tuning configurations are provided in Table 9 in our paper.

#### 7. Evaluate the Finetuned Model

```bash
# Generate evaluation images with pretrained model (baseline)
python python_scripts/generate_images_for_evaluation.py \
  --model_name "stable-diffusion-2-1_obj15_set4_seed100" \
  --pipeline "stabilityai/stable-diffusion-2-1" \
  --unet_path "" \
  --output_dir "output/stable-diffusion-2-1_obj15_set4_seed100/pretrained-evaluation" \

# Evaluate with GPT-4o
python python_scripts/gpt4o_request_evaluation.py \
  --input_dir "output/stable-diffusion-2-1_obj15_set4_seed100/pretrained-evaluation"

# Generate evaluation images with finetuned model
python python_scripts/generate_images_for_evaluation.py \
  --model_name "stable-diffusion-2-1_obj15_set4_seed100" \
  --pipeline "stabilityai/stable-diffusion-2-1" \
  --unet_path "output/stable-diffusion-2-1_obj15_set4_seed100/finetuned_models/d1d2mu1u2-attn1+attn2-qk-true/unet" \
  --output_dir "output/stable-diffusion-2-1_obj15_set4_seed100/generated-images-for-evaluation" \

# Evaluate with GPT-4o
python python_scripts/gpt4o_request_evaluation.py \
  --input_dir "output/stable-diffusion-2-1_obj15_set4_seed100/generated-images-for-evaluation"

# Compare results
python python_scripts/final_evaluation.py \
  --result_files "output/stable-diffusion-2-1_obj15_set4_seed100/pretrained-evaluation/evaluation_results.pkl" "output/stable-diffusion-2-1_obj15_set4_seed100/generated-images-for-evaluation/evaluation_results.pkl" \
  --labels "Pretrained" "Fine-tuned"
```
Note: You can use top-performing seeds when generating evaluation images to further improve the results. A comparison is provided in Table 8 in our paper.

## Customization

You can customize the workflow by modifying parameters in `run_all.sh`:

- `PARAMS`: Model parameters string (format: `model_obj{num}_set{num}_seed{num}`)
- `MODEL_NAME`: Pretrained model to use/finetune
- `NUM_OBJECTS`: Number of objects to use for dataset generation
- `NUM_SETTINGS`: Number of background settings to use
- `SEED_RANGE`: Number of random seeds to evaluate
- `RECTIFIED_DATASET`: Whether to use rectified dataset (true/false)
- `TRAINABLE_LAYER`: Which layers to finetune (e.g., "d1d2mu1u2")
- `TRAINABLE_ATTN`: Which attention blocks to finetune (e.g., "attn1+attn2")
- `TRAINABLE_MODULE`: Which modules in attention to finetune (e.g., "qk")

## Results and Evaluation

The final evaluation will output:

1. Overall count accuracy for both pretrained and finetuned models
2. Per-count accuracy breakdown (how well each model performs on counts 2-6)
3. Per-object accuracy (which objects are recognized most/least accurately)

Results are saved in:
- `output/{model_params}/pretrained-evaluation/` for the baseline model
- `output/{model_params}/generated-images-for-evaluation/` for the finetuned model
- Summary files are generated as CSV and text files for easy analysis

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{li2025seeds,
      title={Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds}, 
      author={Shuangqi Li and Hieu Le and Jingyi Xu and Mathieu Salzmann},
      year={2025},
      booktitle={International Conference on Learning Representations},
}
```



## Acknowledgements

We created our code based on the following repositories:
- [diffusers](https://github.com/huggingface/diffusers/tree/v0.30.0/examples/text_to_image)
- [CogVLM2](https://github.com/THUDM/CogVLM2/blob/main/basic_demo)
