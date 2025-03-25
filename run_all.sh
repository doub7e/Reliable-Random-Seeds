#!/bin/bash
# Full workflow script for count-specific SD2.1 finetuning

# Huggingface login
huggingface-cli login --token=hf_XXXX

# Set common parameters
PARAMS="stable-diffusion-2-1_obj15_set4_seed100"
MODEL_NAME="stabilityai/stable-diffusion-2-1"
CAPTION_COLUMN="prompt"
RECTIFIED_DATASET=true

# Define paths based on above settings
BASE_OUTPUT_DIR="output"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | awk -F'/' '{print $NF}')
NUM_OBJECTS=15
NUM_SETTINGS=4
SEED_RANGE=100
SEED_MINING_DIR="${BASE_OUTPUT_DIR}/${MODEL_SHORT_NAME}_obj${NUM_OBJECTS}_set${NUM_SETTINGS}_seed${SEED_RANGE}"
SEED_MINING_IMG_DIR="${SEED_MINING_DIR}/generated_images"

# Set variables for finetuning and evaluation
DATASET_NAME="Doub7e/${PARAMS}"
TRAINABLE_LAYER="d1d2mu1u2"
TRAINABLE_ATTN="attn1+attn2"
TRAINABLE_MODULE="qk"
OUTPUT_NAME="${BASE_OUTPUT_DIR}/${PARAMS}/finetuned_models/${TRAINABLE_LAYER}-${TRAINABLE_ATTN}-${TRAINABLE_MODULE}-${RECTIFIED_DATASET}"
EVALUATION_DIR="${BASE_OUTPUT_DIR}/${PARAMS}/generated-images-for-evaluation"
PRETRAINED_EVALUATION_DIR="${BASE_OUTPUT_DIR}/${PARAMS}/pretrained-evaluation"

echo "==========================================================="
echo "Starting complete workflow for ${PARAMS}"
echo "==========================================================="

# 1. Generate images for seed mining
echo "Step 1: Generating images for seed mining..."
python python_scripts/generate_images_for_seed_mining.py \
  --pipeline "${MODEL_NAME}" \
  --num_objects ${NUM_OBJECTS} \
  --num_settings ${NUM_SETTINGS} \
  --seed_range ${SEED_RANGE} \
  --output_dir_base "${BASE_OUTPUT_DIR}"

# 2. Use cogvlm2 to generate captions for images
echo "Step 2: Running CogVLM2 evaluation for seed mining..."
# First ensure the server is running (needs to be started separately)
# E.g., nohup python python_scripts/cogvlm2_server.py &
python python_scripts/cogvlm2_request_for_seed_mining.py \
  --input_dir "${SEED_MINING_IMG_DIR}"

# 3. Find top-performing seeds
echo "Step 3: Analyzing seeds to find best performers..."
python python_scripts/get_top_seeds.py \
  --model_params "${PARAMS}" \

# 4. Generate images with top-performing seeds and build a dataset
echo "Step 4: Generating dataset with top-performing seeds..."
python python_scripts/generate_dataset_with_top_seeds.py \
  --model_params "${PARAMS}" \
  --base_model "${MODEL_NAME}" \
  --push_to_hub \
  --dataset_name "${DATASET_NAME}"

# 5. (Optional) Evaluate and rectify the dataset
if [ "$RECTIFIED_DATASET" = true ]; then
    echo "Step 5: Evaluating and rectifying the dataset..."
    python python_scripts/cogvlm2_request_for_dataset_eval.py \
      --model_params "${PARAMS}"
    
    DATASET_NAME="Doub7e/${PARAMS}-rectified"
    python python_scripts/evaluate_and_rectify_dataset.py \
      --model_params "${PARAMS}" \
      --rectify \
      --push_to_hub \
      --dataset_name "${DATASET_NAME}"
else
    echo "Skipping dataset rectification (RECTIFIED_DATASET=false)"
fi

# 6. Finetune text-to-image model with the dataset
echo "Step 6: Finetuning text-to-image model..."
accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes=2 python_scripts/finetune_text_to_image.py \
  --pretrained_model_name_or_path="${MODEL_NAME}" \
  --dataset_name="${DATASET_NAME}" \
  --caption_column="${CAPTION_COLUMN}" \
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
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="${OUTPUT_NAME}/" \
  --trainable_layer="${TRAINABLE_LAYER}" \
  --trainable_attn="${TRAINABLE_ATTN}" \
  --trainable_module="${TRAINABLE_MODULE}"

# 7. Evaluate the finetuned model
echo "Step 7: Evaluating models..."

# First, evaluate the pretrained model for baseline
echo "Generating evaluation images with pretrained model..."
python python_scripts/generate_images_for_evaluation.py \
  --model_name "${PARAMS}" \
  --pipeline "${MODEL_NAME}" \
  --unet_path "" \
  --output_dir "${PRETRAINED_EVALUATION_DIR}" \

# Run GPT-4o evaluation on pretrained model images
echo "Running GPT-4o evaluation on pretrained model images..."
python python_scripts/gpt4o_request_evaluation.py \
  --input_dir "${PRETRAINED_EVALUATION_DIR}"

# Now evaluate the fine-tuned model
echo "Generating evaluation images with fine-tuned model..."
FINETUNED_UNET="${OUTPUT_NAME}/unet"
python python_scripts/generate_images_for_evaluation.py \
  --model_name "${PARAMS}" \
  --pipeline "${MODEL_NAME}" \
  --unet_path "${FINETUNED_UNET}" \
  --output_dir "${EVALUATION_DIR}" \

# Run GPT-4o evaluation on fine-tuned model images
echo "Running GPT-4o evaluation on fine-tuned model images..."
python python_scripts/gpt4o_request_evaluation.py \
  --input_dir "${EVALUATION_DIR}"

# Compare the results between pretrained and fine-tuned models
echo "Comparing evaluation results..."
python python_scripts/final_evaluation.py \
  --result_files "${PRETRAINED_EVALUATION_DIR}/evaluation_results.pkl" "${EVALUATION_DIR}/evaluation_results.pkl" \
  --labels "Pretrained" "Fine-tuned"

echo "==========================================================="
echo "Workflow completed! Results are in:"
echo "  - Finetuned model: ${OUTPUT_NAME}"
echo "  - Evaluation results: ${EVALUATION_DIR}"
echo "  - Pretrained baseline: ${PRETRAINED_EVALUATION_DIR}"
echo "==========================================================="
