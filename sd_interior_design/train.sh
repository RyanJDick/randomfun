#!/bin/bash
set -euxo pipefail

MODEL_NAME=runwayml/stable-diffusion-v1-5
OUTPUT_DIR=finetune/lora/living_room
DATASET_NAME=living_room_dataset_v1

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --validation_prompt="A modern living room with a white couch." \
  --seed=1337 \
  --lora_matrix_rank=64 \
  --enable_xformers_memory_efficient_attention
