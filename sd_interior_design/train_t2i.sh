#!/bin/bash
set -euxo pipefail

# MODEL_NAME=runwayml/stable-diffusion-v1-5
MODEL_NAME=CompVis/stable-diffusion-v1-4
OUTPUT_DIR=finetune/t2i/living_room/$(date "+%Y%m%d-%H%M%S")
DATASET_NAME=living_room_dataset_v2

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention
