#!/bin/bash
set -euxo pipefail

# MODEL_NAME=runwayml/stable-diffusion-v1-5
MODEL_NAME=CompVis/stable-diffusion-v1-4
NOM_EMA_REVISION="non-ema"
OUTPUT_DIR=/data/finetune/t2i/living_room/$(date "+%Y%m%d-%H%M%S")
DATASET_NAME=/data/living_room_dataset_v3

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --non_ema_revision=$NOM_EMA_REVISION \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention
