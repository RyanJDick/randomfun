# Experiment Log

## (2023-06-03) text-to-image LoRA
- Setup:
  - 5000 images
  - Descriptions from unsplash alt descriptions
  - Trained for 15k steps
- Result:
  - Tested at various checkpoints and various LoRA scales. Results were different, but not clearly better (or worse) than the baseline model.

## (2023-06-04) text-to-image LoRA, increase rank 4 -> 64
- Setup:
  - Mostly same configuration as previous experiment.
  - Increased LoRA update matrix rank from 4 to 64.
  - Ran with xformers for more efficient memory usage.
- Result:
  - Training was slightly faster with xformers.
  - Tested at various checkpoints and various LoRA scales. Results were different, but not clearly better (or worse) than the baseline model. So, rank was not the bottleneck holding back performance. I may re-visit the effet of rank once I have a higher-performing model.

## (2023-06-04) text-to-image LoRA, use keywords
- Setup:
  - Mostly the same as the previous experiment.
  - Reverted LoRA rank to 4.
  - Add the keyword "interior design" in every caption.
- Result:
  - Again, no clear benefit to the trained LoRA model.

## (2023-06-14) text-to-image finetuning (no LoRA)
- Setup:
  - 'Plain' finetuning, without LoRA.
  - Using 8-bit Adam Optimizer to get training to run on a T4 with 16GB of VRAM.
- Result:
  - Checked checkpoint after 1000 steps and 4000 steps.
  - The quality of the generated images seemed to be getting progressively worse. On both interior design images and unrelated prompts. The results were blurry and did not clearly match the prompt.

## (2023-06-14) text-to-image fintetuning w/ pokemon dataset
- Setup
  - Same configuration as previous experiment, but use the "lambdalabs/pokemon-blip-captions" dataset.
  - The goal is to replicate the pokemon training example from the docs to determine if there is an issue with the training code.
- Result:
  - The model seems to quickly experience catastrophic forgetting.
  - I tested the prompt "Yoda":
    - Before fine-tuning: Clear image of Yoda
    - After 500 steps: Pokemon style is clearly learned and you can see some Yoda-like features, but it does not look like a Yoda-pokemon.
    - After >4000 steps: The generated image seems to have the Pokemon style, but there is barely any resemblance to Yoda aside from the color selection. It looks like abstract art.

## (2023-06-15) Checkpoint Merging
- Setup
  - Try to interpolate between the trained checkpoint from the previous step and the original model to workaround the problem of catastrophic forgetting.
- Result
  - This was a helpful tool for visualizing the differences between the base model and the fine-tuned model.
  - I was still unable to produce the results reported for pokemon fine-tuning. It felt like the fine-tuned model had learned the pokemon "style", but not how to create characters.

## (2023-06-15) Pokemon Finetune
- Setup
  - Try to reproduce the reported results with this script: https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
  - Ran on a Lambda Labs VM with an A6000 GPU (48GB VRAM).
- Result
  - Progress was looking much better than the previous attempts to do this with the diffusers repo.
  - The implementations of the two repos are completely different, so it is difficult to identify the source of the discrepancy.

## (2023-06-16) Pokemon Finetune (with both ema and non-ema weights)
- Setup
  - Use the NON_EMA_REVISION from here: https://github.com/huggingface/diffusers/issues/1153#issuecomment-1368087432
  - See the rest of that Github discussion for an explanation why. I suspect that this is currently the main difference between training with the CopmVis repo and the diffusers repo.
  - Full config:
```bash
MODEL_NAME=CompVis/stable-diffusion-v1-4
NOM_EMA_REVISION=non-ema
OUTPUT_DIR=/home/ubuntu/data/finetune/t2i/living_room/$(date "+%Y%m%d-%H%M%S")
DATASET_NAME=lambdalabs/pokemon-blip-captions

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --non_ema_revision=$NOM_EMA_REVISION \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention
```
- Result
  - It worked! Results looked roughly comparable to those reported in the original blog post.
  - Looks like starting from the EMA weights makes a big difference.

## (2023-06-29) Reproduce Pokemon finetuning for living rooms
- Setup
  - Same as previous experiment, but with the interior design dataset.
  - Full config:
```bash
MODEL_NAME=CompVis/stable-diffusion-v1-4
NOM_EMA_REVISION="non-ema"
OUTPUT_DIR=/home/ubuntu/data1/data/finetune/t2i/living_room/$(date "+%Y%m%d-%H%M%S")
DATASET_NAME=/home/ubuntu/data1/data/living_room_dataset_v2

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --non_ema_revision=$NOM_EMA_REVISION \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention \
  --validation_prompts "A modern living room with a brown couch" "A rustic living room"
```
- Result
  - The images generated by the trained model were different than the baseline, but not clearly better. Nonetheless, this was an improvement over previous fine-tuning attempts that produced clearly worse results.
  - The model did seem to get worse at generating images unrelated to interior design.

## (2023-07-04) kohya_ss (BLIP) captions
- Setup
  - Same as previous experiment, but using captions generated with the kohya_ss repo (uses BLIP).
  - Full config:
```bash
MODEL_NAME=CompVis/stable-diffusion-v1-4
NOM_EMA_REVISION="non-ema"
OUTPUT_DIR=/home/ubuntu/data1/data/finetune/t2i/living_room/$(date "+%Y%m%d-%H%M%S")
DATASET_NAME=/home/ubuntu/data1/data/living_room_dataset_v3

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --non_ema_revision=$NOM_EMA_REVISION \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --report_to=tensorboard \
  --checkpointing_steps=500 \
  --enable_xformers_memory_efficient_attention
```
- Result
  - Very similar result to previous training attempt. Changing the captions didn't seem to have much of an effect.
  - The finetuned model alone isn't great. The version merged with the baseline is better.
