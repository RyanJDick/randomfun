# sd_interior_design

This directory contains the results of an attempt to fine-tune a stable diffusion model on interior design images.

## Install Dependencies

1. Tested with `python 3.8.6`.
2. (Recommended) Set up python virtual environment.
3. `pip install -r requirements.txt`

## Training Locally

1. Download an image dataset using [this tool](../unsplash_scraper/):
```bash
python scrape_unsplash.py "living room" --out-dir out --max-images 5000
```
2. Convert the dataset to HuggingFace format:
```bash
python prepare_dataset.py --input-dir ~/src/randomfun/unsplash_scraper/out --output-dir ~/src/randomfun/sd_interior_design/living_room_dataset_v2 --caption-prefix="interior design, living room, "
```
3. Configure HuggingFace accelerate:
```bash
# (Tested with default configuration only.)
accelerate config
```
4. Run training:
```bash
./train_t2i_lora.sh
```
5. Monitor training with tensorboard:
```bash
tensorboard --logdir finetune/lora/living_room/
```

## Training on Google Colab

1. Follow step 1 and 2 from "Training Locally" to generate a dataset.
2. Zip the dataset directory:
```bash
zip -r living_room_dataset_v1.zip living_room_dataset_v1
```
3. Upload the zip file to your preferred cloud storage provider (e.g. AWS S3).
4. TODO: Add notebook and link to launch it.

## Inference

1. Use the `test.ipynb` notebook to test model inference.

## Experiment Log

### (2023-06-03) text-to-image LoRA
- Setup:
  - 5000 images
  - Descriptions from unsplash alt descriptions
  - Trained for 15k steps
- Result:
  - Tested at various checkpoints and various LoRA scales. Results were different, but not clearly better (or worse) than the baseline model.

### (2023-06-04) text-to-image LoRA, increase rank 4 -> 64
- Setup:
  - Mostly same configuration as previous experiment.
  - Increased LoRA update matrix rank from 4 to 64.
  - Ran with xformers for more efficient memory usage.
- Result:
  - Training was slightly faster with xformers.
  - Tested at various checkpoints and various LoRA scales. Results were different, but not clearly better (or worse) than the baseline model. So, rank was not the bottleneck holding back performance. I may re-visit the effet of rank once I have a higher-performing model.

### (2023-06-04) text-to-image LoRA, use keywords
- Setup:
  - Mostly the same as the previous experiment.
  - Reverted LoRA rank to 4.
  - Add the keyword "interior design" in every caption.
- Result:
  - Again, no clear benefit to the trained LoRA model.

### (2023-06-14) text-to-image finetuning (no LoRA)
- Setup:
  - 'Plain' finetuning, without LoRA.
  - Using 8-bit Adam Optimizer to get training to run on a T4 with 16GB of VRAM.
- Result:
  - Checked checkpoint after 1000 steps and 4000 steps.
  - The quality of the generated images seemed to be getting progressively worse. On both interior design images and unrelated prompts. The results were blurry and did not clearly match the prompt.

### (2023-06-14) text-to-image fintetuning w/ pokemon dataset
- Setup
  - Same configuration as previous experiment, but use the "lambdalabs/pokemon-blip-captions" dataset.
  - The goal is to replicate the pokemon training example from the docs to determine if there is an issue with the training code.
- Result:
  - The model seems to quickly experience catastrophic forgetting.
  - I tested the prompt "Yoda":
    - Before fine-tuning: Clear image of Yoda
    - After 500 steps: Pokemon style is clearly learned and you can see some Yoda-like features, but it does not look like a Yoda-pokemon.
    - After >4000 steps: The generated image seems to have the Pokemon style, but there is barely any resemblance to Yoda aside from the color selection. It looks like abstract art.

### (2023-06-15) Checkpoint Merging
- Setup
  - Try to interpolate between the trained checkpoint from the previous step and the original model to workaround the problem of catastrophic forgetting.
- Result
  - This was a helpful tool for visualizing the differences between the base model and the fine-tuned model.
  - I was still unable to produce the results reported for pokemon fine-tuning. It felt like the fine-tuned model had learned the pokemon "style", but not how to create characters.

### (2023-06-15) Pokemon Finetune
- Setup
  - Try to reproduce the reported results with this script: https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
  - Ran on a Lambda Labs VM with an A6000 GPU (48GB VRAM).
- Result
  - Progress was looking much better than the previous attempts to do this with the diffusers repo.
  - The implementations of the two repos are completely different, so it is difficult to identify the source of the discrepancy.

### (2023-06-16) Pokemon Finetune (with both ema and non-ema weights)
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

### (2023-06-29) Reproduce Pokemon finetuning for living rooms
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

## Dataset History

### living_room_dataset_v1

- 5000 living room images scraped from Unsplash.
- Unsplash alt descriptions used as captions.

### living_room_dataset_v2

- Same 5000 images as v1.
- All captions prefixed with "living room".

## TODO
- Open a PR on the Diffusers docs to share what I learned?
- Next steps:
  - Look into differences in loss between diffusers and JP's repo.
  - Try running JP's training script with non-EMA starting point.
- Try using Justin Pinkney's training code to see if I can reproduce his results. First, does he do anything differently?
  - After all of the below analysis, I found this discussion which came to the same conclusion as me about the likely source of the discrepancy: https://github.com/huggingface/diffusers/issues/1153
  - Train VAE?
    - Neither repo trains the VAE.
  - Train text encoder?
    - Neither repo trains the VAE.
  - Batch size:
    - JP's repos uses batch size of 8 across 2 GPUs. I tested it with 4 on 1 GPU and it worked fine.
    - I was using the diffusers project with 4 on 1 GPU, so this shouldn't be the source of the problem.
  - Learning rate:
    - Base LR in JP's repo is 1e-04.
    - Base LR in diffusers is 1e-05.
    - The weird thing is that the LR still feels too high in the diffusers repo. Is the loss function different?
  - Learning rate scheduler?
    - In JP's repo, the scheduler is configured in a way that effectively just makes it a constant LR.
    - Diffusers uses a constant LR.
  - Starting point
    - JP's repo explicitly uses EMA weights as a starting point.
    - In diffusers, we are not using the EMA weights (and they don't seem to be readily available.)
    - I could test this by running JP's repo with non-EMA weights as a starting point, to see how it does.
  - Refer to:
    - https://github.com/LambdaLabsML/examples/blob/main/stable-diffusion-finetuning/pokemon_finetune.ipynb
    - https://github.com/justinpinkney/stable-diffusion/blob/main/main.py
    - https://github.com/justinpinkney/stable-diffusion/blob/main/configs/stable-diffusion/pokemon.yaml
- Try training with cloneofsimo/lora?
- Try to train a DreamBooth model to give a name to the interior design style
- Generate captions with BLIP (or WD 1.4?)
- Look into training with multiple image resolutions
- Experiments
	- Show results at various checkpoints throughout the training process.
	- Effect of dataset size on behaviour.
	- Importance of captions: alt descriptions vs. BLIP-generated captions vs. fixed caption
	- Show effect of different LoRA weightings
- Training notes:
	- Could train with Kohya
- Try to train an inpaint model instead?
