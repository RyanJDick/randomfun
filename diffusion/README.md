# Diffusion

Miscellaneous experiments with diffusion models.

## Setup

- Tested on Ubuntu `18.04`, with Python `3.8.6`.
- Install python dependencies with `pip install -r requirements.txt`

## Getting Started

### Notebooks:

- 00_getting_started.ipynb: Stable diffusion text to image demo.
- 01_paint_by_example.ipynb: PaintByExample demo.

### Training DreamBooth

DreamBooth training was done according to the instructions [here](https://huggingface.co/docs/diffusers/training/dreambooth).

Training was tested on a machine with an 8GB GPU.

Below is the rough training workflow:

0. Note that `requirements.txt` specifies source install of the `diffusers` module. This is required for DreamBooth training.
1. Enable `deepspeed` in the Accelerate environment.
2. Set training env vars.
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="output/dog"
export CLASS_DIR="output/prior_preserving_class_images"
export OUTPUT_DIR="output/dog_finetuned_model"
```
3. Download sample dog images.
```bash
cd dreambooth_training/
python download_dog_images.py
```
4. Clone the dreambooth training code.
```bash
# Tested with commit: 5c7a35a25915f29aa79e5b69d831fd0f7d7d8d41"
cd ~/src # Or wherever you want to put it.
git clone https://github.com/huggingface/diffusers.git
```
5. Apply this patch to the dreambooth training code to make it work properly with deepspeed (if it hasn't already been fixed upstream):
```diff
--- a/examples/dreambooth/train_dreambooth.py
+++ b/examples/dreambooth/train_dreambooth.py
@@ -741,8 +741,8 @@ def main(args):
                 sub_dir = "unet" if type(model) == type(unet) else "text_encoder"
                 model.save_pretrained(os.path.join(output_dir, sub_dir))
 
-                # make sure to pop weight so that corresponding model is not saved again
-                weights.pop()
+            # Clear all weights so that the corresponding model is not saved again.
+            weights.clear()
 
         def load_model_hook(models, input_dir):
             while len(models) > 0:
```
6. Run training.
```bash
accelerate launch ~/src/diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --mixed_precision=fp16 \
  --checkpointing_steps=1
```
7. Run inference with fine-tuned model.
```bash
python run_inference.py
```
