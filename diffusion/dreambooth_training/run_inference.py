import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel

model_id = Path(os.environ["OUTPUT_DIR"]).absolute()
checkpoint = None  # "checkpoint-500" or None (to use latest)

from_pretrained_kwargs = {}

if checkpoint is not None:
    unet_path = model_id / checkpoint / "unet"
    if unet_path.exists():
        from_pretrained_kwargs["unet"] = UNet2DConditionModel.from_pretrained(
            unet_path, torch_dtype=torch.float16
        )
        print("Loaded unet from checkpoint.")
    else:
        print("No unet found in checkpoint.")

    text_encoder_path = model_id / checkpoint / "text_encoder"
    if text_encoder_path.exists():
        from_pretrained_kwargs["text_encoder"] = CLIPTextModel.from_pretrained(
            text_encoder_path, torch_dtype=torch.float16
        )
        print("Loaded text_encoder from checkpoint.")
    else:
        print("No text_encoder found in checkpoint.")


pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, **from_pretrained_kwargs
)
pipe.to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

output_file = "dog-bucket.png"
image.save(output_file)
print(f"Saved output image to: '{output_file}'")
