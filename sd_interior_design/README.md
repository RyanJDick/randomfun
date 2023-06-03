# sd_interior_design

This directory contains the results of an attempt to fine-tune a stable diffusion model on interior design images.

## Training Locally

1. Download an image dataset using [this tool](../unsplash_scraper/):
```bash
python scrape_unsplash.py "living room" --out-dir out --max-images 5000
```
2. Convert the dataset to HuggingFace format:
```bash
python prepare_dataset.py --input-dir ~/src/randomfun/unsplash_scraper/out --output-dir ~/src/randomfun/sd_interior_design/living_room_dataset_v1
```
3. Zip the dataset directory and upload it to S3 (assuming that we will be running training in the cloud):
```bash
zip -r living_room_dataset_v1.zip living_room_dataset_v1
```

## Training on Google Colab

1. TODO

## TODO

- Add notebook for running inference on trained checkpoint.
- Try to get training to run on an 8GB GPU?
- Create a script for running locally as well as in Google Colab.
- Any speedup from xformers?
- Experiments
	- Show results at various checkpoints throughout the training process.
	- Effect of dataset size on behaviour.
	- Importance of captions: alt descriptions vs. BLIP-generated captions vs. fixed caption
	- Show effect of different LoRA weightings
