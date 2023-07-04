# sd_interior_design

This directory contains the results of an attempt to fine-tune a stable diffusion model on interior design images.

## Install Dependencies

1. Tested with `python 3.8.6`.
2. (Recommended) Set up python virtual environment.
3. `pip install -r requirements.txt`

## Training Locally

TODO: This section currently has instructions for running one particular fine-tuning configuration. It should be updated after more experimentation to reflect the final workflow.

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

1. Use the `test*.ipynb` notebooks to test model inference.

## Experiment Log

See the [experiment log](/EXPERIMENT_LOG.md).

## Dataset History

### living_room_dataset_v1

- 5000 living room images scraped from Unsplash.
- Unsplash alt descriptions used as captions.

### living_room_dataset_v2

- Same 5000 images as v1.
- All captions prefixed with "living room".

### living_room_dataset_v3

- Same 5000 images as v1 and v2.
- Captions were generated with the `finetune\make_captions.py` script from [kohya_ss](https://github.com/bmaltais/kohya_ss) (uses BLIP under the hood). Captions were converted to Hugging Face format with `kohya_captions_to_hf.py`. The final dataset has both Hugging Face captions (in `metadata.jsonl`) and kohya_ss captions (in `.caption` files).

## TODO
- Increase dataset size?
- Open a PR on the Diffusers docs to share what I learned?
- Next steps:
  - Look into differences in loss between diffusers and JP's repo.
  - Try running JP's training script with non-EMA starting point.
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
