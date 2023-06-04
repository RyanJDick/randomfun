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
./train.sh
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


## TODO
- Larger batch size?
	- Suggestion for cloneofsimo/lora is 4: 
- Train LoRA with larger rank value
- Try to fine-tune a text-to-image model without LoRA.
- Try to train a DreamBooth model to give a name to the interior design style
- Add "interior design" to all prompts?
- Generate captions with BLIP (or WD 1.4?)
- Try to train an inpaint model instead?
- Look into training with multiple image resolutions
- Experiments
	- Show results at various checkpoints throughout the training process.
	- Effect of dataset size on behaviour.
	- Importance of captions: alt descriptions vs. BLIP-generated captions vs. fixed caption
	- Show effect of different LoRA weightings
- Training notes:
	- Could train with Kohya
