# sd_interior_design

This directory contains the results of an attempt to fine-tune a stable diffusion model on interior design images.

## Steps

1. Download an image dataset using [this tool](../unsplash_scraper/):
```bash
python scrape_unsplash.py "living room" --out-dir out --max-images 5000
```
2. Convert the dataset to HuggingFace format:
```bash
python prepare_dataset.py --input-dir ~/src/randomfun/unsplash_scraper/out --output-dir ~/src/randomfun/sd_interior_design/out
```
