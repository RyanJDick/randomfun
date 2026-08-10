# llm-from-scratch

## Dataset

Generate the copy/reverse dataset (100k train / 10k test, sequence lengths 1–10):

```sh
uv run generate_copy_reverse_dataset.py
```

This writes `train.txt` and `test.txt`. Run with `--help` to see the available
options (split sizes, sequence length range, seed, output paths).
