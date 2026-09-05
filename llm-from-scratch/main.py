"""Training-loop skeleton for the copy/reverse task.

There is no model yet: this stands up and exercises the data pipeline
(tokenizer -> dataset -> padded batches -> labels) so it can be verified in
isolation. It prints diagnostics and runs the epoch/batch loops with a
placeholder where the model forward/backward will go.

Run:  uv run main.py
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from data import IGNORE_INDEX, CopyReverseDataset, make_collate_fn
from tokenizer import Tokenizer


def roundtrip_check(
    tokenizer: Tokenizer, dataset: CopyReverseDataset, n: int = 5
) -> None:
    """Assert decode(encode(line)) == line for the first n examples."""
    for line in dataset.lines[:n]:
        decoded = tokenizer.decode(tokenizer.encode(line))
        assert decoded == line, f"round-trip mismatch:\n  in:  {line}\n  out: {decoded}"
    print(f"round-trip OK on first {n} examples")


def show_batch(tokenizer: Tokenizer, batch: dict[str, torch.Tensor]) -> None:
    """Print shapes and a tokens-vs-labels view of the first row."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print(f"input_ids:      {tuple(input_ids.shape)} {input_ids.dtype}")
    print(f"attention_mask: {tuple(batch['attention_mask'].shape)}")
    print(f"labels:         {tuple(labels.shape)}")

    row_ids, row_labels = input_ids[0].tolist(), labels[0].tolist()
    print(f"\nexample decoded: {tokenizer.decode(row_ids)}")
    print(f"{'token':>8} | {'label':>8}")
    print(f"{'-' * 8}-+-{'-' * 8}")
    for tid, lab in zip(row_ids, row_labels):
        tok = tokenizer.itos[tid]
        lab_str = "IGNORE" if lab == IGNORE_INDEX else tokenizer.itos[lab]
        print(f"{tok:>8} | {lab_str:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="train.txt")
    parser.add_argument("--test", default="test.txt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--mask-prompt",
        action="store_true",
        help="compute loss only on the answer (after '=')",
    )
    args = parser.parse_args()

    tokenizer = Tokenizer()
    print(f"vocab ({tokenizer.vocab_size} tokens): {' '.join(tokenizer.vocab)}")

    train_ds = CopyReverseDataset(args.train, tokenizer)
    test_ds = CopyReverseDataset(args.test, tokenizer)
    print(f"train: {len(train_ds):,} examples | test: {len(test_ds):,} examples")
    roundtrip_check(tokenizer, train_ds)

    collate_fn = make_collate_fn(
        pad_id=tokenizer.pad_id,
        eq_id=tokenizer.stoi["="],
        mask_prompt=args.mask_prompt,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    print(
        f"batches/epoch: train {len(train_loader):,} | test {len(test_loader):,} "
        f"(batch_size={args.batch_size}, mask_prompt={args.mask_prompt})"
    )

    print("\n--- first train batch ---")
    show_batch(tokenizer, next(iter(train_loader)))

    print("\n--- training loop (no model yet) ---")
    for epoch in range(args.epochs):
        t0 = time.time()
        n_tokens = 0
        for batch in train_loader:
            n_tokens += int(batch["attention_mask"].sum())
            # TODO: model forward + loss + backward + optimizer step
        dt = time.time() - t0
        print(
            f"epoch {epoch}: iterated {len(train_loader):,} batches, "
            f"{n_tokens:,} real tokens in {dt:.2f}s"
        )


if __name__ == "__main__":
    main()
