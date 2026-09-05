"""Dataset and batching for the copy/reverse task.

`CopyReverseDataset` reads a generated `.txt` split (one example per line) and
returns each example as a 1-D tensor of token ids. `make_collate_fn` pads a
batch to equal length and builds the tensors a training step needs.

Label convention: `labels` is token-aligned with `input_ids` (same shape, no
shifting here). The next-token shift is the training step's job, i.e. compare
`logits[:, :-1]` against `labels[:, 1:]`. Positions set to `IGNORE_INDEX` are
excluded from the loss (this matches `nn.CrossEntropyLoss(ignore_index=-100)`).
"""

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer

# Matches the default `ignore_index` of torch's cross-entropy loss.
IGNORE_INDEX = -100


class CopyReverseDataset(Dataset):
    def __init__(self, path: str, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        with open(path) as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.lines[idx])
        return torch.tensor(ids, dtype=torch.long)


def make_collate_fn(pad_id: int, eq_id: int, mask_prompt: bool = False):
    """Build a collate_fn that right-pads a batch and builds labels.

    Returns a function mapping a list of 1-D id tensors to a dict of three
    `[B, T]` tensors: `input_ids`, `attention_mask`, and `labels`.

    - `input_ids`   : token ids, right-padded with `pad_id`.
    - `attention_mask`: 1 for real tokens, 0 for padding.
    - `labels`      : copy of `input_ids` with padding set to `IGNORE_INDEX`.
                      If `mask_prompt`, every position up to and including the
                      `=` token (id `eq_id`) is also set to `IGNORE_INDEX`, so
                      loss is computed only on the answer + EOS.
    """

    def collate_fn(batch: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        max_len = max(seq.size(0) for seq in batch)
        b = len(batch)

        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        for i, seq in enumerate(batch):
            n = seq.size(0)
            input_ids[i, :n] = seq
            attention_mask[i, :n] = 1

        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX

        if mask_prompt:
            for i, seq in enumerate(batch):
                # Positions 0..eq (inclusive) are the prompt "op ... =".
                eq_positions = (seq == eq_id).nonzero(as_tuple=True)[0]
                eq = int(eq_positions[0])
                labels[i, : eq + 1] = IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn
