"""A fixed, closed-vocabulary tokenizer for the copy/reverse task.

The dataset is already whitespace-tokenized (see
`generate_copy_reverse_dataset.py`), so there is nothing to train: every token
is a single word split on spaces. We only need a stable id <-> token mapping,
plus two special tokens:

    <pad>  used to pad a batch to equal length
    <eos>  appended after each example so generation knows when to stop

The full vocabulary is 15 tokens:

    <pad> <eos> copy reverse 0 1 2 3 4 5 6 7 8 9 =
"""

# Mirror the task tokens from the dataset generator. Kept as literals (rather
# than imported) so the tokenizer stands alone.
OPERATIONS = ["copy", "reverse"]
DIGITS = [str(d) for d in range(10)]
TASK_TOKENS = OPERATIONS + DIGITS + ["="]

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"


class Tokenizer:
    def __init__(self) -> None:
        # <pad> is deliberately id 0 so padded positions read as 0.
        self.vocab = [PAD_TOKEN, EOS_TOKEN, *TASK_TOKENS]
        self.stoi = {tok: i for i, tok in enumerate(self.vocab)}
        self.itos = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, line: str, add_eos: bool = True) -> list[int]:
        """Turn a whitespace-separated example line into a list of token ids."""
        ids = [self.stoi[tok] for tok in line.split()]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Turn token ids back into a line, optionally dropping PAD/EOS."""
        special = {self.pad_id, self.eos_id}
        toks = [
            self.itos[i]
            for i in ids
            if not (strip_special and i in special)
        ]
        return " ".join(toks)
