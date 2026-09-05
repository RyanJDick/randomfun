"""Generate a copy/reverse dataset for training a small transformer from scratch.

The task is one of two operations applied to a short sequence of digits:

    copy:    3 9 2 1        = 3 9 2 1
    reverse: 1 2 9 5 8 9 0  = 0 9 8 5 9 2 1

Vocabulary (the only tokens that ever appear):

    copy  reverse  0 1 2 3 4 5 6 7 8 9  <space>  =

Each example is written on a single line as space-separated tokens, e.g.

    reverse 1 2 9 5 8 9 0 = 0 9 8 5 9 2 1

Each split is deduplicated internally, then train and test are sampled
independently. We deliberately do NOT force the splits to be disjoint: the
short-sequence buckets are tiny (length 1 has only 2 * 10 = 20 possible
examples), so holding them out of train would mean the model never sees a
length-1 sequence. Overlap is therefore expected on short sequences and
negligible on long ones; the actual overlap is reported at the end.
"""

import argparse
import random

# The full, closed vocabulary for this task. Digits are single tokens; the
# operations and "=" are their own tokens; " " separates tokens on a line.
DIGITS = [str(d) for d in range(10)]
OPERATIONS = ["copy", "reverse"]
VOCAB = OPERATIONS + DIGITS + ["="]


def make_example(rng: random.Random, min_len: int, max_len: int) -> str:
    """Sample a single "<op> <seq> = <result>" example as a single line."""
    op = rng.choice(OPERATIONS)
    length = rng.randint(min_len, max_len)
    seq = [rng.choice(DIGITS) for _ in range(length)]
    result = seq if op == "copy" else seq[::-1]
    return " ".join([op, *seq, "=", *result])


def generate(
    n: int,
    rng: random.Random,
    min_len: int,
    max_len: int,
) -> list[str]:
    """Sample n examples, unique within this split, in generation order."""
    examples: list[str] = []
    seen: set[str] = set()
    while len(examples) < n:
        example = make_example(rng, min_len, max_len)
        if example in seen:
            continue
        seen.add(example)
        examples.append(example)
    return examples


def write_split(path: str, examples: list[str]) -> None:
    with open(path, "w") as f:
        f.writelines(example + "\n" for example in examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-out", default="train.txt")
    parser.add_argument("--test-out", default="test.txt")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    train = generate(args.train_size, rng, args.min_len, args.max_len)
    test = generate(args.test_size, rng, args.min_len, args.max_len)

    write_split(args.train_out, train)
    write_split(args.test_out, test)

    overlap = len(set(train) & set(test))
    print(f"vocab ({len(VOCAB)} tokens): {' '.join(VOCAB)}")
    print(f"wrote {len(train):,} train examples -> {args.train_out}")
    print(f"wrote {len(test):,} test examples  -> {args.test_out}")
    print(
        f"train/test overlap: {overlap:,} examples "
        f"({overlap / len(test):.1%} of test; expected on short sequences)"
    )


if __name__ == "__main__":
    main()
