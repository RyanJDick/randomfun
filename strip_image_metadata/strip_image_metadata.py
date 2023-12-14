import argparse
import os
import time
from pathlib import Path

from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        prog="strip_image_metadata",
        description="Strip all image metadata from a directory of images.",
    )
    parser.add_argument("source")
    args = parser.parse_args()

    source = Path(args.source)
    print(f"Stripping images in '{source}'.")

    if not source.is_dir():
        raise ValueError(f"'{args.source}' is not a directory.")

    image_extensions = (".png", ".PNG", ".jpg", ".JPG")

    out_dir = Path(f"out_{str(time.time()).replace('.', '-')}")
    os.mkdir(out_dir)

    for src_path in source.glob("*"):
        if src_path.suffix not in image_extensions:
            print(f"Skipping '{src_path}'.")
            continue

        dst_path = out_dir / Path(src_path).name

        print(f"Processing '{src_path}' -> '{dst_path}'.")

        # A PIL -> numpy -> PIL roundtrip strips all metadata.
        # TODO: This is insanely slow. I'm sure there's a much better way to do
        # this.
        pil_image = Image.open(src_path)
        np_img = np.array(pil_image)
        anonymized_pil_img = Image.fromarray(np_img)
        anonymized_pil_img.save(dst_path)


if __name__ == "__main__":
    main()
