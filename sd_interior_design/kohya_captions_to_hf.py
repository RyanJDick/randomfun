import argparse
import glob
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        "kohya_captions_to_hf",
        description="Convert captions generated by the kohya_ss repo to Hugging Face format.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing image files and kohya_ss-generated .caption files.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to the base directory of the dataset (where the metadata.jsonl file will be stored).",
    )
    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.input_dir, "*"))
    all_basenames = [os.path.basename(f) for f in all_files]

    # Contruct map of image names (without extensions) to image basenames.
    # E.g. { "abcd123": "abcd123.jpg", ...}
    img_map = {}
    for basename in all_basenames:
        name, ext = os.path.splitext(basename)
        if ext == ".caption":
            continue
        img_map[name] = basename

    metadata = []
    for file_name in all_files:
        basename = os.path.basename(file_name)
        name, ext = os.path.splitext(basename)
        if ext != ".caption":
            continue

        # Look up image basename. (Throws if not found.)
        img_basename = img_map[name]

        # Load caption.
        with open(file_name) as f:
            caption = f.readline()

        metadata.append(
            {
                # Path to the image file relative to base_dir.
                "file_name": os.path.relpath(
                    os.path.join(args.input_dir, img_basename), args.base_dir
                ),
                "text": caption.strip(),
            }
        )

    # Save metadata to .jsonl file.
    metadata_file = os.path.join(args.base_dir, "metadata.jsonl")
    with open(metadata_file, "w") as f:
        for image_meta in metadata:
            f.write(json.dumps(image_meta) + "\n")

    print(f"Wrote captions to '{metadata_file}'")


if __name__ == "__main__":
    main()
