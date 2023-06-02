import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        "prepare_dataset",
        description="Convert a directory of images downloaded with the unsplash_scraper to a HuggingFace-compatible dataset.",
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Load manifest file.
    with open(os.path.join(args.input_dir, "manifest.json")) as f:
        manifest = json.load(f)

    os.makedirs(args.output_dir)

    metadata = []
    for i, image_info in enumerate(manifest):
        # Copy image from input directory to output directory.
        src_path = os.path.join(args.input_dir, image_info["file_name"])
        dst_path = os.path.join(
            args.output_dir,
            "train",  # All images are in the 'train' split.
            image_info["query"].replace(" ", "_"),
            image_info["file_name"],
        )
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

        # Populate metadata with image descriptions.
        metadata.append([image_info["file_name"], image_info["alt_description"]])

        if i % 100 == 0:
            print(f"{i+1} / {len(manifest)}")

    # Save metadata to .csv file.
    with open(os.path.join(args.output_dir, "metadata.csv"), "w") as f:
        f.write("file_name, text\n")
        for image_meta in metadata:
            f.write(f"{image_meta[0]}, {image_meta[1]}\n")

    print(f"Prepared dataset at '{args.output_dir}'")


if __name__ == "__main__":
    main()
