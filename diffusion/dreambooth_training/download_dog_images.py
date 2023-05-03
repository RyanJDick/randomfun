import os

from huggingface_hub import snapshot_download


def main():
    local_dir = os.environ["INSTANCE_DIR"]

    snapshot_download(
        "diffusers/dog-example",
        local_dir=local_dir,
        repo_type="dataset",
        ignore_patterns=".gitattributes",
    )

    print(f"Saved images to '{local_dir}'")


if __name__ == "__main__":
    main()
