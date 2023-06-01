import argparse
import json
import os
import urllib.parse

import requests


def url_encode_string(s):
    return urllib.parse.quote(s)


def get_photo_query(query: str, page: int):
    """Query the unsplash API for a list of images that match the query string.

    Args:
        query (str): Query string.
        page (int): The page of results to get.

    Returns:
        dict: Dict of photo results.
    """
    response = requests.get(
        f"https://unsplash.com/napi/search/photos?query={url_encode_string(query)}&per_page=20&page={page}&plus=none"
    )
    # Check if the request was successful.
    response.raise_for_status()
    return json.loads(response.text)


def download_image(url, save_path):
    """Download a single image from url and save it to save_path."""
    response = requests.get(url)
    # Check if the request was successful.
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)


def download_photos(query: str, max_images: int, out_dir: str):
    """Download photos from unsplash. Photos are saved to disk and a manifest
    is returned describing the downloaded images.

    Args:
        query (str): The photo query string.
        max_images (int): The maximum number of images to download. The script
            will stop when it hits this number or has processed all available
            images.
        out_dir (str): The output directory where photos will be saved.

    Returns:
        List[dict]: A manifest describing all of the downloaded photos.
    """
    manifest = []
    total_images = 0
    cur_page = 0
    while total_images < max_images:
        cur_page += 1
        photo_list = get_photo_query(query, cur_page)["results"]

        if len(photo_list) == 0:
            print("Scraped all available images.")
            return manifest

        # TODO: Parallel async download.
        for photo in photo_list:
            if total_images >= max_images:
                return manifest

            if photo["premium"]:
                continue

            url = photo["urls"]["regular"]
            file_name = f"{photo['id']}.jpg"
            out_path = os.path.join(out_dir, file_name)
            download_image(url, out_path)
            manifest.append(
                {
                    "id": photo["id"],
                    "url": url,
                    "file_name": file_name,
                    "query": query,
                    "alt_description": photo["alt_description"],
                }
            )
            total_images += 1
            print(f"{total_images} / {max_images}: '{out_path}'")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        prog="scrape_unsplash",
        description="Scrape free images from unsplash.",
    )
    parser.add_argument("query")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="out/")
    args = parser.parse_args()

    print(f"Scraping images for query '{args.query}'...")

    # Create output directory. Throws if the directory already exists.
    os.makedirs(args.out_dir)

    manifest = download_photos(args.query, args.max_images, args.out_dir)

    # Save manifest file.
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"Saved manifest to '{manifest_path}'.")


if __name__ == "__main__":
    main()
