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
    response = requests.get(url)
    # Check if the request was successful.
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)


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

    total_images = 0
    cur_page = 0
    while total_images < args.max_images:
        cur_page += 1
        photo_list = get_photo_query(args.query, cur_page)["results"]

        if len(photo_list) == 0:
            print("Scraped all available images.")
            return

        for photo in photo_list:
            if total_images >= args.max_images:
                return

            if photo["premium"]:
                continue

            url = photo["urls"]["regular"]
            out_path = os.path.join(args.out_dir, f"{photo['id']}.jpg")
            download_image(url, out_path)
            total_images += 1
            print(f"{total_images} / {args.max_images}: '{out_path}'")


if __name__ == "__main__":
    main()
