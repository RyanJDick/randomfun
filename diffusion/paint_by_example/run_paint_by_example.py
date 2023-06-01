"""The main purpose for this script is to run PaintByExample so that I can easily step
through it with a debugger to understand how it works.
"""

from io import BytesIO

import matplotlib.pyplot as plt
import PIL
import requests
import torch
from diffusers import DiffusionPipeline


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


def main():
    # img_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
    # mask_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"
    # example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

    # init_image = download_image(img_url).resize((512, 512))
    # mask_image = download_image(mask_url).resize((512, 512))
    # example_image = download_image(example_url).resize((512, 512))

    # Big stormtrooper in living room.
    # init_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/living_room.png")
    # mask_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/mask_big.png")
    # example_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/stormtrooper.png")

    # Small stormtrooper in living room.
    # init_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/living_room.png")
    # mask_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/mask.png")
    # example_image = PIL.Image.open("/home/ryan/Downloads/pbe_examples/stormtrooper_in_living_room/stormtrooper.png")

    # Stormtrooper on truck with person.
    init_image = PIL.Image.open(
        "/home/ryan/Downloads/pbe_examples/person_on_truck/person_on_truck.png"
    )
    mask_image = PIL.Image.open(
        "/home/ryan/Downloads/pbe_examples/person_on_truck/mask.png"
    )
    example_image = PIL.Image.open(
        "/home/ryan/Downloads/pbe_examples/person_on_truck/stormtrooper.png"
    )

    init_image = init_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    example_image = example_image.resize((512, 512))

    pipe = DiffusionPipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    image = pipe(
        image=init_image,
        mask_image=mask_image,
        example_image=example_image,
        guidance_scale=10,
    ).images[0]

    image.save("out.png")


if __name__ == "__main__":
    main()
