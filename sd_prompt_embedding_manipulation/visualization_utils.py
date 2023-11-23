import math

import matplotlib.pyplot as plt
import numpy as np


def plot_images_with_titles(images: list[np.ndarray], titles: list[str], cols: int = 3):
    """Display a grid of images with titles."""
    rows = math.ceil(len(images) / cols)

    fig, axs = plt.subplots(rows, cols)

    # Hide axes for all sub-plots.
    for r in range(rows):
        for c in range(cols):
            axs[r, c].set_axis_off()

    # Add images/titles to the grid.
    for idx, (image, title) in enumerate(zip(images, titles, strict=True)):
        r = int(idx / cols)
        c = idx % cols
        axs[r, c].set_title(title)
        axs[r, c].imshow(image)

    fig.tight_layout()
    plt.show()


def plot_image_grid(
    images: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    figsize: tuple[int] | None = None,
):
    """Display a grid of images with row labels and column labels."""
    num_rows, num_cols = len(row_labels), len(col_labels)
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True
    )

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            # plt.subplots handles dimensions equal to 1 weirdly:
            if num_rows == 1:
                ax = axs[col_idx]
            elif num_cols == 1:
                ax = axs[row_idx]
            else:
                ax = axs[row_idx, col_idx]

            ax.imshow(images[row_idx][col_idx])
            ax.set_title(col_label if row_idx == 0 else "")
            ax.set_ylabel(row_label if col_idx == 0 else "")
            ax.set_xticks([])  # Hide x-axis ticks.
            ax.set_yticks([])  # Hide y-axis ticks.

    fig.tight_layout()
    plt.show()
