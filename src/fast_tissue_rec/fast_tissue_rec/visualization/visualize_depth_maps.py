import argparse
from collections.abc import Sequence

import matplotlib

matplotlib.use("tkagg")  # See https://github.com/matplotlib/matplotlib/issues/18406
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

import fast_tissue_rec.data_utils as data_utils


def save_depth_image_visualizations(
    output_dir: Path, depth_maps: Sequence[npt.NDArray[np.float32]]
):
    depth_stack = np.array(depth_maps)
    min_depth, max_depth = np.min(depth_stack[depth_stack > 0.0]), np.max(depth_maps)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving depth image visualizations...")
    cmap = plt.get_cmap("plasma").copy()
    cmap.set_under(color="black")

    for i, depth in enumerate(tqdm(depth_maps)):
        _, ax = plt.subplots(layout="constrained")
        img = ax.imshow(depth, cmap=cmap, vmin=min_depth, vmax=max_depth)
        plt.colorbar(img, ax=ax, shrink=0.95)
        ax.set_axis_off()
        # remove padding margins
        plt.savefig(
            str(output_dir / f"depth_map_{i:04d}.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth-index", help="Depth map index (JSON)", required=True
    )
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    print("Loading depth maps...")
    depth_maps = data_utils.load_depth_maps(Path(args.depth_index))
    save_depth_image_visualizations(Path(args.output_dir), depth_maps)
