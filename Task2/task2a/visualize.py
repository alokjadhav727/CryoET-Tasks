"""
Task 2a — Visualize tomogram slices with ribosome annotations, side by side.
Saves one PNG per slice (raw | annotated) into submission/task2a/visualizations/.
"""

import json
import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # submission/
DATA_DIR = ROOT / "data"
OUT_DIR  = Path(__file__).resolve().parent / "visualizations"

RUNS = {
    "synthetic":     {"voxel_spacing": 10.012, "radius_ang": 150.0},
    "chlamydomonas": {"voxel_spacing": 7.84,   "radius_ang": 150.0},
}


def load_tomogram(run_name):
    tomo_dir = DATA_DIR / run_name / "tomograms"
    zarr_paths = list(tomo_dir.rglob("*.zarr"))
    if not zarr_paths:
        raise FileNotFoundError(f"No .zarr tomogram in {tomo_dir} — run download.py first.")
    store = zarr.open(str(zarr_paths[0]), mode="r")
    arr = store["0"] if "0" in store else store
    print(f"  {run_name} shape: {arr.shape}")
    return arr  # lazy zarr array


def load_annotations(run_name, voxel_spacing):
    anno_file = DATA_DIR / run_name / "overlay" / f"ribosome_annotations_{run_name}.json"
    if not anno_file.exists():
        print(f"  No annotation file for {run_name}")
        return np.empty((0, 3))
    with open(anno_file) as f:
        data = json.load(f)
    print(f"  {run_name} annotations: {data['n_points']}")
    return np.array([[p["z"]/voxel_spacing, p["y"]/voxel_spacing, p["x"]/voxel_spacing]
                     for p in data["points"]])


def visualize_run(run_name):
    info = RUNS[run_name]
    voxel = info["voxel_spacing"]
    radius_vox = info["radius_ang"] / voxel

    print(f"\nVisualizing {run_name}...")
    volume = load_tomogram(run_name)
    pts = load_annotations(run_name, voxel)

    out = OUT_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)

    n_slices = volume.shape[0]
    for z in range(n_slices):
        raw = np.asarray(volume[z])
        vmin, vmax = np.percentile(raw, 1), np.percentile(raw, 99)

        nearby = pts[np.abs(pts[:, 0] - z) <= 1] if len(pts) else np.empty((0, 3))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"{run_name}  |  slice {z:04d}  (raw)")
        axes[0].axis("off")

        axes[1].imshow(raw, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"slice {z:04d}  (annotated, {len(nearby)} ribosomes)")
        axes[1].axis("off")
        for pt in nearby:
            axes[1].add_patch(patches.Circle((pt[2], pt[1]), radius=radius_vox,
                                             linewidth=1.2, edgecolor="red", facecolor="none"))

        plt.tight_layout()
        plt.savefig(out / f"slice_{z:04d}.png", dpi=100)
        plt.close()
        if z % 25 == 0 or len(nearby):
            print(f"  saved slice {z:04d}/{n_slices-1}  ({len(nearby)} ribosomes)")


if __name__ == "__main__":
    for run in RUNS:
        visualize_run(run)
    print("\nDone. Images saved under", OUT_DIR)
