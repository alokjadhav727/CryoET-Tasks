"""
Task 2a — Download tomograms, ribosome annotations, and TopCUP weights.

Everything is fetched into the submission's shared `data/` directory (submission/data),
so the whole pipeline is self-contained and reproducible from any machine:

    submission/data/
      ├── synthetic/tomograms/<run>.zarr
      ├── synthetic/overlay/ribosome_annotations_synthetic.json
      ├── chlamydomonas/tomograms/<run>.zarr
      ├── chlamydomonas/overlay/ribosome_annotations_chlamydomonas.json
      └── weights/topcup_weights/topcup_phantom_24_tomograms.ckpt
"""

import json
from pathlib import Path

from cryoet_data_portal import Client, Tomogram
import copick
from huggingface_hub import hf_hub_download

# Submission root = parent of this script's folder (submission/task2a -> submission)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RUNS = {
    "synthetic":     {"run_id": 16463, "dataset_id": 10440, "voxel_spacing": 10.012},
    "chlamydomonas": {"run_id": 14070, "dataset_id": 10301, "voxel_spacing": 7.84},
}


def download_tomograms():
    client = Client()  # CZ cryoET Data Portal client
    for name, info in RUNS.items():
        dest = DATA_DIR / name / "tomograms"
        dest.mkdir(parents=True, exist_ok=True)
        if list(dest.glob("*.zarr")):
            print(f"Tomogram for {name} already present in {dest} — skipping.")
            continue
        print(f"Downloading tomogram for {name} (run {info['run_id']})...")
        for tomo in Tomogram.find(client, [Tomogram.tomogram_voxel_spacing.run.id == info["run_id"]]):
            if abs(tomo.voxel_spacing - info["voxel_spacing"]) < 0.1:
                tomo.download_omezarr(dest_path=str(dest))
                print(f"  Saved to {dest}")
                break


def download_annotations():
    for name, info in RUNS.items():
        overlay = DATA_DIR / name / "overlay"
        overlay.mkdir(parents=True, exist_ok=True)
        out = overlay / f"ribosome_annotations_{name}.json"
        if out.exists():
            print(f"Annotations for {name} already present — skipping.")
            continue
        print(f"Downloading annotations for {name} (dataset {info['dataset_id']})...")
        root = copick.from_czcdp_datasets(
            dataset_ids=[info["dataset_id"]],
            overlay_root=str(overlay),
            overlay_fs_args={"auto_mkdir": True},
        )
        target_run = next((r for r in root.runs if str(info["run_id"]) in r.name), None)
        if target_run is None:
            print(f"  Run {info['run_id']} not found in dataset {info['dataset_id']}")
            continue
        for pick in target_run.picks:
            if "ribosome" in pick.pickable_object_name.lower():
                pick.load()  # explicitly load from the portal
                pts = pick.points
                print(f"  {len(pts)} ribosome annotations in {target_run.name}")
                anno = {
                    "n_points": len(pts),
                    "points": [{"x": p.location.x, "y": p.location.y, "z": p.location.z} for p in pts],
                }
                with open(out, "w") as f:
                    json.dump(anno, f, indent=2)
                print(f"  Saved annotations to {out}")
                break


def download_weights():
    weights_dir = DATA_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "topcup_weights" / "topcup_phantom_24_tomograms.ckpt"
    if ckpt.exists():
        print("TopCUP weights already present — skipping.")
        return
    print("Downloading TopCUP weights from Hugging Face...")
    path = hf_hub_download(
        repo_id="kevinzhao/TopCUP",
        filename="topcup_weights/topcup_phantom_24_tomograms.ckpt",
        local_dir=str(weights_dir),
    )
    print(f"  Saved to {path}")


if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    download_tomograms()
    download_annotations()
    download_weights()
    print("\nDone. All data is in", DATA_DIR)
