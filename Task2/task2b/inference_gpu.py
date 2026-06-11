"""
Task 2b — TopCUP inference on the synthetic tomogram + quality metrics (GPU-only).

This is a stripped-down variant of inference.py for environments where an NVIDIA
GPU is guaranteed (e.g. a cluster). It removes the entire CPU-compatibility layer
(Lightning->CPU patching, .cuda()/.to('cuda') no-ops, CUDA-autocast disabling) that
the portable inference.py carries. What remains:

  - the torch.load(weights_only=False) checkpoint compat fix (needed on GPU too,
    PyTorch >=2.6)
  - a hard GPU assertion so the script fails fast instead of mis-running
  - the copick config, metrics, and the radius + threshold sweeps (unchanged)

No subprocess wrapper: the portable inference.py spawns a child process so it can
apply the CPU monkey-patches in a clean interpreter before torch imports. With the
CPU layer gone there is nothing to patch, so TopCUP's CLI is called in-process here.
This assumes Linux-style `fork` DataLoader workers (the cluster default); they share
memory with the parent and do NOT re-import this module, so the guard re-run problem
the wrapper avoided does not apply.

Hyperparameters (see TASK2B_QUALITY_ASSESSMENT.md for justification):
  RADIUS          = 15 vox (150 Å)  — one ribosome radius; CZII matching criterion
  score_threshold = 0.19 (ribosome) — the model's own trained per-class threshold
  tomo_type       = denoised        — matches the reconstruction the model trained on
  class config    = full 6-class    — so "ribosome" reads output channel 4
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT     = Path(__file__).resolve().parents[1]      # submission/
DATA_DIR = ROOT / "data"
CKPT     = DATA_DIR / "weights/topcup_weights/topcup_phantom_24_tomograms.ckpt"
RESULTS  = Path(__file__).resolve().parent / "results"

RUN_NAME = "TS_5_4"
VS       = 10.012   # voxel spacing in Å
RADIUS   = 15.0     # matching radius = 150 Å / 10.012 ≈ 15 voxels (1 ribosome radius)


def _synthetic_tomo_zarr():
    """Locate the synthetic tomogram zarr (downloaded by task2a) by globbing."""
    zs = list((DATA_DIR / "synthetic/tomograms").rglob("*.zarr"))
    if not zs:
        raise FileNotFoundError(
            f"No synthetic tomogram .zarr under {DATA_DIR/'synthetic/tomograms'} — run task2a/download.py first.")
    return zs[0].resolve()


def make_copick_config():
    """Build a copick filesystem config pointing at the downloaded synthetic data."""
    static = DATA_DIR / "synthetic/static"
    link_dir = static / f"ExperimentRuns/{RUN_NAME}/VoxelSpacing{VS}"
    link_dir.mkdir(parents=True, exist_ok=True)

    link = link_dir / "denoised.zarr"
    actual = _synthetic_tomo_zarr()
    if not link.exists() and not link.is_symlink():
        link.symlink_to(actual)
        print(f"  symlink: denoised.zarr → {actual}")

    config = {
        "config_type": "filesystem",
        "name": "synthetic",
        "version": "0.5.0",
        # Replicate the phantom model's 6-class TRAINING order so each object maps to the
        # correct output channel. Channels are label order 1..6; "ribosome" is channel 4.
        # (label=1 would make TopCUP read channel 1 = apo-ferritin and mislabel it.)
        "pickable_objects": [
            {"name": "apo-ferritin",        "is_particle": True, "label": 1, "radius": 60.0,
             "metadata": {"score_threshold": 0.16, "score_weight": 1}},
            {"name": "beta-amylase",        "is_particle": True, "label": 2, "radius": 65.0,
             "metadata": {"score_threshold": 0.25, "score_weight": 0}},
            {"name": "beta-galactosidase",  "is_particle": True, "label": 3, "radius": 90.0,
             "metadata": {"score_threshold": 0.13, "score_weight": 2}},
            {"name": "ribosome",            "is_particle": True, "label": 4, "radius": 150.0,
             "metadata": {"score_threshold": 0.19, "score_weight": 1}},
            {"name": "thyroglobulin",       "is_particle": True, "label": 5, "radius": 130.0,
             "metadata": {"score_threshold": 0.18, "score_weight": 2}},
            {"name": "virus-like-particle", "is_particle": True, "label": 6, "radius": 135.0,
             "metadata": {"score_threshold": 0.50, "score_weight": 1}},
        ],
        "overlay_root": str((DATA_DIR / "synthetic/overlay").resolve()),
        "overlay_fs_args": {"auto_mkdir": True},
        "static_root": str(static.resolve()),
    }
    cfg = DATA_DIR / "configs/copick_synthetic.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    json.dump(config, open(cfg, "w"), indent=2)
    print(f"  Config: {cfg}")
    return cfg


def run_inference(cfg, out_dir):
    """
    Run TopCUP inference natively on GPU, in-process (no subprocess wrapper).

    TopCUP's CLI reads its arguments from sys.argv, so we set them directly and call
    cli(standalone_mode=False) — standalone_mode=False stops click from calling
    sys.exit() on completion, which would otherwise kill this script before metrics.
    """
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects this older
    # checkpoint. Override it. (Required on GPU too — not a CPU-specific workaround.)
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

    sys.argv = [
        "topcup", "inference",
        "--copick_config",      str(cfg),
        "--run_names",          RUN_NAME,
        "--pretrained_weights", str(CKPT.resolve()),
        "--pixelsize",          str(VS),
        "--tomo_type",          "denoised",
        "--user_id",            "topcup",
        "--output_dir",         str(out_dir.resolve()),
        "--gpus",               "1",
        "--has_ground_truth",   "False",
    ]
    print("  Running TopCUP on GPU...")
    try:
        from topcup.cli.cli import cli
        cli(standalone_mode=False)
    finally:
        torch.load = _orig_load   # restore, so the metrics code below is unaffected


def load_predictions(out_dir):
    import pandas as pd
    csvs = list(out_dir.glob("*.csv"))
    if not csvs:
        print("  No CSV found — inference may have failed")
        return np.zeros((0, 3))
    df = pd.concat([pd.read_csv(f) for f in csvs])
    df = df[df["particle_type"] == "ribosome"]
    coords = np.column_stack([df["z"] / VS, df["y"] / VS, df["x"] / VS]) if len(df) else np.zeros((0, 3))
    print(f"  {len(coords)} predictions")
    return coords


def load_predictions_df(out_dir):
    """Load all ribosome predictions with confidence scores for post-hoc threshold sweeps."""
    import pandas as pd
    csvs = list(out_dir.glob("*.csv"))
    if not csvs:
        return pd.DataFrame(columns=["z", "y", "x", "conf"])
    df = pd.concat([pd.read_csv(f) for f in csvs])
    return df[df["particle_type"] == "ribosome"].reset_index(drop=True)


def load_ground_truth():
    with open(DATA_DIR / "synthetic/overlay/ribosome_annotations_synthetic.json") as f:
        d = json.load(f)
    pts = np.array([[p["z"] / VS, p["y"] / VS, p["x"] / VS] for p in d["points"]])
    print(f"  {len(pts)} ground truth ribosomes")
    return pts


def compute_metrics(pred, gt):
    """Precision / Recall / F1 (detection) + localization error (spatial accuracy)."""
    if not len(pred) or not len(gt):
        return {"precision": 0, "recall": 0, "f1": 0,
                "loc_err_vox": float("inf"), "loc_err_ang": float("inf"),
                "tp": 0, "fp": len(pred), "fn": len(gt)}
    D = cdist(pred, gt)
    tp_p = np.min(D, axis=1) < RADIUS
    tp_g = np.min(D, axis=0) < RADIUS
    TP, FP, FN = int(tp_p.sum()), int((~tp_p).sum()), int((~tp_g).sum())
    pr = TP/(TP+FP) if TP+FP else 0
    re = TP/(TP+FN) if TP+FN else 0
    f1 = 2*pr*re/(pr+re) if pr+re else 0
    loc = float(np.mean(np.min(D, axis=1)[tp_p])) if TP else float("inf")
    return {"precision": pr, "recall": re, "f1": f1,
            "loc_err_vox": loc, "loc_err_ang": loc*VS,
            "tp": TP, "fp": FP, "fn": FN}


if __name__ == "__main__":
    # Fail fast: this script requires a GPU. Better a loud error than a silent mis-run.
    import torch
    assert torch.cuda.is_available(), \
        "No CUDA GPU found — inference_gpu.py requires a GPU. Use inference.py for CPU."
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("1. Creating copick config...")
    cfg = make_copick_config()

    print("\n2. Running TopCUP inference...")
    out_dir = RESULTS / "synthetic"
    run_inference(cfg, out_dir)

    print("\n3. Loading predictions and ground truth...")
    pred = load_predictions(out_dir)
    gt = load_ground_truth()

    print("\n4. Metrics at standard radius (15 vox = 150 Å = 1 ribosome radius):")
    m = compute_metrics(pred, gt)
    print(f"   Precision : {m['precision']:.3f}")
    print(f"   Recall    : {m['recall']:.3f}")
    print(f"   F1        : {m['f1']:.3f}")
    print(f"   Loc error : {m['loc_err_vox']:.1f} vox  ({m['loc_err_ang']:.0f} Å)")
    print(f"   TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    print("\n   Radius sweep:")
    D = cdist(pred, gt) if len(pred) and len(gt) else None
    for r in [15, 30, 45, 60]:
        if D is not None:
            tp = int((np.min(D, axis=1) < r).sum())
            fn = int((np.min(D, axis=0) >= r).sum())
            fp = int((np.min(D, axis=1) >= r).sum())
            pr = tp/(tp+fp) if tp+fp else 0
            re = tp/(tp+fn) if tp+fn else 0
            f1 = 2*pr*re/(pr+re) if pr+re else 0
            print(f"   r={r:2d} vox ({r*VS:.0f} Å): TP={tp} FP={fp} FN={fn}  P={pr:.2f} R={re:.2f} F1={f1:.2f}")

    print("\n   Threshold sweep (fixed radius=15 vox, varying confidence cutoff):")
    ribo_df = load_predictions_df(out_dir)
    if len(ribo_df) and len(gt):
        print(f"   {'Threshold':>10} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6} {'Rec':>6} {'F1':>6}  Loc(Å)")
        for t in [0.05, 0.10, 0.15, 0.19, 0.25, 0.30, 0.40, 0.50]:
            picks = ribo_df[ribo_df["conf"] >= t]
            if len(picks) == 0:
                print(f"   {t:>10.2f}  {'0':>4} {'--':>4} {len(gt):>4}  {'0.000':>6} {'0.000':>6} {'0.000':>6}  -")
                continue
            p = np.column_stack([picks["z"]/VS, picks["y"]/VS, picks["x"]/VS])
            Dt = cdist(p, gt)
            tp_p = np.min(Dt, axis=1) < RADIUS
            tp_g = np.min(Dt, axis=0) < RADIUS
            TP = int(tp_p.sum()); FP = int((~tp_p).sum()); FN = int((~tp_g).sum())
            pr = TP/(TP+FP) if TP+FP else 0
            re = TP/(TP+FN) if TP+FN else 0
            f1 = 2*pr*re/(pr+re) if pr+re else 0
            loc = float(np.mean(np.min(Dt, axis=1)[tp_p])) * VS if TP else float("inf")
            print(f"   {t:>10.2f}  {TP:>4} {FP:>4} {FN:>4}  {pr:>6.3f} {re:>6.3f} {f1:>6.3f}  {loc:.0f}")

    json.dump(m, open(RESULTS / "metrics_2b.json", "w"), indent=2)
    print(f"\nSaved to {RESULTS / 'metrics_2b.json'}")
