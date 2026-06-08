"""
Task 2b — TopCUP inference on the synthetic tomogram + quality metrics.

Self-contained: reads data from submission/data (populated by task2a/download.py),
writes results next to this script. Device-agnostic — runs on CPU or NVIDIA GPU.

Hyperparameters (see TASK2B_QUALITY_ASSESSMENT.md for justification):
  RADIUS          = 15 vox (150 Å)  — one ribosome radius; CZII matching criterion
  score_threshold = 0.19 (ribosome) — the model's own trained per-class threshold
  tomo_type       = denoised        — matches the reconstruction the model trained on
  class config    = full 6-class    — so "ribosome" reads output channel 4
"""

import json
import subprocess
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
    Device-agnostic: the SAME script runs on CPU or NVIDIA GPU.
      - GPU present: TopCUP runs natively (--gpus 1); only torch.load weights_only fix.
      - No GPU (CPU): also patches Lightning→CPU, no-ops .cuda()/.to(cuda), disables CUDA autocast.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = out_dir / "_wrapper.py"
    wrapper.write_text(f"""\
import torch, functools, contextlib

USE_GPU = torch.cuda.is_available()

# 1. Fix torch.load for old checkpoints (needed on BOTH devices, PyTorch >=2.6)
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{{**kw, 'weights_only': False}})

# 2-4. CPU-only patches — applied only when no NVIDIA GPU is present.
if not USE_GPU:
    def _patch_trainer():
        for mod_name in ('pytorch_lightning', 'lightning.pytorch', 'lightning'):
            try:
                import importlib
                mod = importlib.import_module(mod_name)
                T = getattr(mod, 'Trainer', None)
                if T is None: continue
                orig = T.__init__
                @functools.wraps(orig)
                def _cpu(self, *a, **kw):
                    kw.update(accelerator='cpu')
                    kw.pop('devices', None); kw.pop('gpus', None)
                    if kw.get('precision') in ('16','16-mixed','bf16','bf16-mixed'):
                        kw['precision'] = '32-true'
                    orig(self, *a, **kw)
                T.__init__ = _cpu
                break
            except: continue
    _patch_trainer()

    def _is_cuda(x): return (isinstance(x, str) and 'cuda' in x) or (isinstance(x, torch.device) and x.type == 'cuda')
    torch.cuda.current_device = lambda: 0
    torch.nn.Module.cuda = lambda self, *a, **kw: self
    torch.Tensor.cuda = lambda self, *a, **kw: self
    _ot = torch.Tensor.to
    torch.Tensor.to = lambda self, *a, **kw: self if a and _is_cuda(a[0]) else _ot(self, *a, **kw)
    _om = torch.nn.Module.to
    torch.nn.Module.to = lambda self, *a, **kw: self if a and _is_cuda(a[0]) else _om(self, *a, **kw)

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        _orig_ac = torch.amp.autocast
        class _CPUAutocast(_orig_ac):
            def __init__(self, device_type='cpu', **kw):
                if device_type == 'cuda':
                    device_type = 'cpu'; kw['dtype'] = torch.float32
                super().__init__(device_type, **kw)
        torch.amp.autocast = _CPUAutocast
    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp'):
        torch.cuda.amp.autocast = lambda enabled=True, **kw: contextlib.nullcontext()

import sys
sys.argv = [
    'topcup', 'inference',
    '--copick_config',      '{cfg}',
    '--run_names',          '{RUN_NAME}',
    '--pretrained_weights', '{CKPT.resolve()}',
    '--pixelsize',          '{VS}',
    '--tomo_type',          'denoised',
    '--user_id',            'topcup',
    '--output_dir',         '{out_dir.resolve()}',
    '--gpus',               '1' if USE_GPU else '0',
    '--has_ground_truth',   'False',
]
from topcup.cli.cli import cli
if __name__ == '__main__':      # required: DataLoader workers re-import this module
    cli(standalone_mode=True)
""")
    import torch as _t
    dev = "GPU" if _t.cuda.is_available() else "CPU"
    print(f"  Running TopCUP on {dev} ({'fast' if dev=='GPU' else '20-60 min on CPU'})...")
    subprocess.run([sys.executable, str(wrapper)], check=False)


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
