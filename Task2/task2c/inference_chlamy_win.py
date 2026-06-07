"""
Task 2c — TopCUP inference on a Chlamydomonas in-situ chunk + metrics.

WINDOWS variant of inference_chlamy.py. The ONLY difference vs. the original
is the directory-link step:

  Original (Linux/macOS):
      link.symlink_to(chunk_zarr.resolve())   # needs admin/Dev-Mode on Windows

  This file (Windows):
      _make_dir_link(chunk_zarr.resolve(), link)
                                              # uses mklink /J (directory junction)
                                              # which works without admin rights

Everything else (chunk extraction, copick config, the TopCUP CPU/GPU wrapper,
metrics) is byte-identical to inference_chlamy.py.
"""

import json
import os
import shutil
import subprocess
import sys
import numpy as np
import zarr
from pathlib import Path
from scipy.spatial.distance import cdist

ROOT      = Path(__file__).resolve().parents[1]      # submission/
DATA_DIR  = ROOT / "data"
CKPT      = DATA_DIR / "weights/topcup_weights/topcup_phantom_24_tomograms.ckpt"
RESULTS   = Path(__file__).resolve().parent / "results"
CHUNK_DIR = Path(__file__).resolve().parent / "chunk_data"

VS         = 7.84     # Å/voxel for chlamydomonas
RADIUS     = 19.1     # matching radius = 150 Å / 7.84 ≈ 19 voxels (1 ribosome radius)
RUN_NAME   = "chlamy_chunk"

X_OFF, Y_OFF = 160, 752
CHUNK_SIZE   = 256    # in x and y


# ---------------------------------------------------------------------------
# THE ONLY WINDOWS-SPECIFIC PIECE
# ---------------------------------------------------------------------------
def _make_dir_link(src, dst):
    """
    Make `dst` point at the directory `src` -- the Windows-safe way.

      1. On Windows use `mklink /J` (a directory junction). Junctions do NOT
         require Administrator / Developer Mode, unlike regular symlinks.
      2. On Linux/macOS fall back to a normal symlink.
      3. As a last resort, copy the tree.
    """
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        try:
            subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                           check=True, capture_output=True)
            return
        except subprocess.CalledProcessError:
            pass
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# Everything below this line is the original inference_chlamy.py logic,
# unchanged except `link.symlink_to(...)` -> `_make_dir_link(...)`.
# ---------------------------------------------------------------------------
def _chlamy_tomo_zarr():
    """Locate the chlamydomonas tomogram zarr (downloaded by task2a) by globbing."""
    zs = list((DATA_DIR / "chlamydomonas/tomograms").rglob("*.zarr"))
    if not zs:
        raise FileNotFoundError(
            f"No chlamydomonas tomogram .zarr under {DATA_DIR/'chlamydomonas/tomograms'} -- run task2a/download.py first.")
    return zs[0]


def extract_chunk():
    """Extract the (512, 256, 256) sub-volume and save it as an OME-Zarr TopCUP can read."""
    src = zarr.open(str(_chlamy_tomo_zarr()), mode="r")
    src = src["0"] if "0" in src else src
    print(f"  Full volume shape: {src.shape}")
    chunk = src[:, Y_OFF:Y_OFF+CHUNK_SIZE, X_OFF:X_OFF+CHUNK_SIZE]
    print(f"  Chunk shape: {chunk.shape}")

    chunk_zarr = CHUNK_DIR / "denoised.zarr"
    chunk_zarr.parent.mkdir(parents=True, exist_ok=True)
    if not chunk_zarr.exists():
        store = zarr.open_group(str(chunk_zarr), mode="w")
        store.create_dataset("0", data=chunk, chunks=(64, 64, 64), dtype="float32")
        zattrs = {
            "multiscales": [{
                "axes": [
                    {"name": "z", "type": "space", "unit": "angstrom"},
                    {"name": "y", "type": "space", "unit": "angstrom"},
                    {"name": "x", "type": "space", "unit": "angstrom"}
                ],
                "datasets": [{"coordinateTransformations": [
                    {"scale": [VS, VS, VS], "type": "scale"}
                ], "path": "0"}],
                "name": "/", "version": "0.4"
            }]
        }
        (chunk_zarr / ".zattrs").write_text(json.dumps(zattrs, indent=2))
        print(f"  Saved chunk zarr: {chunk_zarr}")
    else:
        print(f"  Chunk zarr already exists: {chunk_zarr}")
    return chunk_zarr


def make_copick_config(chunk_zarr):
    static = CHUNK_DIR / "static"
    # copick formats voxel spacing as :.3f, so the dir must be VoxelSpacing7.840 (not 7.84)
    link_dir = static / f"ExperimentRuns/{RUN_NAME}/VoxelSpacing{VS:.3f}"
    link_dir.mkdir(parents=True, exist_ok=True)

    link = link_dir / "denoised.zarr"
    if not link.exists() and not link.is_symlink():
        _make_dir_link(chunk_zarr.resolve(), link)         # <-- only change
        print(f"  link: denoised.zarr -> {chunk_zarr.resolve()}")

    config = {
        "config_type": "filesystem",
        "name": "chlamy_chunk",
        "version": "0.5.0",
        # Replicate the phantom model's 6-class training order so "ribosome" maps to
        # output channel 4 (label=1 would read channel 1 = apo-ferritin instead).
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
        "overlay_root": str((CHUNK_DIR / "overlay").resolve()),
        "overlay_fs_args": {"auto_mkdir": True},
        "static_root": str(static.resolve()),
    }
    cfg = CHUNK_DIR / "copick_chlamy_chunk.json"
    json.dump(config, open(cfg, "w"), indent=2)
    print(f"  Config: {cfg}")
    return cfg


def run_inference(cfg, out_dir):
    """Device-agnostic: GPU native if available, else CPU patches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = out_dir / "_wrapper.py"
    wrapper_src = f"""\
import torch, functools, contextlib

USE_GPU = torch.cuda.is_available()

# 1. Fix torch.load (needed on BOTH devices, PyTorch >=2.6)
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{{**kw, 'weights_only': False}})

# 2-4. CPU-only patches -- applied only when no NVIDIA GPU is present.
if not USE_GPU:
    def _patch_trainer():
        for m in ('pytorch_lightning', 'lightning.pytorch', 'lightning'):
            try:
                import importlib; mod = importlib.import_module(m)
                T = getattr(mod, 'Trainer', None)
                if T is None: continue
                orig = T.__init__
                @functools.wraps(orig)
                def _cpu(self, *a, **kw):
                    kw.update(accelerator='cpu'); kw.pop('devices', None); kw.pop('gpus', None)
                    if kw.get('precision') in ('16','16-mixed','bf16','bf16-mixed'): kw['precision'] = '32-true'
                    orig(self, *a, **kw)
                T.__init__ = _cpu; break
            except: continue
    _patch_trainer()

    def _is_cuda(x): return (isinstance(x, str) and 'cuda' in x) or (isinstance(x, torch.device) and x.type == 'cuda')
    torch.cuda.current_device = lambda: 0
    torch.nn.Module.cuda = lambda self, *a, **kw: self
    torch.Tensor.cuda  = lambda self, *a, **kw: self
    _ot = torch.Tensor.to
    torch.Tensor.to = lambda self, *a, **kw: self if a and _is_cuda(a[0]) else _ot(self, *a, **kw)
    _om = torch.nn.Module.to
    torch.nn.Module.to = lambda self, *a, **kw: self if a and _is_cuda(a[0]) else _om(self, *a, **kw)

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        _oa = torch.amp.autocast
        class _AC(_oa):
            def __init__(self, device_type='cpu', **kw):
                if device_type == 'cuda': device_type = 'cpu'; kw['dtype'] = torch.float32
                super().__init__(device_type, **kw)
        torch.amp.autocast = _AC
    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'amp'):
        torch.cuda.amp.autocast = lambda enabled=True, **kw: contextlib.nullcontext()

import sys
sys.argv = [
    'topcup', 'inference',
    '--copick_config',      r'{cfg}',
    '--run_names',          '{RUN_NAME}',
    '--pretrained_weights', r'{CKPT.resolve()}',
    '--pixelsize',          '{VS}',
    '--tomo_type',          'denoised',
    '--user_id',            'topcup',
    '--output_dir',         r'{out_dir.resolve()}',
    '--gpus',               '1' if USE_GPU else '0',
    '--has_ground_truth',   'False',
]
from topcup.cli.cli import cli
if __name__ == '__main__':
    cli(standalone_mode=True)
"""
    # Force UTF-8 so Windows' default cp1252 doesn't mangle non-ASCII characters.
    wrapper.write_text(wrapper_src, encoding="utf-8")

    import torch as _t
    dev = "GPU" if _t.cuda.is_available() else "CPU"
    print(f"  Running TopCUP on chunk ({dev}{'' if dev=='GPU' else ', 20-60 min'})...")
    subprocess.run([sys.executable, str(wrapper)], check=False)


def load_predictions(out_dir):
    import pandas as pd
    csvs = list(out_dir.glob("*.csv"))
    if not csvs:
        print("  No CSV -- inference failed")
        return np.zeros((0, 3))
    df = pd.concat([pd.read_csv(f) for f in csvs])
    df = df[df["particle_type"] == "ribosome"]
    pts = np.column_stack([df["z"]/VS, df["y"]/VS, df["x"]/VS]) if len(df) else np.zeros((0, 3))
    print(f"  {len(pts)} predictions")
    return pts


def load_chunk_gt():
    """Filter the 202 full-volume GT annotations to the chunk; shift to chunk-relative voxels."""
    with open(DATA_DIR / "chlamydomonas/overlay/ribosome_annotations_chlamydomonas.json") as f:
        d = json.load(f)
    pts_full = np.array([[p["z"]/VS, p["y"]/VS, p["x"]/VS] for p in d["points"]])
    in_chunk = (
        (pts_full[:, 2] >= X_OFF) & (pts_full[:, 2] < X_OFF + CHUNK_SIZE) &
        (pts_full[:, 1] >= Y_OFF) & (pts_full[:, 1] < Y_OFF + CHUNK_SIZE)
    )
    chunk_pts = pts_full[in_chunk].copy()
    chunk_pts[:, 2] -= X_OFF
    chunk_pts[:, 1] -= Y_OFF
    print(f"  GT in chunk: {in_chunk.sum()} of {len(pts_full)}")
    return chunk_pts


def compute_metrics(pred, gt):
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

    print("1. Extracting chunk (x=160:416, y=752:1008, full z)...")
    chunk_zarr = extract_chunk()

    print("\n2. Creating copick config...")
    cfg = make_copick_config(chunk_zarr)

    print("\n3. Running TopCUP inference on chunk...")
    out_dir = RESULTS / "chlamy_chunk"
    run_inference(cfg, out_dir)

    print("\n4. Loading predictions and GT...")
    pred = load_predictions(out_dir)
    gt   = load_chunk_gt()

    print(f"\n5. Metrics  (radius = {RADIUS:.0f} vox = 150 A = 1 ribosome radius)")
    m = compute_metrics(pred, gt)
    print(f"   Precision : {m['precision']:.3f}")
    print(f"   Recall    : {m['recall']:.3f}")
    print(f"   F1        : {m['f1']:.3f}")
    print(f"   Loc error : {m['loc_err_vox']:.1f} vox  ({m['loc_err_ang']:.0f} A)")
    print(f"   TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    json.dump(m, open(RESULTS / "metrics_2c.json", "w"), indent=2)
    print(f"\nSaved to {RESULTS / 'metrics_2c.json'}")
