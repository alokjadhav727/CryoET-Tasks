# TopCUP Cryo-ET Particle Picking - Task 2 Pipeline

An end-to-end, reproducible pipeline that runs the pretrained **TopCUP** ribosome
picker on two CZ cryoET Data Portal tomograms - a **synthetic** (in-distribution)
volume and a **Chlamydomonas** in-situ volume - and reports a quality assessment.

Everything is self-contained: the scripts download their own data into `data/` and
resolve all paths relative to this folder, so the pipeline runs from any machine with no
edits. It works on **CPU or NVIDIA GPU**.

## Layout

```
Task2/
├── README.md
├── requirements.txt
├── requirements_win_py311.txt    # relaxed pins for Windows + Python 3.11
├── data/                         # created by task2a/download.py (git-ignored)
│   ├── synthetic/{tomograms,overlay}
│   ├── chlamydomonas/{tomograms,overlay}
│   └── weights/topcup_weights/topcup_phantom_24_tomograms.ckpt
│
├── task2a/   Data download + visualization
│   ├── download.py               # tomograms + ribosome picks (copick/CZCDP) + TopCUP weights
│   └── visualize.py              # per-slice raw | annotated overlays
│
├── task2b/   Inference + quality assessment on SYNTHETIC (in-distribution)
│   ├── inference.py              # device-agnostic TopCUP inference + metrics (Mac/Linux)
│   ├── inference_win.py          # Windows drop-in (uses junctions instead of symlinks)
│   └── TASK2B_QUALITY_ASSESSMENT.md   # hyperparameter choices + detection/localization analysis
│
├── task2c/   Inference on the CHLAMYDOMONAS in-situ chunk (held-out)
│   ├── inference_chlamy.py       # extract chunk + inference + metrics (Mac/Linux)
│   ├── inference_chlamy_win.py   # Windows drop-in (uses junctions instead of symlinks)
│   ├── analyze_drift.py          # train-vs-test data-drift comparison -> figures/ + drift_stats.json
│   └── TASK2C_FAILURE_ANALYSIS.md  # where/why the model fails + dataset differences (data drift)
│
└── task2d/   Proposal: improving in-situ performance
    └── TASK2D_PROPOSAL.md        # technical plan to close the data-drift gap + how to evaluate it
```

## Setup

> **First:** `cd Task2` - all commands below assume you're running from inside this folder.

`topcup 1.2.2` pins `numcodecs==0.11.0` internally, which conflicts with `numcodecs 0.12.x`
that the rest of the stack needs. The install is a two-step process to override that pin.

**Apple Silicon (M-series Mac) - Python 3.12**
```bash
conda create -n topcup python=3.12
conda activate topcup
pip install topcup==1.2.2 --no-deps   # install topcup without its strict numcodecs pin
pip install -r requirements.txt        # installs numcodecs 0.12.1 + everything else
```

**Linux - Python 3.11**
```bash
conda create -n topcup python=3.11
conda activate topcup
pip install topcup==1.2.2 --no-deps   # install topcup WITHOUT its strict numcodecs pin
pip install -r requirements.txt        # installs the rest (topcup is not in requirements.txt)
```

**Windows - Python 3.11**

The pinned `requirements.txt` was generated on Apple Silicon / Python 3.12 and
several pins (`tifffile==2026.6.1`, `numpy==2.4.6`, `scikit-learn==1.8.0`,
`torch==2.12.0`, ...) have no cp311/Windows wheels. On Windows, install topcup
**with** its dependencies and then use the relaxed companion file
`requirements_win_py311.txt`:

```powershell
conda create -n topcup python=3.11
conda activate topcup
pip install topcup==1.2.2                       # install topcup + compatible deps
pip install -r requirements_win_py311.txt       # relaxed pins that have cp311 wheels
```

The relaxed file overrides `numcodecs` to `>=0.12,<0.13` (matching the
Apple-Silicon override) and only lower-bounds the rest, so pip can pick the
versions that actually publish Windows + Python 3.11 wheels.

**Windows inference scripts** - the original `task2b/inference.py` and
`task2c/inference_chlamy.py` use `Path.symlink_to()` to wire the copick
directory layout, which on Windows raises `OSError: [WinError 1314] A required
privilege is not held by the client` for non-admin users. Drop-in replacements
that use a Windows directory **junction** (`mklink /J`, no admin required) are
provided next to them:

- `task2b/inference_win.py`
- `task2c/inference_chlamy_win.py`

On Windows, run those instead of `inference.py` / `inference_chlamy.py` in the
"Run the whole pipeline" section below. Everything else (config, inference,
metrics) is byte-identical.



`requirements.txt` pins the exact versions used to produce these results (TopCUP 1.2.2,
torch 2.12, copick 1.24, etc.). On an NVIDIA machine, install the CUDA-matched `torch`
wheel after; the inference scripts auto-detect the GPU.

## Run the whole pipeline

**Mac / Linux**
```bash
# 1. Download tomograms, ribosome annotations, and TopCUP weights into ./data
python task2a/download.py
python task2a/visualize.py                 # optional: per-slice overlays

# 2. Inference + quality assessment on the synthetic (in-distribution) tomogram
python task2b/inference.py

# 3. Inference on the held-out Chlamydomonas in-situ chunk
python task2c/inference_chlamy.py
```

**Windows**
```powershell
# 1. Download tomograms, ribosome annotations, and TopCUP weights into ./data
python task2a\download.py
python task2a\visualize.py                 # optional: per-slice overlays

# 2. Inference + quality assessment on the synthetic (in-distribution) tomogram
#    inference_win.py uses Windows directory junctions instead of symlinks (no admin required)
python task2b\inference_win.py

# 3. Inference on the held-out Chlamydomonas in-situ chunk
python task2c\inference_chlamy_win.py
```

Each script writes its outputs (predictions, `metrics_*.json`, visualizations) into its
own task folder. Re-running is safe - downloads and the extracted chunk are cached.

## Headline results

| Dataset | Regime | Ribosome F1 | Localization | Notes |
|---|---|---|---|---|
| CZII synthetic (run 16463) | in-distribution | **0.76** (R=0.94) | **37 Å** | 30/32 GT found |
| Chlamydomonas chunk (run 14070) | in-situ, held-out | **0.00** | - | 0 confident detections vs 56 GT |

The synthetic result validates the pipeline and characterizes TopCUP's in-distribution
ceiling; the Chlamydomonas chunk demonstrates the in-distribution -> in-situ
generalization gap. See `task2b/TASK2B_QUALITY_ASSESSMENT.md` for the full Task 2b
write-up (hyperparameters + metrics).

## Two implementation notes that matter

1. **Device-agnostic by design.** TopCUP is hard-coded for NVIDIA GPU. Each
   `inference.py` checks `torch.cuda.is_available()`: on GPU it runs natively
   (`--gpus 1`, only a `torch.load(weights_only=False)` PyTorch-2.6 compat fix); on CPU
   it additionally patches Lightning->CPU, no-ops `.cuda()`/`.to('cuda')`, and disables
   CUDA autocast. One script, both devices. (MPS/Apple-Silicon is intentionally not
   implemented - the task states it is not evaluated.)

2. **The model is 6-class - the config must say so.** `topcup_phantom_24_tomograms.ckpt`
   outputs 6 particle channels in training-label order (apo-ferritin=1, beta-amylase=2,
   beta-galactosidase=3, **ribosome=4**, thyroglobulin=5, virus-like-particle=6). The
   copick config replicates all six so "ribosome" reads **channel 4**. A single-object
   config (`ribosome: label 1`) would silently read channel 1 (apo-ferritin) and
   mislabel it - producing all false positives. This was the key correctness fix.
