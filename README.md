# CryoET Tasks — AI + CryoET Project

This repo contains submissions for the CZ cryoET project tasks.

---

## Task 1 — Transformer Training Review

**Folder:** `Task1/`

A code review and analysis of a protein fitness Transformer training script.

| File | Description |
|---|---|
| `code_review.md` | Full code review: what works, main issues, and how to restructure |
| `training_curves.png` | Training vs validation loss curves |
| `predictions.png` | Predicted vs actual scatter plot |

---

## Task 2 — CryoET Ribosome Picking with TopCUP

**Folder:** `Task2/`

An end-to-end pipeline that runs the pretrained **TopCUP** particle picker on two CZ
cryoET Data Portal tomograms and reports a quality assessment.

| Sub-task | What it does |
|---|---|
| **2a** — Download + visualize | Fetches tomograms, ribosome annotations, and model weights via the CZ data portal |
| **2b** — Inference on synthetic data | Runs TopCUP on an in-distribution synthetic tomogram; F1 = 0.76, localization 37 Å |
| **2c** — In-situ failure analysis | Runs the same model on real Chlamydomonas cellular data; diagnoses why it produces 0 detections (data drift) |
| **2d** — Improvement proposal | Technical plan to close the train→test distribution gap and restore in-situ performance |

### Key result

| Dataset | F1 | Localization |
|---|---|---|
| Synthetic (in-distribution) | **0.76** | 37 Å |
| Chlamydomonas in-situ (held-out) | **0.00** | — |

The drop from F1 0.76 to 0.00 on real cellular data — despite identical model, weights,
and config — is the core finding. It's driven by data drift: the model trained on clean
synthetic phantoms but the cellular tomogram is crowded, lower-contrast, and has a
different acquisition signature. Task 2c quantifies this; Task 2d proposes the fix.

### Quick start

```bash
cd Task2
pip install topcup==1.2.2 --no-deps
pip install -r requirements.txt
python task2a/download.py         # ~3 GB download
python task2b/inference.py        # auto-detects CPU/GPU
python task2c/inference_chlamy.py
```

See `Task2/README.md` for full setup (including the numcodecs install note for Apple
Silicon) and detailed run instructions.
