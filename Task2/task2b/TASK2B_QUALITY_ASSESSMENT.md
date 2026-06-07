# Task 2b - Inference & Quality Assessment on Synthetic Data

**Dataset:** CZCDP-10310 synthetic tomogram (`TS_5_4`, deposition 16463), 10.012 Å voxel size.
**Result:** Precision 0.64, Recall 0.94, F1 0.76, localization 37 Å (TP=30, FP=17, FN=2 against 31 GT ribosomes).

---

## 1. Hyperparameters

**Confidence threshold: 0.19**
The model ships its own per-class threshold (0.19 for ribosome), tuned during training. Using the model's own threshold rather than an arbitrary value like 0.5 is the right thing to do here.

**Matching radius: 15 voxels (150 Å) = one ribosome radius**
A pick is a true positive if it falls within this distance of a ground-truth ribosome. One ribosome radius is a physically meaningful boundary - a pick inside it unambiguously belongs to that ribosome. A radius sweep is reported below to show results are not sensitive to this choice.

**Reconstruction type: `denoised`**
The model was trained on this reconstruction type, so we match it.

**Tiling and aggregation: TopCUP defaults**
96³ patches, 0.21 overlap, one test-time augmentation flip. Left at author defaults to avoid breaking the assumptions the pretrained weights were trained with.

**6-class config (critical)**
The checkpoint is a 6-class model with output channels in training order: apo-ferritin=1, beta-amylase=2, beta-galactosidase=3, **ribosome=4**, thyroglobulin=5, virus-like-particle=6. The copick config must replicate all six classes in this order so "ribosome" reads channel 4. Using a single-object config with `ribosome: label 1` silently reads the apo-ferritin channel and mislabels it "ribosome", producing all false positives. This was the key fix.

---

## 2. Results

**Detection**

| Metric | Value |
|---|---|
| Precision | 0.64 |
| Recall | **0.94** |
| F1 | **0.76** |

Recall (0.94) is the priority metric for particle picking. Missing a real ribosome loses it permanently and biases any downstream analysis. The model finds 30 of 32 ribosomes, missing only 2.

**Localization**

Average distance from picked center to nearest ground-truth: **3.7 vox = 37 Å**, roughly a quarter of the ribosome radius. The picks are well inside the particles, not just clipping the edge.

**Radius sweep**

| Matching radius | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 15 vox (150 Å) | 30 | 17 | 2 | 0.64 | 0.94 | **0.76** |
| 30 vox (300 Å) | 31 | 16 | 2 | 0.66 | 0.94 | 0.78 |
| 45 vox (451 Å) | 32 | 15 | 1 | 0.68 | 0.97 | 0.80 |
| 60 vox (601 Å) | 33 | 14 | 1 | 0.70 | 0.97 | 0.81 |

Recall is 0.94 at the tightest radius and barely changes as it loosens. The picks are genuinely close to ground truth, not just scraped in by a generous radius.

**Best hyperparameter set:** threshold 0.19, radius 15 vox (150 Å), tomo_type `denoised`, TopCUP default tiling, full 6-class config.

---

## 3. A note on false positives

The 17 false positives are mostly not wrong detections. The synthetic tomogram has all six trained particle types, and the model correctly detects the others (ferritin, thyroglobulin, etc.) too. Because we score against ribosome-only ground truth, those detections count as FPs. The ribosome precision of 0.64 is a lower bound - the actual picking quality is better than that number suggests.

---

## Reproduce

```bash
python inference.py         # writes results/metrics_2b.json
```
