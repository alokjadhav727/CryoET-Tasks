# Task 2c - Held-out Inference & Failure Characterization

> **Note:** I ran this on the exact chunk the task specifies - run 14070, voxel offset
> `(x=160, y=752)`, size `256` in x and y, full z - using the *same* model, weights, and
> config that scored F1 0.76 on the synthetic dataset. On CPU, ~15 min. The result:
> **0 confident ribosome detections** against **56 ground-truth ribosomes** in the chunk.
> Precision/Recall/F1 all 0.00. That zero is the finding, and the rest of this doc is
> about *why*.

The short version: **there is a clear data drift between the dataset the model was trained
on (the synthetic data) and the dataset we're testing it on (the in-situ Chlamydomonas
data).** The model was only ever trained on the synthetic distribution, so when the test
data looks different, its performance falls off a cliff.

I kept everything identical to Task 2b on purpose - same checkpoint, same thresholds, same
matching radius (one ribosome radius, here 19 vox = 150 Å), same metrics - so the two
numbers are directly comparable and any difference comes from the data, not the setup.

---

## 1. The headline comparison (same metrics as Task 2b)

| | Synthetic (training distribution) | Chlamydomonas chunk (held-out) |
|---|---|---|
| Ribosome predictions (≥ threshold) | 47 | **0** |
| Ground-truth ribosomes | 31 (whole tomogram) | 56 (in chunk) |
| Precision | 0.64 | **0.00** |
| Recall | 0.94 | **0.00** |
| F1 | **0.76** | **0.00** |
| Localization error | 37 Å | - (nothing to match) |

Same model, same settings - and it goes from finding 30 of 32 ribosomes to finding none of
56. So the question isn't "is it a bit worse," it's "why does it switch off entirely."

---

## 2. Where and why it fails - looking at the model's own output

An empty prediction list could mean two different things: the pipeline is broken, or the
model genuinely sees nothing it recognizes. To tell them apart I looked past the final
picks at the model's raw output.

**It's not a bug.** The chunk we feed in is exactly the right slice of the tomogram, the
data loads correctly, and the *same* config gives F1 0.76 on the synthetic data. So the
empty list is the model's honest answer, not a plumbing problem.

**The model's confidence collapses.** Instead of the final picks, I looked at the model's
raw per-voxel ribosome score (its heatmap) across the chunk. On the synthetic data the
model is very confident - its scores reach **0.998** on real ribosomes. On the held-out
chunk the scores are basically flat: the single highest voxel only reaches **~0.52**, and
almost everything else is **below 0.05**. If I drop the threshold all the way to zero, the
strongest detections still only score around **0.008 - about 125× lower than on synthetic
data.** The model isn't hovering just under the cutoff; it's confidently saying "nothing
here" everywhere.

**And the weak responses don't point at ribosomes anyway.** Even with the threshold at
zero, the handful of near-zero blips don't line up with the real ribosomes - most
ribosomes have nothing near them at all. So there's no hidden correct answer waiting to be
unlocked by lowering the threshold; the model's output simply isn't tracking ribosomes in
this dataset.

**In one line:** the model fails because of **data drift** - its scores collapse to near
zero on the test dataset, so no detection ever clears the bar.

---

## 3. What's different between the two datasets (the drift)

The model was trained on the synthetic dataset and tested on the in-situ one, and these two
datasets simply don't look the same. Here are the differences I could measure directly
(each is backed by a figure from our `analyze_drift.py` comparison):

- **The test data is far more crowded.** The synthetic dataset has well-separated
  particles. The in-situ chunk has the 56 ribosomes *plus* 706 other annotated particles
  (nucleosomes) and lots of other dense structure packed around them. The model never saw
  this kind of crowding during training.

- **The ribosomes stand out much less.** I measured how strongly each ribosome stands out
  from its surroundings. In the synthetic data a ribosome is about **2.7×** brighter than
  the local background; in the in-situ data it's only **1.2×**. The signal the model keys
  on is roughly half as strong.

- **The overall "texture" of the data is different.** The synthetic images are mostly
  empty with a few sharp, dense spots - a very distinctive pattern (skew −3.0, kurtosis
  ~30). The in-situ images are much more uniform and busy throughout (skew −0.1, kurtosis
  ~0.6). (After the model's normalization both inputs sit in the same numeric range, so
  this is a difference in *pattern*, not just brightness - I checked that scale isn't the
  issue.)

- **The frequency content is different.** A 2D FFT shows the in-situ data is about **10×
  more directionally lopsided** than the synthetic data (a power ratio of ~92 vs ~9), and
  the two were produced with different processing. So even the same particle looks
  different between the two datasets.

Any one of these differences nudges the model off the distribution it learned; together
they collapse its ribosome scores to near zero, which is exactly what the heatmap shows.

---

## 4. How I'd fix it (brief)

The diagnosis points straight at the fixes (the lead-in to Task 2d):
1. **Fine-tune the model on some of the in-situ data**, so it learns what ribosomes look
   like in the test distribution rather than only the training one.
2. **Make the training data look more like the test data** - add the kind of crowding and
   variability the in-situ data has.
3. **Recalibrate the threshold** for the in-situ data; the value tuned on synthetic data is
   far too high once the scores are uniformly suppressed.
4. **Use unlabeled in-situ data** (self-training / pseudo-labels) to close the gap without a
   large new labeling effort.

---

## 5. Reproduce

```bash
python inference_chlamy.py    # extracts the chunk, runs inference, writes results/metrics_2c.json
python analyze_drift.py       # generates drift figures in figures/ + drift_stats.json
```
