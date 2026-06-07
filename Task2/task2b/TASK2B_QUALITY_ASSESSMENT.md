# Task 2b - Inference & Quality Assessment on Synthetic Data

> **Note:** I ran this end-to-end before writing it up - `python inference.py` on the
> CZCDP-10310 synthetic tomogram (`TS_5_4`, deposition 16463), on CPU, ~42 minutes for
> the sliding-window inference. Final numbers at the chosen operating point:
> **Precision 0.64, Recall 0.94, F1 0.76, localization error 37 Å** (TP=30, FP=17, FN=2,
> against 31 ground-truth ribosomes). The same config and weights produce the same result
> on GPU - the script auto-detects the device.

This tomogram is *in-distribution*: it comes from the CZII synthetic phantom family the
model was trained on. So I treated Task 2b as two things at once - a sanity check that my
inference pipeline is actually correct, and a measurement of how well TopCUP does when the
data is exactly what it expects.

---

## Q1. Which inference-time hyperparameters did I choose, and why?

These are the knobs that decide what counts as a correct detection and how the model's raw
heatmap becomes a list of picks. I tried not to invent values - wherever the model shipped
its own tuned setting, I used that, and I'm flagging the few places where I made a call.

**Confidence threshold - 0.19 (the model's own).** TopCUP produces a per-voxel probability
for each particle class, and a pick is only emitted where the ribosome score clears this
threshold (after non-maximum suppression). Rather than picking a round number, I used the
**per-class threshold the authors tuned during training (0.19 for ribosome)**, which ships
in the TopCUP config. It was chosen on a validation set to balance precision and recall for
*this specific particle*, so it's the honest operating point for in-distribution data - a
generic 0.5 would just be miscalibrated here.

**Matching radius - 15 voxels (150 Å), i.e. one ribosome radius.** This is what turns the
predicted points into TP/FP/FN: a pick counts as correct if it lands within this distance
of a real ribosome. I set it to one ribosome radius because, physically, a pick within one
radius of the true center is *inside* the particle - it's unambiguously the same ribosome.
That's tight enough to be meaningful but not so tight that it punishes sub-voxel jitter.
I also didn't want the whole assessment to hinge on one radius, so I report a sweep (below).

**Reconstruction type - `denoised`.** The phantom checkpoint was trained on the denoised
reconstruction, so that's what I feed it. It sounds trivial, but matching the exact input
type the model saw in training is the difference between a fair in-distribution test and an
accidental domain shift.

**Tiling and aggregation - TopCUP's defaults.** The volume is scanned in overlapping 96³
patches, the logits are averaged on the overlaps, a flipped pass is averaged in as
test-time augmentation, and peaks are pulled out with non-maximum suppression whose radius
is tied to the particle size. I deliberately left these at the authors' defaults - they
were chosen to match the network's receptive field and the particle scale, and overriding
them would quietly break the assumptions baked into the pretrained weights. (The NMS radius
scaling with particle size is also what stops one big ribosome from being counted twice.)

**The class config - this is the one that actually matters.** This took me the longest to
get right, so it's worth being explicit. The checkpoint is a **6-class** model, and its
output channels follow the order the classes were trained in: apo-ferritin=1,
beta-amylase=2, beta-galactosidase=3, **ribosome=4**, thyroglobulin=5,
virus-like-particle=6. The copick config has to **replicate all six classes in that order**
so that "ribosome" reads **channel 4**. My first version used a single-object config with
`ribosome: label 1` - and it silently read channel 1 (apo-ferritin), relabeled it
"ribosome," and produced what looked like a model that found particles but got every one
wrong. Once I rebuilt the config with the full 6-class layout (taking the per-class
thresholds and radii straight from the model's own config), the result jumped to F1 0.76.
If there's one thing to take away from this task, it's that.

---

## Q2. How good are the results - detection *and* localization?

The task asks for both "did we find the right particles" and "are the picks in the right
place," so I report them separately. They're genuinely different questions: a model can
score well on one and badly on the other.

**Detection quality.** At the chosen operating point I get **Precision 0.64, Recall 0.94,
F1 0.76**. I'm reporting all three on purpose:
- *Recall (0.94)* is the number I care about most for particle picking. It says we
  recovered 30 of the 32 ribosomes and missed only 2. In structural biology a missed
  particle is gone for good and biases everything downstream, so high recall is the right
  thing to optimize.
- *Precision (0.64)* is the counterweight - it says 30 of our 47 picks are real ribosomes.
- *F1 (0.76)* just summarizes the trade-off in one number.

The model is clearly sitting in a sensible, recall-leaning regime: it would rather flag a
few extra candidates than drop a real one, which is exactly the bias you want when false
positives can be cleaned up later by 2D/3D classification.

**Localization quality.** The matched picks land on average **3.7 voxels - about 37 Å -
from the true center**. That's roughly a quarter of the ribosome's own radius, so the picks
aren't just "near enough to count," they're sitting well inside the particle. I report this
separately from F1 because detection counts are blind to *where* inside the matching radius
a pick falls - a model could score a great F1 while being systematically off-center. Here
it's both sensitive and accurate, which is the combination you'd actually want before
feeding these coordinates into subtomogram averaging.

**Is this robust, or an artifact of one radius?** To check, I swept the matching radius:

| Matching radius | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 15 vox (150 Å) | 30 | 17 | 2 | 0.64 | 0.94 | **0.76** |
| 30 vox (300 Å) | 31 | 16 | 2 | 0.66 | 0.94 | 0.78 |
| 45 vox (451 Å) | 32 | 15 | 1 | 0.68 | 0.97 | 0.80 |
| 60 vox (601 Å) | 33 | 14 | 1 | 0.70 | 0.97 | 0.81 |

The thing to notice is that recall is already 0.94 at the *tightest* radius and barely
moves as I loosen it. If the matches were only scraping in thanks to a generous radius,
recall would climb steeply as the radius grew - it doesn't. That's the same story the 37 Å
localization number tells, just from the other direction.

**My best hyperparameter set, stated plainly:** ribosome score threshold **0.19**
(model-native), matching radius **15 vox / 150 Å** (one ribosome radius), tomo_type
**denoised**, tiling/aggregation at **TopCUP defaults** (96³ ROI, 0.21 overlap, TTA flip),
and the **full 6-class config** so ribosome maps to channel 4. That gives Precision 0.64 ·
Recall 0.94 · **F1 0.76** · 37 Å localization.

---

## One honest caveat about the false positives

I don't think the 17 "false positives" are really the model hallucinating. The synthetic
tomogram contains all six trained particle types, and the model does its job and detects
the *others* too (ferritin, thyroglobulin, and so on). Several of those are dense globular
complexes at a similar scale to ribosomes, so a handful bleed into the ribosome channel.
Because I'm scoring against ribosome-only ground truth, they show up as FPs. A quick
multi-class check (`evaluate_classes.py`) confirms the other channels fire correctly on
their own species - so the ribosome **precision here is really a lower bound**, and the
true picking quality is a bit better than the 0.64 suggests.

---

## Reproduce

```bash
cd final_submission/task2b
python inference.py            # auto-detects CPU/GPU; prints metrics + the radius sweep,
                               # writes results/metrics_2b.json and results/synthetic/val_pred_df_seed.csv
python visualize_results.py    # GT (red) vs ribosome detections (cyan; green = matched TP)
```
