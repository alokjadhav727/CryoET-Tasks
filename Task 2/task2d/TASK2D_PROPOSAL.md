# Task 2d — Proposal: Closing the Gap to In-Situ Cellular Data

This is a plan, not an implementation. The goal is to take the model from Task 2b/2c —
which works on the synthetic dataset but finds nothing on the real Chlamydomonas data —
and make it actually detect ribosomes in the cellular setting.

I'm writing this off the back of what we already learned in Task 2c, so let me start there,
because the diagnosis is what makes the plan concrete rather than a wishlist.

---

## What we're actually up against

From Task 2c, the model didn't fail in some subtle "it's a bit worse" way — its confidence
**collapsed**. On the synthetic data it fires at ~0.998 on real ribosomes; on the cell
chunk the strongest response anywhere is ~0.008, about 125× lower, and nothing clears the
detection threshold. So this isn't a tuning problem we can fix with a slider. The model has
simply never seen data that looks like the cell, and three concrete differences explain it:

1. **The cell is crowded.** Ribosomes sit shoulder-to-shoulder with hundreds of other
   particles and dense structure. The training data had well-separated particles.
2. **The ribosomes barely stand out** — about 1.2× above their surroundings in the cell
   versus 2.7× in the synthetic data.
3. **The images look different overall** — different texture, different frequency content,
   produced by a different acquisition and processing pipeline.

And the hard constraint the task calls out: there's **no big clean labeled set** for this
regime, and we **can't trust another picker's output as ground truth** (it would just bake
in someone else's bias). So whatever we propose has to work with very little trusted
labeled data.

That framing points pretty directly at the plan: don't fight the model, **move its training
distribution toward the cell**, and lean on the small amount of trusted labels we do have.

---

## The proposal, in four parts

### Part 1 — Get a small, trustworthy foothold of real labels

We don't need a huge labeled set; we need a *clean* one. The most reliable signal we have
is exactly what Task 2a gave us: the portal's expert ribosome annotations on real
tomograms (202 on this Chlamydomonas run, 56 in our chunk). I'd treat those as the gold
set — not for training a model from scratch, but for two jobs: a small fine-tuning set and,
more importantly, a trustworthy *evaluation* set we never train on.

Practically:
- **Hold out** a portion of the expert-labeled real tomograms purely for evaluation, and
  never let the model see them in any training step. This is the only honest way to know if
  we're actually improving.
- Keep the rest as a small, high-quality fine-tuning pool.

The point is to anchor everything to expert labels we trust, rather than another model's
guesses.

### Part 2 — Make the training data look more like the cell

This is the biggest lever, because the core problem is that the model trained on one
distribution and we're testing on another. Two complementary ways to close that:

- **Fine-tune (domain adaptation), don't retrain.** Start from the existing TopCUP weights
  — they already know what a ribosome roughly looks like — and continue training on the
  small set of real labeled tomograms. We're nudging the model toward the cell, not
  teaching it from zero, which is exactly what you want when labeled data is scarce. I'd
  keep the learning rate low, freeze the early layers, and only let the later layers adapt,
  so we don't wash out the useful features it already has.

- **Improve the synthetic data so it's a fairer rehearsal.** Right now the synthetic
  training data is too clean. If we add the things that actually differ — crowd the volumes
  with lots of particles and surrounding structure, add realistic noise, and reproduce the
  acquisition artifacts we measured in the FFT comparison — then "synthetic" training starts
  to resemble "real" testing. The nice part is this needs **no new labels** (we generate the
  data, so we know where every particle is), and it directly targets the three differences
  we measured.

In practice I'd do both: pre-train / continue-training on the more realistic synthetic data
to get the model into the right regime cheaply, then fine-tune on the small real labeled set
to land it on the actual cell distribution.

### Part 3 — Use the unlabeled real data we have plenty of

We have far more real tomograms than we have labels for. We can put that to work without
trusting any other picker:

- **Self-training with confidence.** Run our best current model on unlabeled real
  tomograms, keep only the *high-confidence* detections as tentative "pseudo-labels," add
  those to the fine-tuning set, and retrain. Repeat. The key safeguard is to only trust the
  model's most confident picks and to keep a human-verified evaluation set on the side, so
  we can catch it if this starts drifting in a bad direction.
- This is a way to grow the effective training set using the abundant unlabeled data, while
  the expert labels keep us honest.

### Part 4 — Fix the operating point and the post-processing for the cell

Even a well-adapted model needs its output interpreted correctly for this data:

- **Recalibrate the threshold.** The 0.19 cutoff was tuned on the synthetic data. Once the
  model is adapted, I'd re-pick the threshold on the held-out *real* labels — the right
  operating point for the cell is almost certainly different.
- **Account for crowding in the peak-finding.** The step that turns the heatmap into
  discrete picks assumes particles are reasonably separated. In a crowded cell that can
  merge neighbors. I'd revisit the suppression radius so two adjacent ribosomes don't get
  collapsed into one.

---

## How I'd know if it actually worked

This matters as much as the method — it's easy to fool yourself here. The whole evaluation
hangs on **one rule: only ever measure on the held-out, expert-labeled real tomograms that
the model never trained on.** No pseudo-labels, no other picker's output, in the test set.

Concretely:
- **Use the same metrics as Task 2b/2c so everything is comparable** — Precision, Recall,
  F1 at the one-ribosome-radius matching distance, plus localization error. The single
  headline number is "did F1 on the real chunk move off 0.00 toward the 0.76 we get on
  synthetic data?"
- **Always compare against the baseline** (the current, un-adapted TopCUP on the same chunk)
  so any gain is clearly attributable to our changes.
- **Watch the confidence histogram, not just F1.** Task 2c showed the failure as a collapse
  from ~0.998 to ~0.008. A successful adaptation should visibly pull that distribution back
  up above the threshold — that's a sensible early signal even before F1 fully recovers.
- **Sanity-check by eye.** Re-run the same GT-vs-prediction slice overlays we built in
  Task 2c on the held-out tomograms; if the green "matched" circles start showing up where
  before there were only red "missed" ones, we're moving the right way.
- **Be honest about precision/recall trade-offs.** For this kind of work, recovering most of
  the real ribosomes (recall) usually matters more than a few extra false picks, so I'd
  report the operating point explicitly rather than hide it inside a single F1.

---

## Practical constraints I'm keeping in mind

- **Labels are the bottleneck, not compute.** The plan is built to need as few trusted
  labels as possible — fine-tuning instead of retraining, better synthetic data that needs
  no labels, and self-training on unlabeled data — precisely because there's no clean
  large-scale real training set.
- **Don't trust other pickers as truth.** They're fine as a rough starting point for
  pseudo-labels (filtered by confidence and human spot-checks), never as the evaluation
  ground truth.
- **Keep it incremental.** Each part (fine-tune → better synthetic → self-train → recalibrate)
  can be added one at a time and measured against the held-out set, so we always know which
  change bought which improvement, and we can stop when the gains level off.

The throughline is simple: the model failed because of **data drift** between the dataset it
trained on and the one we tested it on, so the fix is to **shrink that drift** — partly by
making the training data look more like the cell, partly by adapting the model on the small
amount of real labeled data we trust — and to **measure progress only against expert labels
the model never saw.**
