# Iteration 005 — cifar10_iter005_adamw (Cell D — +adamw)
Date: 2026-04-27 08:15 | GPU: 0 | Duration: ~36 min (60 epochs × ~35 s/epoch)

## 1. Hypothesis
Iter 005 targets **Cell D — +adamw** (program.md §Required ablation
strategy): single-axis delta vs Cell B (iter 003) is
`training.optimizer: sgd → adamw`, with the catalog-bounded `training.lr:
0.1 → 1.0e-3` (program.md §A4 sets AdamW range to {1e-3, 5e-4}); momentum/
nesterov fields are SGD-only and dropped. All other knobs (aug=standard,
cosine, 60 ep, wd=5e-4, seed=42) match Cell B. Per the iter-004 §8 forecast
this row is expected to **Fail or Partial**: AdamW is a known
underperformer vs tuned SGD+momentum on CIFAR-10 ResNets, with predicted
test_acc 0.93–0.945 (Δ vs Cell B between −1.5 and 0 pp) and a train-vs-test
gap roughly Cell B-like (~5 pp, since A1 stays "standard"). The point of
running it is **matrix completion** for the A2 axis, not a winner attempt.

## 2. Falsification criterion
The hypothesis is **expected-negative**, so falsification is asymmetric:
the AdamW-underperforms-on-CIFAR-ResNet prior would be refuted if test_acc
≥ 0.9531 (≥ Cell B + 0.5 pp = matches/beats Cell C as the new best). It
would be over-confirmed (i.e. an even stronger negative than expected) if
test_acc < 0.93 — that would warrant a quick LR sanity check before
discarding the cell. Mechanism check: the train-vs-test gap should land in
the same neighborhood as Cell B (~5 pp); a *much* tighter gap with low acc
would mean AdamW under-fit, while a much wider gap would mean AdamW
over-fit faster (either would change the next-step decision). Outcome: see
§4 — falsification cleared on the headline (acc landed inside the
predicted 0.93–0.945 band) but the gap came in **wider** than predicted.

## 3. Changes made
New file `configs/ablation/iter005_adamw.yaml` (already present at iter
launch). Diff vs `configs/ablation/iter003_std_aug.yaml`:

```diff
 training:
-  optimizer: sgd
-  lr: 0.1
-  momentum: 0.9
-  nesterov: true
+  optimizer: adamw
+  lr: 1.0e-3
   weight_decay: 5.0e-4
   scheduler: cosine
   epochs: 60
```

`exp_name` updated to `cifar10_iter005_adamw`. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far (Cell B) | This run (Cell D) | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | **0.9379** | **−0.0102** | **+0.0509** |
| final_acc  | 0.8868 | 0.9478 | 0.9377     | −0.0101     | +0.0509     |
| test_loss  | 0.4231 | 0.2106 | 0.4271     | +0.2165     | +0.0040     |
| train_acc  | 1.0000 | 0.9993 | 0.9997     | +0.0004     | −0.0003     |
| train_loss | 0.0014 | 0.0037 | 0.0012     | −0.0025     | −0.0002     |
| best_epoch | 52 / 60 | 57 / 60 | 51 / 60   | —           | —           |
| epochs run | 60      | 60      | 60         | —           | —           |

Source: `runs/cifar10_iter005_adamw/final.pth`,
`ckpt['metrics'] = {'acc': 0.9377, 'loss': 0.4271, 'best_acc': 0.9379,
'best_epoch': 51}`. `history.json` (60 rows) shows AdamW reaches 0.85+
test_acc by epoch 10 (faster early than Cell B), then **plateaus much
earlier** — by epoch 30 test_acc=0.9245, best lands at epoch 51 (vs Cell B
57), and the last 9 epochs of cosine tail give essentially zero further
gain. Train_acc is fully saturated by epoch 50 (≥ 0.999), but unlike Cell
B the **test_loss climbs back from 0.33 (ep 30) to 0.43 (ep 59)** even as
test_acc stays roughly flat — classic adaptive-optimizer overfit signature
where the model becomes more confident on its mistakes.

The headline mechanism — "AdamW under-performs SGD+momentum on this
recipe" — fired exactly: test_acc landed at 0.9379, comfortably inside
the 0.93–0.945 predicted band, ~1 pp short of Cell B. The **gap landed
wider than predicted**: train 0.9997 vs test 0.9377 ⇒ **6.2 pp** (vs
Cell B's 5.2 pp and the predicted ~5 pp). AdamW with the catalog-bounded
LR=1e-3 effectively over-fits *more* than SGD+momentum at the same
weight_decay, despite a slower optimization start. The Δ vs Cell B is
−1.04 pp — well outside the ±0.3 pp Noise band, **drops > 0.5 pp** vs
current best, so this is a **Failure** verdict for matrix purposes. Cell B
(0.9481) remains "Best so far".

## 5. Visualization evidence

**Per-class (`figs/iter_005/per_class.csv`).** Spread 0.874 (cat) →
0.969 (automobile) ≈ 9.5 pp, similar to Cell B's 9.7 pp. The headline
fact: **9 of 10 classes regressed vs Cell B; only deer improved
(+0.3 pp)**. This is a *uniform across-the-board degradation*, not the
class-specific rearrangement Cell C (autoaug) showed. Per-class Δ vs
Cell B's inferred 60-ep std-aug per-class accs (reconstructed from
iter003/iter004 logs):
- airplane **−2.7 pp** (0.962 → 0.935) ⚠
- dog      **−2.8 pp** (0.930 → 0.902) ⚠
- horse    −1.1 pp (0.966 → 0.955)
- bird     −0.9 pp (0.925 → 0.916)
- truck    −0.9 pp (0.964 → 0.955)
- ship     −0.7 pp (0.970 → 0.963)
- automobile −0.6 pp (0.975 → 0.969)
- frog     −0.6 pp (0.956 → 0.950)
- cat      −0.4 pp (0.878 → 0.874)
- deer     **+0.3 pp** (0.957 → 0.960)

Two surprises: (a) the biggest regressions are airplane and dog —
neither of which is the worst-class cat, and both of which Cell C also
regressed (airplane −0.6, dog −1.5), suggesting these two classes are
genuinely sensitive to the optimizer/regularizer combo on this recipe;
(b) cat barely moved (−0.4 pp), because cat is already the floor and
SGD's edge over AdamW manifests on the classes that have headroom, not
the saturated ones or the deepest-confusion one.

**t-SNE (`figs/iter_005/tsne.png`).** Eight clusters cleanly isolated
(automobile/truck pair, frog, bird, deer, cat/dog pair, horse), but a
**new airplane↔ship contamination zone has opened on the right side**
of the plot — several ship (yellow) points sit inside the airplane
(blue) cluster and vice versa, with a clear bridge of mixed labels
between them. This is the kind of degradation that explains the
airplane −2.7 pp regression: AdamW's penultimate features for airplane
and ship are no longer linearly separable the way Cell B's were. The
**cat↔dog bridge is still present** with roughly the same width as
Cell B/C — AdamW didn't help this confusion any more than autoaug
did, consistent with the per-class numbers. Automobile↔truck also
shows visible intermixing in the top cluster (truck cyan points
embedded in orange automobile region), again consistent with the
truck −0.9 / auto −0.6 regressions.

**Grad-CAM (`figs/iter_005/cam.png`).** 8/8 correct — no class errors
on the random sample, the same headline as Cell B. Heatmaps localize
plausibly on each object (ship → superstructure, frog → body,
airplane → fuselage, automobile → engine bay). The qualitative
difference vs Cell B/C: heatmaps under AdamW look **more uniformly
round and slightly more central** — peaks are compact circular blobs
in the center half of the panel rather than the elongated, off-center
peaks Cell B produced for the airplane-down-low and frog-upper-left
panels. The Cell A "look-at-the-middle prior" hasn't fully returned
(off-center objects still get correct classification), but the
heatmap *signature* sits closer to Cell A's center-bias than to
Cell B's location-following pattern. Plausible mechanism: AdamW's
per-parameter adaptive scaling damps gradients on the spatially
peripheral feature channels relative to SGD's uniform LR, so the
network puts proportionally more weight on the always-active central
channels.

So Cell D's degradation is real and visible at every level: −1 pp
headline acc, uniform per-class regression with airplane/dog as the
worst hits, a new airplane↔ship feature-space bridge, and a Grad-CAM
heatmap signature that has drifted back toward central-bias. Mechanism
prediction (AdamW under-performs SGD+momentum on this recipe) confirmed
in full.

## 6. Verdict
**Failure** — best_acc dropped −1.04 pp vs current best Cell B
(0.9481 → 0.9379), well past the ±0.3 pp Noise band and well past the
−0.5 pp Failure threshold. The §Verdict §Failure criterion as
interpreted for negative-expected hypotheses: Δ vs best is firmly
negative, the cell cannot be a winner on this matrix, and the
mechanism (AdamW under-performs tuned-SGD on CIFAR-10 ResNets) was
confirmed exactly as predicted. The prediction itself was *correct*
(landed in the 0.93–0.945 band) — but the verdict labels the run, not
the prediction. The matrix gains a clean negative datapoint on the A2
axis: AdamW is **not** the right optimizer for this recipe at the
catalog-bounded LR.

## 7. Decision
**Discard for hardening; keep as a documented A2-axis negative.** Cell B
(0.9481) remains "Best so far"; do not propagate AdamW to any
downstream cell. Update CLAUDE.md matrix row **D** with `acc=0.9379,
Best iter#=5, Verdict=Failure`. Do not bother with the second AdamW LR
(5e-4) unless a future iter explicitly needs more A2 evidence — the
expected delta of a halved LR with a 6 pp train-test gap is more
under-fit, not less, so the A2 axis is effectively pinned at SGD. The
genuinely high-leverage remaining cells are **Cell F (long-train, 100
ep on std-aug)** — already pre-staged as `iter006_long_train.yaml` and
*currently running on GPU 1 per state/iterations.tsv* — and **Cell E
(multistep schedule)**. Cell F is the correct next cell because iter
003/004 both showed std-aug/autoaug under-fit at 60 ep (best_epoch
landed at 57 / 58 with cosine still descending), so adding 40 epochs
should buy real headroom; Cell E is lower-leverage on the same recipe.

## 8. Next hypothesis
Iter 006 (already running, Cell F — long-train): single-axis delta vs
Cell B is `training.epochs: 60 → 100` (cosine extends accordingly), with
all other knobs identical to iter003. Predicted: **Partial or Success.**
Expected test_acc 0.950–0.955 (i.e. Δ vs Cell B between +0.2 and +0.7 pp);
mechanism = the Cell B run's best landed at epoch 57/60 (cosine still
falling, train_acc only 0.9993), so the model has under-fit headroom that
40 more epochs of cosine should consume. Falsifier: if iter 006 lands
≤ 0.9481 (no gain from 100 ep), then std-aug at 60 ep was already
saturated and the small under-fit signal in the iter-003 history was
noise. If it lands > 0.9528 (above Cell C), Cell F becomes a candidate
for the 2-seed hardening pass alongside Cell C. After iter 006 lands,
iter 007 is Cell E (multistep) for matrix completion.
