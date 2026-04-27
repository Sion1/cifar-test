# Iteration 006 — cifar10_iter006_long_train (Cell F — long-train)
Date: 2026-04-27 08:42 | GPU: 1 | Duration: ~56 min (100 epochs × ~33 s/epoch)

## 1. Hypothesis
Iter 006 targets **Cell F — long-train** (program.md §Required ablation
strategy): single-axis delta vs Cell B (iter 003) is `training.epochs:
60 → 100` (cosine schedule extends in proportion); everything else
identical (aug=standard, sgd 0.1, momentum 0.9, wd 5e-4, nesterov,
seed=42). Mechanism: iter 003 reported best_epoch=57/60 with cosine
still descending and train_acc=0.9993 — i.e. Cell B looked under-fit
on the recipe's own schedule, so the predicted +0.2…+0.7 pp came from
"40 more epochs of cosine should consume the remaining headroom."
Predicted: **Partial or Success**, expected test_acc 0.950–0.955.

## 2. Falsification criterion
The iter-005 §8 falsifier was explicit: **"if iter 006 lands ≤ 0.9481
(no gain from 100 ep), then std-aug at 60 ep was already saturated and
the small under-fit signal in the iter-003 history was noise."** A
strong over-confirmation (≥ 0.9531, the ≥Cell C threshold) would make
Cell F a 2-seed hardening candidate alongside Cell C. Mechanism check:
the train-vs-test gap at 100 ep should approach Cell B's 5.2 pp from
*below* (gap shrinks as more cosine refinement occurs); a *wider* gap
with no acc gain would mean the longer schedule is not consuming
under-fit headroom but instead trading it for overfit.

## 3. Changes made
New file `configs/ablation/iter006_long_train.yaml`. Diff vs
`configs/ablation/iter003_std_aug.yaml`:

```diff
 training:
   optimizer: sgd
   lr: 0.1
   momentum: 0.9
   weight_decay: 5.0e-4
   nesterov: true
   scheduler: cosine
-  epochs: 60
+  epochs: 100
```

`exp_name` updated to `cifar10_iter006_long_train`. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far (Cell B) | This run (Cell F) | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | **0.9465** | **−0.0016** | **+0.0595** |
| final_acc  | 0.8868 | 0.9478 | 0.9457     | −0.0021     | +0.0589     |
| test_loss  | 0.4231 | 0.2106 | 0.2237     | +0.0131     | −0.1994     |
| train_acc  | 1.0000 | 0.9993 | 0.9997     | +0.0004     | −0.0003     |
| train_loss | 0.0014 | 0.0037 | 0.0018     | −0.0019     | +0.0004     |
| best_epoch | 52 / 60 | 57 / 60 | 95 / 100 | —           | —           |
| epochs run | 60      | 60      | 100        | —           | —           |

Source: `runs/cifar10_iter006_long_train/final.pth`,
`ckpt['metrics'] = {'acc': 0.9457, 'loss': 0.2237, 'best_acc': 0.9465,
'best_epoch': 95}`. `history` (100 rows) shows test_acc rises slowly:
0.7343 (ep10) → 0.8157 (ep30) → 0.9037 (ep60) → 0.9245 (ep80) →
plateau in the 0.945–0.946 band over the last 10 epochs (mean 0.9456).
Note that **at epoch 60 — the natural Cell B comparison point — this
run's test_acc is only 0.9037, ~4.4 pp below Cell B's 0.9478**: the
slower 100-ep cosine schedule keeps LR high longer, so the same
optimization horizon converges to a noticeably worse intermediate
checkpoint. Best lands at epoch 95/100 (cosine effectively flat by
then), train_acc 0.9997 (fully memorized like Cell A/D, slightly more
than Cell B's 0.9993). Train-vs-test gap = 5.40 pp, **comparable to
Cell B's 5.2 pp** — the predicted "gap shrinks" mechanism did not
fire.

The headline mechanism — "40 more epochs consume Cell B's under-fit
headroom" — **failed to fire**: best_acc 0.9465 < Cell B's 0.9481, Δ
= −0.16 pp. This Δ falls inside the §Verdict §Noise band (|Δ| <
0.3 pp), but the §Verdict §Failure clause "mechanism doesn't fire on
an expected-positive hypothesis" is also satisfied. Per the iter-005
§8 falsifier ("≤ 0.9481 ⇒ Cell B was already saturated"), this run
clears the falsifier and so the longer-schedule prior is **rejected**.
Verdict = **Failure** — magnitude is within seed variance, but the
predicted positive did not materialize and Cell F cannot be a winner
candidate. Cell B (0.9481) remains "Best so far."

## 5. Visualization evidence

**Per-class (`figs/iter_006/per_class.csv`, with Cell B re-measured
from `runs/cifar10_iter003_std_aug/best.pth` for honest Δ).** Spread
0.893 (cat) → 0.987 (automobile) ≈ 9.4 pp, ~ Cell B's 9.7 pp.
Per-class Δ vs Cell B (this iter − Cell B, in pp):
- airplane:    0.962 → 0.962  (**+0.0**)
- automobile:  0.975 → 0.987  (**+1.2**) ⬆
- bird:        0.925 → 0.918  (−0.7)
- cat:         0.878 → 0.893  (**+1.5**) ⬆ — cat improved!
- deer:        0.957 → 0.960  (+0.3)
- dog:         0.930 → 0.905  (**−2.5**) ⬇
- frog:        0.956 → 0.965  (+0.9)
- horse:       0.966 → 0.962  (−0.4)
- ship:        0.967 → 0.958  (−0.9)
- truck:       0.964 → 0.954  (−1.0)

Two surprises versus the iter-005 §8 expectation: (a) **cat actually
*gained* +1.5 pp** under long-train (the *only* matrix-cell so far
where cat moves meaningfully), but (b) **dog regressed −2.5 pp**, and
the top off-diagonal confusions in the confusion matrix are still
`dog→cat` (60) and `cat→dog` (59) — **the cat↔dog confusion is now
asymmetric**: the longer schedule *re-allocated* errors from cat-mis-
classified-as-dog into dog-misclassified-as-cat, leaving the joint
error roughly conserved. So extra epochs polished the cat manifold
at dog's expense, rather than carving cleaner boundaries. Other
notable losses are spread across saturated vehicle/animal classes
(truck −1.0, ship −0.9, bird −0.7) that had little headroom; these
small per-class regressions sum to the −0.16 pp headline. No class
fell off a cliff and no class jumped out — consistent with a Noise-
band magnitude.

**t-SNE (`figs/iter_006/tsne.png`).** Ten clusters generally well-
isolated. Confirmed: **cat↔dog bridge is still present** with a
narrow strip of mixed-color points connecting the cat (red) and
dog (brown) clusters along the left side, just like Cell B. The
**airplane↔ship contamination zone has partially reopened**:
several yellow ship points sit on the right edge of the blue
airplane cluster (visible in the upper-right of the plot), milder
than Cell D's (iter 005) airplane↔ship bridge but more visible than
Cell B's. This matches the per-class numbers (airplane held at
0.962 but ship dropped −0.9 pp) — the longer schedule has slightly
eroded the airplane↔ship boundary even though the recipe is identical
to Cell B. Automobile↔truck remains cleanly separated (orange and
cyan well apart, only one truck point bleeding into automobile),
contrary to Cell D's intermixing — so SGD-with-cosine preserves the
vehicle-pair separation that AdamW broke. Bird (green) has a couple
of cat-red points stranded inside it, and the deer-dog corridor
along the left has a few mixed mid-strip points (deer/cat/dog mix).

**Grad-CAM (`figs/iter_006/cam.png`).** **8/8 correct** (matches
Cell B). However the **heatmap signature has drifted back toward
Cell A/D's center-bias**, NOT preserving Cell B's location-following
shape — peaks across all 8 panels are compact, roughly circular red
blobs near the image center. The airplane (top-row) panel is the
clearest tell: the plane sits in the lower half of the frame but the
heatmap peak is centered on the middle of the image, not following
the plane down. Similarly the row-3 frog is upper-left in the frame
but the heatmap is mid-frame. This is the *same* signature drift
iter 005 (Cell D, AdamW) flagged — and it's a surprise here because
A1=standard and A2=sgd are both unchanged from Cell B. Plausible
mechanism: with 100 epochs, the cosine schedule keeps LR moderate
for ~2× longer than the 60-ep schedule, giving the BN-running-stats
+ classifier head much more time to over-rely on the
always-active center channels (which dominate gradient mass under
std aug's pad-4 RandomCrop). So even though the recipe nominally
preserves spatial-invariance, the longer optimization horizon
*erodes* the off-center prior that Cell B's faster cosine had
locked in. **This is the most important non-headline finding from
iter 006**: heatmap shape is sensitive not just to A1/A2 but
also to A6 (epochs).

So Cell F's degradation is small (−0.16 pp) but visible at every
level: per-class regression concentrated on dog (−2.5 pp) with cat
trading places, t-SNE airplane↔ship boundary partially eroded, and
Grad-CAM heatmaps drifting back toward central blobs. Mechanism
prediction (extra epochs consume Cell B's under-fit headroom)
falsified.

## 6. Verdict
**Failure** — the predicted-positive mechanism (40 more epochs of
cosine consume Cell B's under-fit headroom, lifting acc into the
0.950–0.955 band) did not fire. best_acc 0.9465 < Cell B 0.9481
(Δ = −0.16 pp), the iter-005 §8 falsifier "≤ 0.9481 ⇒ Cell B was
already saturated and the small under-fit signal in iter-003's
history was noise" is cleared, and the train-vs-test gap stayed at
~5.4 pp (no shrinkage from more refinement, which the mechanism
required). The Δ magnitude itself is within the §Noise band
(|Δ| < 0.3 pp), so a 2-seed replay *could* still flip the sign by
±0.2 pp; but at the matrix-completion level Cell F is a clean
mechanism failure and not a winner candidate.

The structural finding worth carrying forward: **on this recipe the
60-ep cosine schedule is not under-fit**; lengthening to 100 ep keeps
LR high longer (epoch-60 checkpoint of this run is 0.9037, ~4.4 pp
below Cell B's epoch-60 of 0.9478) and the extra 40 epochs only
recover what the slower decay schedule lost, finishing slightly
*below* Cell B. The falsifier did its job: iter-003's "best_epoch
57/60 with cosine still descending" was a *cosine-shape* artifact, not
a real under-fit signal — at any epochs setting the curve will look
"still descending" near the end because cosine derivative → 0.

## 7. Decision
**Discard Cell F as a winner candidate; keep as the A6=epochs negative
datapoint.** Cell B (0.9481) remains "Best so far"; do not propagate
A6=100 to any downstream cell. Update CLAUDE.md matrix row **F** with
`acc=0.9465, Best iter#=6, Verdict=Failure`. The genuine remaining
high-leverage matrix cell is **Cell E (multistep schedule)** — A3 is
the only single-axis catalog axis still untouched on the std-aug
recipe, and multistep's discrete LR drops sometimes outperform cosine
at 60 ep on CIFAR ResNets when the milestones align with the
generalization plateau the loss curve hits around epoch 30–40. If
Cell E also fails, phase 1 ends with B (0.9481) and C (0.9528) as the
2-seed hardening candidates. Do **not** combine A1=autoaug + A6=100
or A6=100 + A3=multistep: phase 1 is single-axis only.

## 8. Next hypothesis
Iter 007: **Cell E — multistep**. Single-axis delta vs Cell B is
`training.scheduler: cosine → multistep` (with milestones at
[30, 45] and gamma=0.1, the standard CIFAR ResNet recipe within the
catalog A3 axis). All other knobs identical to iter003. Predicted:
**Partial** (most likely) — multistep's discrete drops typically
land within ±0.3 pp of cosine on this recipe at 60 ep; a small win
is possible if the model is mid-plateau at epoch 30 and the LR drop
breaks it through. Falsifier: if iter 007 lands ≤ 0.9451 (Cell B
−0.3 pp, outside Noise band on the negative side), multistep is a
clean negative; if it lands ≥ 0.9531 (Cell C threshold), it joins
the 2-seed hardening candidates. After iter 007, phase 1 (Cells A–F
+ E) is complete and the 2-seed hardening pass begins, alongside
updating CLAUDE.md "Current best" with the 2-seed mean.
