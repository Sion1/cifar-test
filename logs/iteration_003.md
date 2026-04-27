# Iteration 003 — cifar10_iter003_std_aug (Cell B — +std aug)
Date: 2026-04-27 07:09 | GPU: 1 | Duration: ~34 min (60 epochs × ~31.5 s/epoch)

## 1. Hypothesis
Iter 003 targets **Cell B — +std aug** (program.md §Required ablation
strategy): single-axis delta vs Cell A (iter 002) is
`data.augmentation: none → standard` (pad-4 RandomCrop + HorizontalFlip);
all other knobs (sgd 0.1, momentum 0.9, wd 5e-4, nesterov, cosine, 60 ep,
seed=42) match iter 002. This row quantifies the **standalone contribution
of A1** on top of the bare baseline. Expected: acc rises ~0.93–0.94, the
~11.3 pp train-vs-test gap shrinks to ~3–4 pp, and (per iter 002's visual
prediction) Grad-CAM peaks shift off-center on at least some panels.

## 2. Falsification criterion
Refuted if any of: (a) test_acc ≤ 0.892 (i.e. ≤ Cell A floor + 0.5 pp —
Failure under §Verdict criteria for an "expected positive" axis);
(b) train-vs-test gap remains ≥ 8 pp (would mean RandomCrop+HFlip didn't
regularize); (c) Grad-CAM heatmaps remain center-locked on every panel
(would mean the spatial-invariance mechanism didn't fire even though acc
moved); (d) `final.pth` missing `metrics`/`history`. None triggered — see §4.

## 3. Changes made
New file `configs/ablation/iter003_std_aug.yaml`. The only delta vs
`configs/ablation/iter002_bare_baseline.yaml` is one line:

```diff
 data:
-  augmentation: none
+  augmentation: standard
```

`exp_name` and the comment block updated accordingly. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.8870 | **0.9481** | **+0.0611** | **+0.0611** |
| final_acc  | 0.8868 | 0.8868 | 0.9478     | +0.0610     | +0.0610     |
| test_loss  | 0.4231 | 0.4231 | 0.2106     | −0.2125     | −0.2125     |
| train_acc  | 1.0000 | 1.0000 | 0.9993     | −0.0007     | −0.0007     |
| train_loss | 0.0014 | 0.0014 | 0.0037     | +0.0023     | +0.0023     |
| best_epoch | 52 / 60 | 52 / 60 | 57 / 60   | —           | —           |
| epochs run | 60     | 60      | 60         | —           | —           |

Source: `runs/cifar10_iter003_std_aug/final.pth`,
`ckpt['metrics'] = {'acc': 0.9478, 'loss': 0.2106, 'best_acc': 0.9481,
'best_epoch': 57}`. `history.json` (60 rows) shows epoch 0
test_acc=0.3052 (vs 0.2691 for Cell A — small gain at init from the
augmentation acting as a regularizer even before the schedule kicks in),
mid-train (epoch 25) test_acc=0.877 (already approaching Cell A's final),
epoch 35 test_acc=0.897, epoch 45 test_acc=0.930, plateau begins ~epoch
50 at 0.94+ and best lands at epoch 57.

The **train-vs-test gap collapsed**: train_acc=0.9993, test_acc=0.9478 ⇒
~5.2 pp gap, less than half of Cell A's 11.3 pp. Test_loss halved
(0.4231 → 0.2106). Best_epoch shifted from 52 → 57, consistent with
augmentation slowing memorization and letting the cosine tail
contribute. Falsification criteria (a)–(c) all unmet — see §5 for the
Grad-CAM spatial check.

## 5. Visualization evidence

**Per-class (`figs/iter_003/per_class.csv`).** Spread shrank from
~20 pp (Cell A) to ~10 pp: 0.878 (cat) → 0.975 (automobile). The
*ranking* matches Cell A exactly, but every class moved up — and the
hard classes moved most:
- cat **+13.0 pp** (0.748 → 0.878)
- bird **+10.6 pp** (0.819 → 0.925)
- dog **+10.0 pp** (0.830 → 0.930)
- deer **+7.2 pp**, airplane **+4.9 pp**, horse **+4.5 pp**, truck
  **+3.2 pp**, automobile **+2.9 pp**, frog **+2.4 pp**, ship **+2.3 pp**.

This is the textbook regularizer signature: **the saturated vehicle
classes had little room to gain (≤ +5 pp); the small-mammal/bird
classes that were starving for invariance gained 10–13 pp.** The
prediction in §5-skeleton ("cat → ~0.88+") landed almost exactly. Cat
remains the hardest, but the cat–next-class gap closed from 7.1 pp
(Cell A) to 4.7 pp.

**t-SNE (`figs/iter_003/tsne.png`).** All 10 classes form denser,
more-isolated clusters than Cell A. Vehicles are now spread around
the outer perimeter (airplane top-center, ship top-right, truck
mid-right, automobile bottom-right) instead of huddled on one side,
and animals fill the interior. The two contamination zones from Cell
A behave differently:
- **cat ↔ dog still touch** — the red (cat) and brown (dog) clusters
  share an edge with a small visible bridge of mixed points; this
  is consistent with cat staying the worst class and the +AA / +long
  cells having clear remaining mechanism to attack.
- **airplane ↔ ship bridge is gone** — the two clusters are now
  fully separated in the upper region. The horizontal-silhouette
  contamination from Cell A has been absorbed, presumably because
  HFlip taught the model that "ship-vs-plane" doesn't depend on
  orientation. Bird is also markedly cleaner from airplane.

**Grad-CAM (`figs/iter_003/cam.png`).** 8/8 correct (vs 7/8 in Cell A).
**The center-bias signature is broken** — the iter-002 visual
prediction lands. Concrete shifts:
- airplane panel: heatmap follows the plane down to the lower-left
  third of the frame (Cell A would have peaked at center regardless
  of where the plane sat).
- frog (row 4): heatmap shifts to the upper-left where the frog
  actually is.
- ship (row 3): heatmap stretches horizontally along the ship hull
  rather than collapsing to a central blob.
- cat / frog (row 3): peaks still near center, but the centered
  examples *should* peak there — the difference is that off-center
  examples now follow the object, which is the falsifiable claim.

So +std aug fired on its hypothesized mechanism: the model learned a
spatial prior that adapts to object location, not a "look at the
middle" prior. This is the single most important visual confirmation
of the iter-002 prediction.

## 6. Verdict
**Success** — acc jumped +6.11 pp vs Cell A (0.8870 → 0.9481), well above
the +0.5 pp threshold in §Verdict criteria; mechanism evidence (gap
collapse 11.3 → 5.2 pp, test_loss halved, best_epoch pushed later)
matches the hypothesis exactly. The hypothesis-level expectation
("~0.93–0.94, gap ~3–4 pp") is slightly *exceeded* on acc and slightly
*undershot* on gap closure (5.2 vs 3–4 pp predicted) — consistent with
"standard" being weaker than autoaug, leaving residual overfit room for
Cell C / Cell F to attack.

## 7. Decision
**Keep & propagate.** This row replaces Cell A as **best so far** in the
matrix (Cell B = 0.9481). Cell A stays the floor; every later cell must
beat 0.9481 to count as the new best. The 5.2 pp residual gap means the
matrix is *not* converged — Cell C (autoaug, iter 004 already running on
GPU 2) and Cell F (long-train at 100 ep) both have a clear mechanistic
target (further gap closure). CLAUDE.md matrix row **B** gets
`acc=0.9481, Best iter#=3, Verdict=Success`. The Cell A baseline target
(≥ 0.94 in program.md §Goal) is **already cleared** at Cell B.

## 8. Next hypothesis
Iter 004 (Cell C, autoaug) is already running on GPU 2 — its analysis
will land next loop tick. Predicted vs Cell B: +0.5 to +1.5 pp acc
(autoaug attacks color/contrast invariance that pad-crop+HFlip can't),
gap shrinks further to ~3 pp. After iter 004 lands, the natural Cell D
(`+adamw`, `configs/ablation/iter005_adamw.yaml` already staged) becomes
the next iter; given Cell B already cleared the program-level 0.94 acc
target, the matrix should be completed for evidence even if no later
cell beats Cell C.
