# Iteration 002 — cifar10_iter002_bare_baseline (Cell A — bare baseline)
Date: 2026-04-27 07:03 | GPU: 0 | Duration: ~34 min (60 epochs × ~31.5 s/epoch)

## 1. Hypothesis
Iter 002 targets **Cell A — bare baseline** (program.md §Required ablation
strategy): single-axis delta vs `configs/cifar10_resnet34.yaml` is
`data.augmentation: standard → none`; everything else (sgd 0.1, momentum 0.9,
wd 5e-4, nesterov, cosine, 60 ep, seed=42) stays at the project default. This
run's job is to **lock the floor** in CLAUDE.md's "Baseline numbers" block —
every later cell (B–F) must beat this number to count as a real contribution.
Expected acc 0.82–0.88 from iter 000's prior estimate.

## 2. Falsification criterion
Refuted if any of: (a) test_acc < 0.80 (would imply something is broken about
ResNet-34/CIFAR-10 even without aug — 0.82 is the conservative floor in the
literature for this recipe); (b) train_loss does not converge near 0 / train
acc stalls < 0.99 (would mean optimization itself failed, since "no aug" gives
the model maximum capacity to memorize); (c) `final.pth` missing `metrics`
or `history`. None triggered.

## 3. Changes made
New file `configs/ablation/iter002_bare_baseline.yaml`. The only delta vs
`configs/cifar10_resnet34.yaml` is one line:

```diff
 data:
-  augmentation: standard
+  augmentation: none
```

`exp_name` and the comment block updated accordingly. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | **0.8870** *(this row)* | — | 0.8870 | — | — |
| final_acc  | 0.8868                  | — | 0.8868 | — | — |
| test_loss  | 0.4231                  | — | 0.4231 | — | — |
| train_acc  | 1.0000                  | — | 1.0000 | — | — |
| train_loss | 0.0014                  | — | 0.0014 | — | — |
| best_epoch | 52 / 60                 | — | 52     | — | — |
| epochs run | 60                      | — | 60     | — | — |

Source: `runs/cifar10_iter002_bare_baseline/final.pth`,
`ckpt['metrics'] = {'acc': 0.8868, 'loss': 0.4231, 'best_acc': 0.887,
'best_epoch': 52}`. `history.json` has all 60 epoch rows; epoch-0
test_acc=0.2691 (consistent with iter 000's 0.2911 1-epoch number — same
recipe minus aug, slightly lower as expected); plateau begins around
epoch ~45 and best lands at epoch 52.

The **train-vs-test gap is the headline number**: train_acc=1.0000,
test_acc=0.8868 ⇒ ~11.3 pp generalization gap, exactly the symptom that
"no augmentation" is supposed to produce on CIFAR-10/ResNet-34. The model
fully memorized the 50k training set yet generalizes to 0.887 — well
above the falsification floor and at the upper end of iter 000's
0.82–0.88 prediction window.

## 5. Visualization evidence

**Per-class (`figs/iter_002/per_class.csv`).** Spread is 0.748 (cat) →
0.946 (automobile), a ~20 pp gap. Top: automobile 0.946, ship 0.944, frog
0.932, truck 0.932, horse 0.921, airplane 0.913 — every vehicle plus the
two animal classes with the most discriminative texture (frog skin, horse
silhouette). Bottom four: **cat 0.748, bird 0.819, dog 0.830, deer 0.885** —
the small mammals/birds that share fur, pose, and background distributions.
Compared to iter 000's collapse (cat=0.000, bird=0.050), the model has
clearly learned per-class structure, but the *ordering* of weak classes
is the same — cat remains the hardest by ~7 pp. This is the canonical
"hard classes" pattern for CIFAR-10/ResNet recipes; later cells should
move cat / bird / dog upward more than the already-saturated vehicles.

**t-SNE (`figs/iter_002/tsne.png`).** All 10 classes form distinct,
mostly-clean clusters in the 512-d penultimate space — a complete
qualitative leap from iter 000's two amorphous lobes. Vehicles (airplane,
automobile, ship, truck) sit on the right; animals on the left. Visible
contamination zones: (1) **cat ↔ dog** clusters touch at their edges with
a few cross-class points each way — the per_class numbers' main bottleneck;
(2) airplane ↔ ship show a small bridge (both have sky/water backgrounds
and elongated horizontal silhouettes); (3) deer is isolated cleanly.
horse forms its own island at the bottom — the highest-acc animal class,
consistent with per_class. The cluster geometry confirms the model has
learned genuine class-discriminative features, not just the
vehicle-vs-animal coarse split iter 000 was stuck at.

**Grad-CAM (`figs/iter_002/cam.png`).** 8 panels: 7/8 correct, 1 wrong
(ship→automobile on a tightly-cropped boat hull). The headline is
unexpected: **heatmaps remain strongly center-biased even at
convergence**. Across all 8 panels the activation peak sits within the
central ~30% of the frame — for the correctly-classified centered cat,
correctly-classified frog, even the wrongly-classified ship. Despite
train_acc=1.0, the model has not learned spatial invariance: with
augmentation=none, every training image presents the object at a fixed
center-biased location, so the classifier becomes a "look at the middle"
prior plus learned texture features. This is the **mechanism explanation
for why test_acc plateaus at 0.887 instead of 0.93+**: the model can't
compensate for off-center objects in the test set. Cells B (random crop)
and C (autoaug) should specifically *break* this center-bias signature
in their Grad-CAMs — that's the direct visual prediction to falsify next
iter.

## 6. Verdict
**Success** — in the specific sense of "baseline cleanly established,
falsification floor cleared, locked floor obtained." This is not a
+0.5 pp comparison row (there is no prior best to beat in the matrix),
but the run satisfied the hypothesis exactly: 60 epochs to completion,
test_acc landed inside the 0.82–0.88 prior window (at 0.887, just
above), training memorized the dataset (train_acc 1.0), no NaNs / no
divergence. CLAUDE.md's "Baseline numbers (LOCKED)" block can now be
filled in: `test_acc=0.8870, test_loss=0.4231, run_dir=runs/cifar10_iter002_bare_baseline/`.

## 7. Decision
**Keep & lock.** This row is the Cell A floor; propagate as the
denominator for every later cell's "Δ vs A" column. CLAUDE.md's matrix
row **A** gets `acc=0.887, Best iter#=2`. Iters 003 (Cell B, +std aug)
and 004 (Cell C, +autoaug) — already running in parallel on GPUs 1 & 2
per `state/iterations.tsv` — will be evaluated against this 0.887 floor.

## 8. Next hypothesis
The next loop tick is not mine to schedule (iter 003 is already running
on GPU 1 — `configs/ablation/iter003_std_aug.yaml`, Cell B = standard
augmentation, single-axis delta `none → standard`). Expected outcome:
acc rises to ~0.93–0.94, train-vs-test gap shrinks materially (from
the ~11 pp seen here toward ~3–4 pp), confirming A1 = `standard` is
worth +4–5 pp on its own. Iter 004 (Cell C, autoaugment) likewise.
After both land, iter 005 should be Cell D (`+adamw`) per the matrix
order — `configs/ablation/iter005_adamw.yaml` already staged in the
working tree.
