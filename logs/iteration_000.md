# Iteration 000 — cifar10_resnet34 smoketest
Date: 2026-04-27 09:07 | GPU: 2 | Duration: ~1 min (training 17.9 s)

## 1. Hypothesis
Sanity-check that the wired ResNet-34 + SGD(lr=0.1, mom=0.9, wd=5e-4, Nesterov) +
cosine + standard augmentation pipeline runs end-to-end on CIFAR-10 and produces
a non-trivial first-epoch test accuracy. This is a framework smoketest, not a
real ablation cell — `EPOCHS_OVERRIDE=1` shortened the 60-epoch config to one
epoch on iter 0.

## 2. Falsification criterion
Refuted if (a) training crashed / NaNed, (b) saved checkpoint or `history.json`
were missing/corrupt, or (c) test_acc after 1 epoch was ≤ 0.15 (≈ chance + a
tiny bump), which would imply a broken loss / data wiring.

## 3. Changes made
None vs. checked-in `configs/cifar10_resnet34.yaml` (augmentation=standard,
optimizer=sgd, scheduler=cosine, lr=0.1, wd=5e-4, epochs=60). Runtime
`EPOCHS_OVERRIDE=1` from the framework's first-launch smoketest path forced a
1-epoch run. No code or YAML edits.

## 4. Results
| Metric    | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A |
|-----------|-------------------|-------------|----------|-----------|--------|
| test_acc  | TBD               | TBD         | 0.3045   | n/a       | n/a    |
| test_loss | TBD               | TBD         | 1.8709   | n/a       | n/a    |
| train_acc | —                 | —           | 0.1884   | —         | —      |
| epochs    | 60 (planned)      | —           | 1        | —         | —      |

Run dir: `runs/cifar10_baseline/`. Best epoch: 0 (only one).
This iter is **not** a Cell A bare-baseline (Cell A requires augmentation=none);
config is closer to Cell B but truncated to 1 epoch, so it cannot fill any cell
of the §4 matrix.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_000/per_class.csv`): huge spread across classes
  after one epoch — `cat=0.0%`, `bird=0.6%`, `deer=13.9%` have nearly collapsed
  while `dog=55.5%`, `automobile=52.4%`, `frog=47.6%`, `airplane=40.2%` are
  already well above chance. Pattern is consistent with the model latching onto
  coarse colour/texture cues (warm fur, green ground, blue sky / metal body)
  rather than shape — small thin-bodied animals that share backgrounds with
  others (cat/bird/deer) get swallowed by the dominant classes. Sanity-OK for
  a 1-epoch run, but a useful baseline of which classes will be hardest later.
- **t-SNE** (`figs/iter_000/tsne.png`): only a coarse vehicle-vs-animal split
  is visible (vehicles + sky concentrate in the upper-right lobe, animals in
  the lower-left lobe). No class forms its own cluster; everything bleeds. Exactly
  what 1 epoch should look like — penultimate features are barely organized.
- **Grad-CAM grid** (`figs/iter_000/cam.png`): of 8 random test images, 5 are
  correct (ship, frog, automobile, frog, ship) and 3 wrong
  (airplane→ship, cat→bird, frog→deer). On the correct cases the heatmap is
  broadly object-centred but with a wide footprint that bleeds into background;
  on the airplane→ship miss the model attends to the dark band along the bottom
  rather than the plane silhouette — a clear texture/position shortcut. The
  attention is not yet sharp, again expected after one epoch.

## 6. Verdict
**Noise** — smoketest of the framework, not a meaningful ablation result.
Mechanism (training loop, data, optimizer, checkpointing) all fired cleanly;
test_acc=0.3045 after 1 epoch is in the expected range for SGD lr=0.1 + std-aug
+ cosine on CIFAR-10 ResNet-34 (typically 0.25–0.40 after epoch 0). No bug, but
no comparable cell yet.

## 7. Decision
Discard for ablation accounting. Iter 1 should launch the **real Cell A bare
baseline** — `configs/ablation/no_aug.yaml`, full 60 epochs, no `EPOCHS_OVERRIDE`
override — to establish the floor that the rest of the matrix is measured against.

## 8. Next hypothesis
Run Cell A (bare): `configs/ablation/no_aug.yaml` (augmentation=none, sgd,
cosine, lr=0.1, wd=5e-4, epochs=60, seed=42) for the full schedule. Falsifier:
test_acc < 0.80 would imply something seriously wrong with the unaugmented
pipeline; expected ≈ 0.83–0.86 based on common ResNet-34/CIFAR-10 numbers.
