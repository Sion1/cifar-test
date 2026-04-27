# Iteration 000 — cifar10_baseline (1-epoch fast smoketest)
Date: 2026-04-27 06:14 | GPU: 0 | Duration: ~1.4 min (29.5 s training)

## 1. Hypothesis
Iter 0 is the framework's pipeline-confirmation launch — `EPOCHS_OVERRIDE=1`
applied to the default `configs/cifar10_resnet34.yaml` (augmentation=standard,
sgd lr=0.1 momentum=0.9 wd=5e-4 nesterov, cosine, target 60 ep). The implicit
hypothesis is just "the train→checkpoint→reap pipeline works end-to-end on
this host and a single epoch produces a saved `final.pth` whose loss is
materially below random-guess level." It is **not** a cell-A baseline — A
requires augmentation=none and 60 epochs.

## 2. Falsification criterion
Pipeline broken if any of: (a) `final.pth` absent or missing `metrics`,
(b) test_acc ≤ random (0.10) — implying the model didn't learn,
(c) train_loss does not drop below the 2.30 cross-entropy chance level.
None of those triggered.

## 3. Changes made
None to code or YAML. Launched with `EPOCHS_OVERRIDE=1` against the unmodified
`configs/cifar10_resnet34.yaml`. The override is the only delta vs a real
baseline run; it is documented in `scripts/first_launch_setup.sh:401` as the
canonical smoketest invocation.

## 4. Results
| Metric    | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| test_acc  | — (not run yet)   | —           | 0.2911   | n/a       | n/a    |
| test_loss | —                 | —           | 1.8552   | n/a       | n/a    |
| train_acc | —                 | —           | 0.1779   | n/a       | n/a    |
| train_loss| —                 | —           | 2.3718   | n/a       | n/a    |
| epochs run| 60                | —           | 1        | n/a       | n/a    |

Source: `runs/cifar10_baseline/final.pth` ckpt['metrics'] =
`{'acc': 0.2911, 'loss': 1.8552, 'best_acc': 0.2911, 'best_epoch': 0}`,
`history.json` confirms a single epoch was logged.

This is ~3× better than the 0.10 random-guess floor — confirming the model
*is* learning — but well below the 0.94 program-goal floor and obviously not
comparable to any matrix cell (cells A–F all assume the configured
`training.epochs` runs to completion).

## 5. Visualization evidence

**Per-class (`figs/iter_000/per_class.csv`).** Sharply bimodal, dominated by
class-prior collapse. Top: airplane 0.663, automobile 0.522, dog 0.470,
frog 0.419. Bottom: **cat 0.000** (zero correct out of 1000), bird 0.050,
ship 0.109, truck 0.158. The 0% cat row is the headline: at one epoch the
softmax has not yet allocated probability mass to the cat class — almost all
true cats are being shipped to dog/bird/deer (the visually adjacent classes).
The dog rate (0.47) being ~10× the cat rate (0.00) on visually similar
inputs is the smoking gun for "cat predictions are being absorbed by dog."
This pattern would be alarming after 60 epochs but is normal at epoch 0.

**t-SNE (`figs/iter_000/tsne.png`).** Two coarse super-blobs are visible —
a "vehicle" lobe on the right where automobile/truck/ship/airplane points
intermix, and an "animal" lobe on the left where bird/cat/deer/dog/horse/
frog overlap heavily. Inside each lobe, individual classes are not
separable; cat (red) is sprinkled across the entire animal lobe rather than
forming any cluster, consistent with the 0% per-class. Even the vehicle
lobe shows no per-class clustering — only the binary vehicle-vs-animal
distinction has been learned.

**Grad-CAM (`figs/iter_000/cam.png`).** 8-image grid: 3/8 correct (airplane,
automobile, two frogs), 5/8 wrong (ship→airplane, frog→deer, cat→dog,
ship→automobile). Heatmaps are **uniformly center-biased**: the activation
peak sits near the image center across nearly every panel regardless of
where the object actually is, including ones the model gets right. The
model has effectively learned "look at the middle" plus a coarse
foreground-vs-background bias rather than object-specific localization —
the textbook signature of insufficient training.

Combined verdict from viz: pipeline is fine, model is genuinely learning,
but the representation is at a "vehicle vs animal + center prior" stage,
nowhere near the per-class discrimination needed to score >0.90.

## 6. Verdict
**Bug** — *not* in the sense of broken code, but per program.md §Verdict
criteria: "sanity baselines broken — halt and debug". The matrix's required
baselines (cells A–F) all assume the YAML's configured 60 epochs run; this
row is a 1-epoch override and therefore is **not** a usable baseline. Marking
it Bug excludes it from the loop's "best so far" / `PREV_ITER` selection
(`loop.sh:615` filters `verdict != "Bug"`), which is the desired behavior:
future iters should compare to a real 60-epoch baseline, not to 0.29.

## 7. Decision
Discard for the matrix. Do NOT propagate this number anywhere. The next loop
tick should propose **cell A — bare baseline** (augmentation=none, sgd 0.1,
cosine, wd=5e-4, full 60 epochs, seed=42) as iter 1, since CLAUDE.md still
shows the "Baseline numbers (LOCKED)" placeholder unfilled and §Required
ablation strategy mandates A→F before any winner.

## 8. Next hypothesis
**Iter 001 — Cell A bare baseline.** Clone `configs/cifar10_resnet34.yaml`
into `configs/ablation/iter001_bare_baseline.yaml`, change ONE axis:
`data.augmentation: standard → none`. Run for the configured 60 epochs (no
EPOCHS_OVERRIDE). Expected outcome: test_acc lands somewhere in the
0.82–0.88 range (typical ResNet-34/CIFAR-10 without augmentation), which
becomes the locked floor in CLAUDE.md's "Baseline numbers" block. Cell B
(+std aug) then runs as iter 2 to quantify A1's contribution alone.
