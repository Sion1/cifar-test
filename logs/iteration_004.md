# Iteration 004 — cifar10_iter004_autoaug (Cell C — +autoaug)
Date: 2026-04-27 07:15 | GPU: 2 | Duration: ~34 min (60 epochs × ~31 s/epoch)

## 1. Hypothesis
Iter 004 targets **Cell C — +autoaug** (program.md §Required ablation
strategy): single-axis delta vs Cell B (iter 003) is
`data.augmentation: standard → autoaugment`; all other knobs (sgd 0.1,
momentum 0.9, wd 5e-4, nesterov, cosine, 60 ep, seed=42) match iter 003.
This row quantifies the **upper end of the A1 axis** by adding learned
photometric/geometric policies on top of pad-crop+HFlip. Expected vs Cell B:
+0.5 to +1.5 pp acc, train-vs-test gap shrinks further (5.2 pp → ~3 pp),
test_loss drops below 0.18, and per-class gains continue to be concentrated
on the small-mammal classes (cat / dog / bird) that still had headroom after
Cell B.

## 2. Falsification criterion
Refuted if any of: (a) test_acc ≤ 0.9481 (i.e. ≤ Cell B — Failure for an
expected-positive A1-strengthening hypothesis); (b) train-vs-test gap stays
≥ 5 pp (would mean autoaug isn't acting as a stronger regularizer than
"standard"); (c) per-class regression on the saturated vehicle classes
exceeds −1 pp on more than one class (would mean autoaug's policies are
distorting vehicles more than they help small mammals); (d) `final.pth`
missing `metrics`/`history`. Outcome: (a) cleared but only narrowly — see §4
below for the Partial verdict; (b) decisively cleared (gap ≈ 1 pp); (d)
clean. (c) folded into §5 once per_class.csv lands.

## 3. Changes made
New file `configs/ablation/iter004_autoaug.yaml` (already present at iter
launch). The only delta vs `configs/ablation/iter003_std_aug.yaml` is one
line:

```diff
 data:
-  augmentation: standard
+  augmentation: autoaugment
```

`exp_name` and the comment block updated accordingly. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far (Cell B) | This run (Cell C) | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | **0.9528** | **+0.0047** | **+0.0658** |
| final_acc  | 0.8868 | 0.9478 | 0.9522     | +0.0044     | +0.0654     |
| test_loss  | 0.4231 | 0.2106 | 0.1535     | −0.0571     | −0.2696     |
| train_acc  | 1.0000 | 0.9993 | 0.9625     | −0.0368     | −0.0375     |
| train_loss | 0.0014 | 0.0037 | 0.1107     | +0.1070     | +0.1093     |
| best_epoch | 52 / 60 | 57 / 60 | 58 / 60   | —           | —           |
| epochs run | 60      | 60      | 60         | —           | —           |

Source: `runs/cifar10_iter004_autoaug/final.pth`,
`ckpt['metrics'] = {'acc': 0.9522, 'loss': 0.1535, 'best_acc': 0.9528,
'best_epoch': 58}`. `history.json` (60 rows) shows the slowest start of
the three (epoch 0 test_acc=0.1842 vs Cell A 0.2691 / Cell B 0.3052 — autoaug
hurts at init because the policy distortions are out-of-distribution for an
untrained network), then steady climb past Cell B around epoch 45
(test_acc=0.92), plateau begins ~epoch 53 at 0.95+ and best lands at epoch
58 (one epoch later than Cell B's 57).

The headline mechanism — **gap collapse** — fired more strongly than
Cell B: train_acc=0.9625 vs test_acc=0.9522 ⇒ ~1.0 pp gap, an order of
magnitude tighter than Cell A (11.3 pp) and ≈ 1/5 of Cell B (5.2 pp).
Test_loss further halved Cell B's value (0.2106 → 0.1535). However, the
headline acc number gained only +0.47 pp vs Cell B — **below the §Verdict
+0.5 pp threshold for "Success"**, putting this iter into **Partial**
territory: the mechanism (regularization) fired hard, but the acc gain
saturated. This is exactly the failure mode predicted by program.md's
diminishing-returns reasoning on the A1 axis.

## 5. Visualization evidence

**Per-class (`figs/iter_004/per_class.csv`).** Spread compressed
modestly, 0.885 (cat) → 0.983 (automobile) ≈ 9.8 pp, vs Cell B's 9.7
pp — essentially unchanged. The key surprise is that the gains are
**not** concentrated on the small-mammal classes the way Cell B's were.
Per-class Δ vs Cell B:
- frog **+1.5 pp** (0.956 → 0.971)
- bird **+1.1 pp** (0.925 → 0.936)
- deer **+1.0 pp** (0.957 → 0.967)
- automobile **+0.8 pp**, horse **+0.8 pp**, cat **+0.7 pp**, ship
  **+0.6 pp**, truck **+0.4 pp**
- airplane **−0.6 pp** (0.962 → 0.956)
- dog **−1.5 pp** (0.930 → 0.915) ⚠

Cat (the long-standing worst class) gained only +0.7 pp, and **dog
regressed by 1.5 pp** — both small-mammal cells where Cell B made the
big +10 pp jump. The headline +0.47 pp gain came mostly from frog/
bird/deer/automobile, not the cat↔dog confusion zone the iter-003 log
expected autoaug to attack. This is a genuine refutation of the
"autoaug helps small mammals most" sub-hypothesis.

**t-SNE (`figs/iter_004/tsne.png`).** All 10 classes are visibly
separated, with horse, automobile, truck, frog, deer cleanly isolated.
The **cat↔dog bridge from Cell B is still present and arguably worse**:
red (cat) and brown (dog) clusters share a wide contact zone with
clearly mixed points along the boundary, and a few greens (bird) and
pinks (frog) leak in at the edge. Meanwhile the airplane (blue) /
ship (yellow) / bird (green) corner shows a *new* small contamination
zone — bird sits between airplane and ship in the lower-right, with
a few crossed labels at each junction. So autoaug rearranged where
the residual confusion lives more than it eliminated it: Cell B's
clean airplane/ship separation has slightly degraded, while cat/dog
hasn't improved.

**Grad-CAM (`figs/iter_004/cam.png`).** 7/8 correct (vs 8/8 on Cell B).
The single failure is row-1 panel-1: a ship predicted as automobile —
the heatmap correctly localizes on the bridge/superstructure of the
ship, but apparently the model has learned that compact, mid-frame
metallic-rectangle shapes look more car-like than ship-like, which is
plausibly a side-effect of autoaug's color/contrast policies muddling
hull-color cues. The other 7 panels show object-following heatmaps
(frog → frog body, cat → cat face/torso, airplane → fuselage centered
on the actual plane, automobile → wheel-arch region). The Cell B
"center-bias broken" signature is preserved.

So autoaug bought a real but small acc gain through gap-collapse, did
not concentrate gains on cat/dog as expected, and slightly degraded
airplane (likely color-policy interference). Cell B remains the stronger
single-seed picture for the small-mammal axis.

## 6. Verdict
**Partial** — best_acc rose +0.47 pp vs current best Cell B (0.9481 →
0.9528), which is **below** the §Verdict +0.5 pp threshold for "Success"
but well above the ±0.3 pp "Noise" band. Mechanism evidence is strong
(gap 5.2 pp → 1.0 pp, test_loss 0.2106 → 0.1535, best_epoch shifted later)
and the prediction sign was correct, but the acc improvement saturated
short of the +0.5 pp bar. The hypothesis was *partially* refuted on
magnitude (predicted +0.5 to +1.5 pp; got +0.47 pp), confirmed on
mechanism, and confirmed on the gap-collapse direction.

## 7. Decision
**Keep but DO NOT rebrand as new best.** Cell B (0.9481) remains
"Best so far" by the §Verdict criteria — autoaug saved Cell C from
Failure but didn't earn the Success tag. Update CLAUDE.md matrix row **C**
with `acc=0.9528, Best iter#=4, Verdict=Partial`. The aggressive
gap-collapse (1.0 pp train-vs-test) means **autoaug under-fit slightly at
60 ep**: the model has more capacity to absorb more epochs of autoaug
than Cell B did of std-aug. This makes Cell F (long-train, 100 ep) the
single highest-leverage remaining cell — it stacks 100 ep on the
autoaug-paired regularizer that hasn't saturated. The pre-staged
`configs/ablation/iter005_adamw.yaml` (Cell D) is the next iter on the
ablation matrix path; Cell F can be promoted later once D and E are in.

## 8. Next hypothesis
Iter 005 (Cell D — +adamw): single-axis delta vs Cell B is
`training.optimizer: sgd → adamw` with `training.lr: 0.1 → 1e-3` (the
catalog-bounded AdamW LR). Predicted: **Failure or Partial.** AdamW is
known to underperform tuned-SGD+momentum on CIFAR-10 ResNets; expect
test_acc 0.93–0.945 (i.e. Δ vs Cell B between −1.5 and 0 pp) and a
train-vs-test gap similar to Cell B (~5 pp, since A1 stays "standard").
The point of running it isn't to win — it's to nail down the A2 axis with
single-seed evidence so the matrix is complete before any 2-seed
hardening pass. After Cell D, iter 006 should be Cell E (multistep)
unless a fast-result ordering argues otherwise.
