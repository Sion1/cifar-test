# Iteration 003 — iter003_bare (Cell A: bare baseline, no augmentation)
Date: 2026-04-27 09:27–09:47 | GPU: 1 | Duration: ~20 min wall (≈17.9 min net train, 60 ep × 17.9 s)

## 1. Hypothesis
Training ResNet-34 on CIFAR-10 with **no augmentation** (Cell A of the
ablation matrix) — single-axis delta `data.augmentation: standard → none`
vs. `configs/cifar10_resnet34.yaml` — establishes the program's **floor**.
Everything else fixed (sgd lr=0.1 mom=0.9 wd=5e-4 nesterov, cosine, 60 ep,
seed=42). Expectation: the model fully memorizes the training set and
generalizes worse than any augmented cell, with a large train–test gap.

## 2. Falsification criterion
Hypothesis is refuted if (a) test_acc beats Cell C (iter002, 0.9519) — that
would mean augmentation is irrelevant on this dataset/architecture, or (b)
the train–test gap is small (< ~3 pp), implying augmentation isn't the
binding regularizer. Either outcome would invalidate the cell-A→C ordering
the matrix assumes. A run failing to train (NaN, < 0.5 acc) would be a Bug.

## 3. Changes made
Cloned `configs/cifar10_resnet34.yaml` → `configs/ablation/iter003_bare.yaml`,
single-axis edit:

```diff
- exp_name: cifar10_baseline
+ exp_name: cifar10_iter003_bare
- augmentation: standard
+ augmentation: none
```

No code changes. Launched via `bash run_experiment.sh
configs/ablation/iter003_bare.yaml 3` (GPU 1 per loop scheduler).

## 4. Results
| Metric           | Cell A (this run) | Best so far (iter002 C) | Δ vs best (C) |
|------------------|------------------|-------------------------|---------------|
| test_acc (final) | **0.8812**       | 0.9519                  | −0.0707       |
| test_acc (best)  | **0.8828** @ ep55 | 0.9519 @ ep59          | −0.0691       |
| test_loss (final)| 0.4469           | 0.1630                  | +0.2839       |
| train_acc (final)| 1.0000           | 0.9642                  | +0.0358       |
| train–test gap   | **0.1188**       | 0.0123                  | +0.1065       |
| best_epoch       | 55               | 59                      | —             |
| epochs           | 60               | 60                      | —             |

Run dir: `runs/cifar10_iter003_bare/`. Test acc stalls in the 0.87–0.88 band
from epoch 50 onward (50→0.8811, 54→0.8819, 55→0.8828, 58→0.8805, 59→0.8812)
— cosine schedule has effectively converged; no further headroom from more
epochs at this regularization. Train_acc hits 1.0 by epoch ~50 with
train_loss ≈ 1.4e-3, while test_loss flattens at ~0.45 — the textbook
overfitting curve when no augmentation is in place. The 11.9 pp generalization
gap dwarfs Cell C's 1.2 pp, exactly the regularization signature the matrix
expects.

This LOCKS the §Required-ablation-strategy floor:
**Cell A test_acc = 0.8812 (final) / 0.8828 (best).** All later cells must
beat 0.8812. Cell C already does so by **+6.91 pp** (final) / **+6.91 pp**
(best), confirming AutoAugment is delivering substantial isolated value.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_003/per_class.csv`): the spread widens
  dramatically vs Cell C. Range = 0.770 (`cat`) → 0.959 (`automobile`) =
  **0.189 spread** (Cell C: 0.105). Class-by-class Δ vs iter002:
  airplane −0.047, automobile −0.025, bird **−0.131**, cat **−0.109**,
  deer −0.076, dog **−0.111**, frog −0.053, horse −0.056, ship −0.038,
  truck −0.045. The pattern is unambiguous: vehicles lose 2.5–5.3 pp,
  but the visually-ambiguous animal classes (`bird`, `cat`, `dog`) lose
  10.9–13.1 pp — exactly the classes where AutoAugment's color/shape
  jitter manufactures the texture invariance no-aug training can't learn
  on its own. `cat=0.770` and `dog=0.811` are the floor; the canonical
  CIFAR-10 hard pair is wide-open here.
- **t-SNE** (`figs/iter_003/tsne.png`): visibly looser than iter002 —
  only ~6 well-separated lobes (horse bottom-left, frog bottom-center,
  truck right, automobile lower-right, deer upper-left, ship upper-right);
  the rest show structural confusion. **`cat`+`dog` form one fused
  mammal blob in the middle-bottom** (matching the per-class table) with
  `bird` smearing into it. **`airplane` bleeds into `ship`** along the
  top-right cluster (sky-vs-water shared backgrounds). Multiple "bridge"
  tendrils connect classes — automobile↔truck has a continuous arc,
  cat↔dog↔deer have a scatter of mixed points — none of which were
  present in iter002's clean 8-lobe layout. This is the geometric
  signature of a model that has memorized rather than learned features
  invariant to small input perturbations.
- **Grad-CAM grid** (`figs/iter_003/cam.png`): 7 of 8 random samples
  correctly classified; the lone miss is a `frog→deer` confusion (the
  last panel). Heatmaps are object-centred *but visibly blobbier and
  broader* than iter002's tight, contour-following maps — the bare model
  fixates on a coarse central hot region (whole-object glow) rather than
  the discriminative parts iter002 found (fuselage, frog body, hull
  edges). This looser localization is consistent with the train_acc=1.0
  / test_acc=0.881 overfit profile: features are good enough to score
  the training images but lack the spatial sharpness AutoAugment imposes.
  No background-shortcut pathology — the model is using object pixels,
  just with less discrimination.

## 6. Verdict
**Success** (relative to its purpose: establishing the floor). The run
behaves exactly as the matrix predicts — fully memorized train set, ~12 pp
generalization gap, and a test_acc cleanly below every augmented cell that
will follow. Hypothesis confirmed; not a research win in absolute terms
(0.8812 < program target 0.94), but a methodologically necessary cell.
Not a "Success" in the verdict-criteria table sense (no acc rise vs best);
the framework labels Cell A "baseline=floor" — recording verdict as
**baseline** in `state/iterations.tsv` ($10) since it's neither rising nor
regressing relative to its role.

## 7. Decision
Keep. This is the canonical Cell A. Do **not** propagate any of its choices
downstream (the whole point of A is to be the thing other cells beat).
Lock `0.8828 (best) / 0.8812 (final)` as the floor in CLAUDE.md's matrix.
The Δ(C − A) = +0.0691 figure is the first clean *isolated* augmentation
delta we have; record it. Cell B (iter004_std) and Cell D (iter005_adamw)
are still running — once Cell B lands, Δ(C − B) gives the AutoAugment-vs-
standard delta and Δ(B − A) gives the standard-aug delta.

## 8. Next hypothesis
Already queued: iter004_std (Cell B, augmentation=standard) and iter005_adamw
(Cell D) are running on GPUs 0/3 respectively. The propose-step in the next
loop tick should pick up Cell E (multistep) or Cell F (long-train, 100 ep)
once Cell B and D finish — Cell E in particular will let us test whether
cosine is actually the right schedule, since A1 alone can't be the full
story if standard aug doesn't already saturate ≥ 0.94.
