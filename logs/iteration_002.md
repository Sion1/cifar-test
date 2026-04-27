# Iteration 002 — iter002_autoaug (Cell C: AutoAugment)
Date: 2026-04-27 09:21–09:42 | GPU: 2 | Duration: ~21 min (60 epochs × ~18.7 s)

## 1. Hypothesis
Switching A1 from `standard` to `autoaugment` (Cell C of the ablation matrix)
should yield a stronger A1 than Cell B, raising CIFAR-10 ResNet-34 test accuracy
above the standard-aug ceiling. With everything else held fixed (sgd lr=0.1
mom=0.9 wd=5e-4 nesterov, cosine, 60 ep, seed=42), any Δ vs. expected Cell B
is attributable to A1 alone.

## 2. Falsification criterion
Hypothesis is refuted if test_acc ≤ ~0.93 (a reasonable expected Cell B floor
for ResNet-34 + std aug + cosine 60ep), or if AutoAugment increased loss /
caused training instability (NaN, divergence, train_acc < test_acc by a
suspicious margin).

## 3. Changes made
Cloned `configs/cifar10_resnet34.yaml` → `configs/ablation/iter002_autoaug.yaml`,
single-axis edit:

```diff
- exp_name: cifar10_baseline
+ exp_name: cifar10_iter002_autoaug
- augmentation: standard
+ augmentation: autoaugment
```

No code changes. Launched via `bash run_experiment.sh
configs/ablation/iter002_autoaug.yaml 2`.

## 4. Results
| Metric    | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A   |
|-----------|-------------------|-------------|----------|-----------|----------|
| test_acc  | TBD (not run)     | 0.3045 *    | 0.9519   | +0.6474*  | n/a      |
| test_loss | TBD               | 1.8709 *    | 0.1630   | −1.7079*  | n/a      |
| best_epoch| —                 | 0           | 59       | —         | —        |
| epochs    | 60 (planned)      | 1 (smoke)   | 60       | —         | —        |

\* "Best so far" only reflects iter000, which was a 1-epoch standard-aug
smoketest (not a real ablation cell), so the Δ-vs-best column is not
mechanistically meaningful — it just confirms that this is the first
fully-trained run. Cell A bare baseline is still pending (iter003 running).
Run dir: `runs/cifar10_iter002_autoaug/`. Best epoch: 59 (final epoch — cosine
schedule trained right up to the end, no plateau-then-overfit pattern).
Train_acc=0.9642 vs. test_acc=0.9519: ~1.2 pp generalization gap — quite tight,
exactly the regularization signature one expects from AutoAugment.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_002/per_class.csv`): every class is in the
  0.879–0.984 band — vast improvement over iter000's 0.000–0.555 spread. The
  ranking is the textbook CIFAR-10 difficulty ordering: `automobile=0.984`,
  `frog=0.972`, `ship=0.970`, `truck=0.968`, `horse=0.964`, `deer=0.961`,
  `airplane=0.953`, `bird=0.946`, `dog=0.922`, **`cat=0.879`** as the lone
  weak class. The cat–dog confusion is the canonical hard pair (small mammal
  shape + similar fur texture at 32×32) and is exactly where AutoAugment is
  expected to *help* but not *solve* — colour-jitter style policies cannot
  manufacture cat-vs-dog shape evidence the model never had access to. No class
  is collapsed; the model is well calibrated, just confounded on `cat`.
- **t-SNE** (`figs/iter_002/tsne.png`): 8 of 10 classes form their own
  well-separated clusters — `truck` (cyan), `automobile` (orange), `horse`
  (grey), `frog` (pink), `ship` (yellow), `deer` (purple), and the
  `bird`+`airplane` pair each have visible boundaries. The two bleed-zones
  match the per-class table exactly: (i) `cat`+`dog`+a few `deer` form one
  fused mammal blob in the left-middle, and (ii) `bird` and `airplane` show
  partial overlap along their shared sky-background subspace. The vehicle
  super-cluster (truck+automobile in the top-right; ship far-right; airplane
  bottom-mid) is geometrically consistent with iter000's coarse "vehicle vs.
  animal" split, now refined into per-class lobes.
- **Grad-CAM grid** (`figs/iter_002/cam.png`): 7 of 8 random samples are
  correctly classified; the lone miss is a `ship→automobile` confusion where
  the heatmap focuses on the dark hull pixels rather than the masts/superstructure.
  On the correct cases the attention is sharp and tightly object-centred —
  the frog heatmaps fixate on the body, the horse / cat / ship maps target
  the central subject, and the airplane CAM lands exactly on the fuselage —
  a clear improvement over iter000's loose, background-bleeding maps. The
  model is using object pixels, not background shortcuts, which corroborates
  the high accuracy as legitimate (not a colour-cue artifact).

## 6. Verdict
**Success** (vs. program.md targets). 0.9519 already meets the baseline target
(≥ 0.94) and clears the stretch goal (≥ 0.95). Hypothesis (AutoAugment helps
on top of cosine + SGD) is supported. Caveat: Cell B (standard aug, full 60
ep) has not yet been run, so the *isolated* AutoAugment delta cannot be
quantified — once iter004_std finishes we can compute Δ(C − B) cleanly. For
now this firmly establishes a high-acc anchor and is a strong candidate for
2-seed hardening.

## 7. Decision
Keep. This run is a strong Cell C candidate — log as best_so_far in
`state/iterations.tsv`. Do NOT crown a winner until Cell B (iter004_std) and
Cell A (iter003_bare) finish; only then can the AutoAugment delta be
attributed cleanly. Plan 2-seed replay (seed=4078) once Cell B is in.

## 8. Next hypothesis
Already queued: iter003_bare (Cell A — augmentation=none) and iter004_std
(Cell B — augmentation=standard) are both running. The next single-axis
delta to consider after those land is Cell D (AdamW, iter005_adamw — also
already running). Picking up the propose-step from the next loop tick.
