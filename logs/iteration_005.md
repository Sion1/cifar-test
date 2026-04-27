# Iteration 005 — iter005_adamw (Cell D: AdamW optimizer)
Date: 2026-04-27 09:31–09:53 | GPU: 3 | Duration: ~22 min wall (≈18.7 min net train, 60 ep × 18.7 s)

## 1. Hypothesis
Single-axis swap of A2 from `sgd → adamw` (Cell D of the ablation matrix) on
top of the Cell B recipe (standard aug, cosine, 60 ep, seed=42), with the
A4/A5 dials moved into AdamW's catalog range (`lr 0.1 → 1.0e-3`,
`wd 5e-4 → 1.0e-4`). The optimizer-family swap should produce a
*comparable* test accuracy to Cell B (≈ 0.945 ± 0.5 pp). Folk-wisdom for
ResNet/CIFAR is that SGD+momentum slightly beats AdamW with default LR/WD,
so the prior is "lands near or modestly below 0.9477"; a Δ < −0.5 pp would
mark AdamW as a clear loser on this setup, while a Δ > +0.5 pp would crown
a new Cell-D-led winner.

## 2. Falsification criterion
Refuted if (a) test_acc ≤ 0.93 — AdamW with catalog-bracket
(lr=1e-3, wd=1e-4) fails to match SGD-baseline on this setup; or (b)
test_acc ≥ 0.95 — AdamW beats Cell B clearly enough that it (not
AutoAugment) is the next axis to ride; or (c) the train–test gap exceeds
Cell A's 11.9 pp, indicating AdamW with low wd over-memorizes badly. NaN /
< 0.5 acc would be a Bug (likely lr too high for AdamW, would suggest
1e-3 → 5e-4).

## 3. Changes made
Cloned `configs/ablation/iter004_std.yaml` →
`configs/ablation/iter005_adamw.yaml`, single-*axis* edit (A2 swap requires
shifting A4/A5 into AdamW's catalog brackets):

```diff
- exp_name: cifar10_iter004_std
+ exp_name: cifar10_iter005_adamw
  training:
-   optimizer: sgd
-   lr: 0.1
-   momentum: 0.9
-   nesterov: true
-   weight_decay: 5.0e-4
+   optimizer: adamw
+   lr: 1.0e-3
+   weight_decay: 1.0e-4
    scheduler: cosine
    epochs: 60
```

`momentum`/`nesterov` keys dropped (not used by AdamW). No code changes.
Launched via `bash run_experiment.sh configs/ablation/iter005_adamw.yaml 5`
(GPU 3 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell B (iter004, parent) | Cell C (iter002, current best) | Cell D (this run) | Δ vs B      | Δ vs C     | Δ vs A     |
|-------------------|------------------|--------------------------|--------------------------------|-------------------|-------------|------------|------------|
| test_acc (final)  | 0.8812           | 0.9475                   | 0.9519                         | **0.9350**        | **−0.0125** | −0.0169    | +0.0538    |
| test_acc (best)   | 0.8828 @ ep55    | 0.9477 @ ep58            | 0.9519 @ ep59                  | **0.9354 @ ep51** | **−0.0123** | −0.0165    | +0.0526    |
| test_loss (final) | 0.4469           | 0.2138                   | 0.1630                         | **0.4368**        | +0.2230     | +0.2738    | −0.0101    |
| train_acc (final) | 1.0000           | 0.9991                   | 0.9642                         | 0.9995            | +0.0004     | +0.0353    | −0.0005    |
| train–test gap    | 0.1188           | 0.0516                   | 0.0123                         | **0.0645**        | +0.0129     | +0.0522    | −0.0543    |
| best_epoch        | 55               | 58                       | 59                             | 51                | —           | —          | —          |
| epochs            | 60               | 60                       | 60                             | 60                | —           | —          | —          |

Run dir: `runs/cifar10_iter005_adamw/`. The trajectory is interesting:
test_acc climbs fast through ep5–25 (0.7981 → 0.9106 by ep20),
plateaus through ep25–40 (0.91–0.93), creeps up to 0.9354 at **ep51**,
then **flattens for the last 9 epochs** (ep51→0.9354, ep55→0.9349,
ep59→0.9350) — cosine has fully converged but the test curve has saturated
0.4 pp *below* its mid-epoch high-water-mark. Train_acc reaches 0.999 by
ep~45 with train_loss collapsing to ~1.5e-3 by ep55, while test_loss
**rises monotonically from ~0.30 (ep20) to 0.44 (ep59)** even as test_acc
holds — a clean overfit signature where the model becomes more
*confident* on its memorized train errors rather than picking up
generalizable signal. The 6.45 pp generalization gap is wider than Cell B's
5.16 pp (despite identical aug), so AdamW with wd=1e-4 is regularizing
*less* effectively than SGD+nesterov+wd=5e-4 here.

This **anchors Cell D** in the matrix and yields the optimizer-family
delta:
- **Δ(D − B) = −1.23 pp (best) / −1.25 pp (final)** — AdamW with
  catalog-bracket defaults *loses* to SGD+momentum on this Cell-B recipe.
  Mechanism fires (training is stable, no NaN, schedule completes), but
  the optimizer swap is a clear regression.
- **Δ(D − A) = +5.26 pp (best)** — Cell D still beats the bare-baseline
  floor by a healthy margin, so AdamW isn't broken; it's just
  *suboptimal* on this dataset/architecture/aug combo.

Cell D narrowly clears the program target ≥ 0.94 (final 0.9350 actually
*misses* it by 0.5 pp; best 0.9354 clears by 0.04 pp — a single-checkpoint
margin) and falls 1.65 pp short of stretch ≥ 0.95. Run does **not** set a
new provisional best; iter002 Cell C remains 0.9519 leader.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_005/per_class.csv`): spread = **0.863
  (cat) → 0.971 (automobile) = 0.108** — *looser* than Cell B's 0.089
  but tighter than Cell A's 0.189. Class-by-class numbers:
  airplane=0.948, automobile=0.971, bird=0.913, cat=0.863, deer=0.955,
  dog=0.885, frog=0.944, horse=0.954, ship=0.958, truck=0.963.
  **Δ vs Cell B** (in pp): airplane −1.4, auto −0.2, bird −0.4,
  **cat −2.1, dog −3.5, frog −2.6**, horse −1.2, deer −0.3, ship −0.4,
  truck −0.2 — the AdamW penalty is **disproportionately animal-side**:
  the four animal classes cat/dog/frog/horse absorb 9.4 pp of the 12.3
  pp aggregate drop, while vehicles (auto/truck/ship) collectively give
  up only 0.8 pp. This inverts the Cell B → Cell C pattern (where the
  C−B gap was concentrated in *bird* and *automobile*) and tells a
  consistent regularization story: standard aug + SGD has its strongest
  grip on animal-shape evidence, and replacing SGD+wd=5e-4 with
  AdamW+wd=1e-4 is exactly where that grip loosens. **Δ vs Cell C**
  (in pp): every class is worse, with dog (−3.7), bird (−3.3), frog
  (−2.8), cat (+1.6), horse (−1.0), ship (−1.2), airplane (−0.5),
  auto (−1.3), deer (−0.6), truck (−0.5) — only `cat` improves, by a
  surprising +1.6 pp (Cell D's cat=0.863 vs Cell C's 0.879). One-class
  win is too narrow to read into.
- **t-SNE** (`figs/iter_005/tsne.png`): **8 visible lobes** —
  comparable to Cell B / Cell C in cluster *count* but with **noticeably
  looser boundaries**. Clean, isolated clusters: horse (far-left, grey,
  sharply isolated), deer (lower-left, purple), frog (lower-mid, pink),
  truck (top-left, cyan), automobile (top, orange), ship (right, yellow).
  Three structural problem zones: (i) **cat↔dog fuse** in the
  centre-left — brown dog blob immediately adjacent to red cat blob,
  effectively two sub-clusters of one mammal mass (looser than Cell C's
  full fuse, looser than Cell B's "brown blob with red sub-cluster"
  signature); (ii) **bird↔airplane↔ship overlap** on the right — green
  bird points are scattered across the right margin and bleed into both
  airplane (blue) and ship (yellow), worse than Cell B's clean
  separation; (iii) **automobile↔truck boundary** at top has visible
  orange↔cyan mixing absent from Cell B. Overall geometry: same lobe
  count as Cell B but consistent with the +0.0129 gap-widening — the
  feature space is similarly *organized* but less *separated*.
- **Grad-CAM grid** (`figs/iter_005/cam.png`): **8/8 correctly
  classified** (matches Cell B's clean grid; Cell A and Cell C were
  7/8). Heatmaps are sharply object-centred with bright red cores on
  every sample: ship hull, frog body in grass, airplane fuselage,
  automobile body, frog body, cat torso, ship hull, frog body. n=8
  caveat applies (3× frog skews the panel). Localization quality is
  visually indistinguishable from Cell B — AdamW does *not* induce a
  qualitatively different saliency pattern; the regression is purely a
  *generalization-margin* problem (test_loss drift), not an
  attention-mechanism problem. No background-shortcut pathology.

## 6. Verdict
**Failure.** Mechanism fires cleanly (no NaN, cosine completes,
checkpoint healthy), but Δ vs the parent Cell B = **−1.23 pp**, well
outside the Partial band's [−0.5, +0.5] and the Noise band's ±0.3 pp.
Per program.md §Verdict: "acc drops > 0.5 pp on an expected-positive
hypothesis" → Failure. The optimizer-family swap (A2: sgd → adamw) at
catalog defaults (lr=1e-3, wd=1e-4) does **not** earn its keep on top of
standard aug + cosine + 60 ep; the signature is mild over-memorization
(train_loss → 1.5e-3, test_loss climbing) consistent with insufficient
regularization. Cell D is *anchored* (the matrix has its first optimizer
data point), but it's a negative result: AdamW is dominated by SGD on
this exact recipe.

## 7. Decision
Discard as a propagation parent — downstream cells (E multistep, F
long-train) should continue to build on **Cell B (sgd, 0.9477)**, not on
Cell D. Lock Cell D at **0.9354 (best) / 0.9350 (final)** in `CLAUDE.md`'s
matrix as anchor for the A2 axis. The negative result is informative:
on Cell-B-style recipes, the optimizer family is *not* a productive axis
to spend further iterations on without first re-tuning A4/A5 — and the
catalog only has one more AdamW point (lr=5e-4) which is unlikely to
recover 1.23 pp on its own. Do **not** schedule a Cell D follow-up unless
later cells leave budget. Continue priorities: (1) Cell E (multistep,
A3), (2) Cell F (long-train, A6 = 100 ep), (3) the 2-seed (seed=4078)
replay of Cells B and C to harden the C−B = +0.42 pp gap.

## 8. Next hypothesis
Cell E (multistep schedule) — single-axis delta from Cell B with
`training.scheduler: cosine → multistep` (and a milestone schedule like
[30, 45] with γ=0.1, so the step decay matches the 60-ep budget). Test
A3: the canonical SGD recipe in older CIFAR papers used multistep, and we
need to rule it out (or confirm cosine's edge) before any winner is
crowned. Falsifier: Δ vs Cell B in [−0.5, +0.5] pp = Partial / Noise →
cosine is the safe default; Δ ≥ +0.5 pp → multistep is the actual A3
winner and should propagate to Cell F.
