# Iteration 009 — cifar10_iter009_std_aug_seed4078 (Cell B hardening — seed=4078)
Date: 2026-04-27 14:03 | GPU: 2 | Duration: ~35 min (60 epochs × ~33 s/epoch)

## 1. Hypothesis
Iter 009 is the **Cell B 2-seed hardening pass** mandated by program.md
§Required ablation strategy ("after phase 1, pick top 2, run with
seed=4078, report 2-seed mean"). It pairs with iter 008 (Cell C
seed=4078, finished as Partial) so the top-2 cells (C @ 0.9528 and
B @ 0.9481) both have 2-seed evidence before any winner is crowned.
Single-axis delta vs iter 003 (Cell B, seed=42) is `seed: 42 → 4078`;
all other knobs identical (aug=standard, sgd 0.1, momentum 0.9, wd
5e-4, nesterov, cosine, epochs=60). Mechanism question: does Cell B's
+6.11 pp gain over Cell A replicate at a different seed, or did
seed=42 ride a favorable optimization trajectory?

## 2. Falsification criterion
Per iter-008 §8: predicted band **[0.9451, 0.9511]** (Cell B seed=42
± 0.3 pp). Strong falsifier on the **upside**: a result **> 0.9528**
(above Cell C seed=42's single-seed result) would invert the matrix
ordering — Cell B would beat Cell C s=42 on a single seed and the
"+std vs +autoaug" question would become unresolved at 2 seeds.
Strong falsifier on the **downside**: a result **< 0.9451** would
match the same negative seed-luck Cell C just exhibited, dropping
Cell B's 2-seed mean to ~0.9466 and restoring Cell C's lead to
~+0.45 pp. Mechanism check: train-vs-test gap should land near
Cell B seed=42's 5.20 pp — anywhere ≤ 2 pp would mean std-aug
suddenly regularizes much harder at the new seed (very surprising);
anywhere > 7 pp would mean std-aug under-regularizes here.

## 3. Changes made
New file `configs/ablation/iter009_std_aug_seed4078.yaml`. Diff vs
`configs/ablation/iter003_std_aug.yaml` (Cell B seed=42):

```diff
 exp_name: cifar10_iter009_std_aug_seed4078
-seed: 42
+seed: 4078
 data:
   augmentation: standard
   ...
```

`exp_name` updated. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Cell B s=42 (iter003) | Cell C s=42 (iter004) | Cell C s=4078 (iter008) | This run (Cell B s=4078) | Δ vs Cell B s=42 | Δ vs Cell A |
|---|---|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | 0.9528 | 0.9497 | **0.9478** | **−0.0003** | **+0.0608** |
| final_acc  | 0.8868 | 0.9478 | 0.9522 | 0.9497 | 0.9478     | 0.0000      | +0.0610     |
| test_loss  | 0.4231 | 0.2106 | 0.1535 | 0.1599 | 0.2085     | −0.0021     | −0.2146     |
| train_acc  | 1.0000 | 0.9993 | 0.9625 | 0.9618 | 0.9995     | +0.0002     | −0.0005     |
| best_epoch | 52/60  | 57/60  | 58/60  | 59/60  | 58/60      | —           | —           |
| epochs run | 60     | 60     | 60     | 60     | 60         | —           | —           |

Source: `runs/cifar10_iter009_std_aug_seed4078/final.pth`,
`ckpt['metrics'] = {'acc': 0.9478, 'loss': 0.2085, 'best_acc': 0.9478,
'best_epoch': 57}` (0-indexed → 58/60). Train-vs-test gap = 0.9995 −
0.9478 = **5.17 pp** (Cell B seed=42: 5.20 pp; Cell C s=4078: 1.21
pp). Gap is **identical** (within 0.03 pp) to Cell B seed=42 — std-aug
regularizes the same way at this seed.

Trajectory check (std-aug-with-cosine signature):
- ep 0:  test_acc 0.3372 (faster start than autoaug — aug is weaker)
- ep 10: ~0.78
- ep 20: ~0.86
- ep 30: 0.8829 (Cell B s=42 at ep30: ~0.91 — slightly slower start
  this seed; recovered by epoch ~45)
- ep 50: 0.9411
- ep 55: 0.9468
- ep 57 (best): **0.9478**
- ep 59: 0.9478

Last 10 epochs: 0.9411, 0.9453, 0.9458, 0.9452, 0.9471, 0.9468,
0.9476, 0.9478, 0.9469, 0.9478 — plateau at ~0.945–0.948, no
late-epoch slope, confirming std-aug at 60 ep is **not under-fit**
(consistent with iter-006's finding that "best_epoch near termination
on cosine is a schedule-shape artifact, not under-fit headroom").

**Headline:** 0.9478 lands **dead center** in the predicted band
[0.9451, 0.9511]. Strong falsifiers neither triggered (margin to
upside +0.0050, margin to downside +0.0027). Δ vs Cell B s=42 =
−0.0003 — well inside the §Verdict **Noise band** (|Δ| < 0.3 pp).
**2-seed Cell B mean = (0.9481 + 0.9478) / 2 = 0.94795.** Combined
with Cell C's 2-seed mean = 0.95125, **Cell C leads at the 2-seed
mean by +0.33 pp** — outside the §Verdict Noise band on Cell C's
side, inside the Success threshold (< 0.5 pp). The matrix winner is
**Cell C by 2-seed mean**, but the lead is in the Partial range, not
Success.

## 5. Visualization evidence

**Per-class (`figs/iter_009/per_class.csv`, with Cell B seed=42
re-measured from `runs/cifar10_iter003_std_aug/best.pth` for honest Δ).**
Per-class accuracy and Δ vs Cell B seed=42 (this iter − Cell B s=42, in pp):
- airplane:    0.962 → 0.965  (**+0.3**)
- automobile:  0.975 → 0.981  (**+0.6**)
- bird:        0.925 → **0.936** (**+1.1**) ⬆ — biggest single gain
- cat:         0.878 → 0.873  (−0.5)
- deer:        0.957 → 0.958  (+0.1)
- dog:         0.930 → **0.910** (**−2.0**) ⬇⬇ — biggest single loss
- frog:        0.956 → 0.965  (+0.9)
- horse:       0.966 → 0.961  (−0.5)
- ship:        0.967 → 0.965  (−0.2)
- truck:       0.964 → 0.963  (−0.1)

Mean Δ ≈ −0.03 pp ≈ headline −0.03 pp ✓. Despite the headline acc
being effectively identical (0.9477 vs 0.9480 from per-class
recompute), the **per-class redistribution is non-trivial: dog
−2.0 pp is the largest single-class shift seen in any 2-seed pair so
far** (Cell C's s=42→s=4078 max single-class shift was cat −1.1 pp).
Bird +1.1 pp partially compensates. Cat barely moved (−0.5 pp).
Top off-diagonals: cat→dog=63 (Cell B s=42: 71), dog→cat=55
(Cell B s=42: 44). Same direction as Cell B s=42 (cat→dog dominant)
but the **gap shrank from 27 to 8** — the dog regression is mostly
"new" dog→cat errors, not "lost" cat→dog correction. **Implication:
Cell B s=42's characteristic 71-vs-44 cat↔dog asymmetry is itself
seed-dependent in magnitude** (direction stable, magnitude swings
factor-of-3). The asymmetry is preserved (no flip to dog→cat
dominant — that pattern remains specific to schedule perturbations
E and F, not seed perturbations). airplane↔ship: 12 vs 18 here
(s=42: 14 vs 15) — direction flipped at this seed but magnitudes
within ±4 of s=42's, so the airplane↔ship boundary is mildly seed-
sensitive.

**t-SNE (`figs/iter_009/tsne.png`).** Ten mostly-separated clusters,
better-organized than expected. The **cat (red) and dog (brown)
clusters are clearly two distinct lobes** with only a thin band of
mixing — *visually cleaner than Cell C s=4078's merged cat↔dog blob*.
This contradicts the iter-008 reading that "any deviation off Cell
B muddies cat↔dog": Cell B s=4078 has a different per-class profile
(dog −2.0 pp) than Cell B s=42 but t-SNE feature-space structure is
*not* worse — the dog regression is concentrated in ambiguous-image
errors, not a wholesale collapse of the dog manifold. Bird (green)
sits between cat (red) and airplane (blue) with a few scatters
across each — bird gained +1.1 pp here despite being adjacent to
two confusion zones. **Vehicle pair (orange automobile, cyan truck)
is clean** and well-separated (consistent with the matrix-wide
finding that automobile↔truck is the most robust separation).
**Airplane↔ship contamination zone is mildly visible** as a few
yellow ship points on the right edge of the airplane cluster (and
vice versa), matching the 12/18 confusions. Frog (pink), horse
(gray), deer (purple) are all clean isolated clusters. **No
schedule-style structural anomalies** (no dog→cat flip, no broad
fragmentation as in Cell E's bird).

**Grad-CAM (`figs/iter_009/cam.png`).** **8/8 correct**, but the
**heatmap signature is centered round blobs across all 8 panels**,
NOT the location-following shape Cell B seed=42 had locked in. The
row-1 ship panel (ship occupies the lower 60%) shows a heatmap
centered mid-image, not tracking the hull down. The row-1 frog panel
(frog mid-lower) is a centered blob. The row-3 cat panel (cat
upper-left of frame, blue background lower-right) shows a heatmap
centered mid-image, NOT elongating toward the cat. The row-3 ship
panel shows mild horizontal elongation along the hull — the only
panel with measurable location-following. So **7/8 panels are
centered blobs, 1/8 has mild location tracking**. **This is the
same centered-blob signature observed in Cells C s=42, C s=4078, D,
E, F** — and crucially, it appears here on a recipe with
A1/A2/A3/A4/A5/A6 *identical* to Cell B seed=42. **The
location-following Grad-CAM signature is therefore unique to Cell
B's seed=42 optimization trajectory, not a property of the Cell B
recipe.** This is the **decisive test of the iter-005/006/007
"heatmap shape = recipe canary" hypothesis: it FAILS.** The
spatial-invariance canary is a seed=42-specific artifact;
heatmap shape is **NOT** a reliable mechanism diagnostic on this
matrix — even when the recipe is identical, the seed determines
whether the network organizes spatially-localized vs centered
feature channels. Going forward: drop heatmap shape as a
diagnostic; use train-vs-test gap, per-class profile, and t-SNE
structure as the mechanism canaries instead.

## 6. Verdict
**Noise** — per the §Verdict criteria, |Δ vs Cell B s=42| = 0.03 pp
is well inside the |Δacc| < 0.3 pp seed-variance band. This is the
**desired** Noise verdict for a 2-seed hardening pass: Cell B's
+6.11 pp gain over Cell A is **fully replicated** at seed=4078
(0.9478 vs 0.9481), and the train-vs-test gap (5.17 vs 5.20 pp) shows
the std-aug regularization mechanism fires identically. Both
falsifiers from iter-008 §8 cleared cleanly. **Cell B is the most
seed-stable cell measured so far** (2-seed peak-to-peak = 0.03 pp,
vs Cell C's 0.31 pp).

For the matrix-level question: combined with iter 008, the 2-seed
means are **Cell B = 0.94795** and **Cell C = 0.95125**, a +0.33 pp
Cell C lead. This is outside the noise band but well below the
Success threshold (< 0.5 pp). **Cell C wins phase 2 by Partial
margin**, not Success margin. The headline reading: autoaug *does*
beat std-aug on this recipe, but only by ~0.3 pp at the 2-seed mean
— roughly half the +0.47 pp single-seed lead suggested.

## 7. Decision
**Crown Cell C as phase-2 winner with a Partial-margin lead.** Update
CLAUDE.md matrix row **B** with `2-seed mean = 0.94795` and
"Current best" with **Cell C at 2-seed mean 0.95125**. Cell B is
locked as the seed-stable, mechanism-clean reference recipe; Cell C
is the headline winner by margin but with seed sensitivity (cat
component is unstable). **Phase 2 is now closed**; phase 3 (axis
sweeps off the winner) can begin. iter 010 (autoaug at lr=0.05, A4
sweep off Cell C) is already running per state/iterations.tsv —
that's the right axis to probe now that Cell C is crowned, but its
launch was technically premature relative to iter 009's analysis.
Do NOT propagate the +std vs +autoaug choice as settled — the
+0.33 pp 2-seed margin is small enough that *any* phase-3 finding
on the Cell C lr/wd axes that doesn't transfer back to Cell B should
be flagged as Cell-C-specific, not "+autoaug is genuinely better".

## 8. Next hypothesis
Iter 010 (already running, GPU 0): **A4 sweep on Cell C** — autoaug
at lr=0.05 (vs Cell C's 0.1). Predicted: lr=0.05 will land in the
**0.945–0.953 band** with 2 mechanisms competing — slower
optimization may help when paired with autoaug's strong regularizer
(less aggressive memorization → smoother class manifolds), but a
60-ep budget at half the LR may simply under-train (test_acc trajectory
will not have descended its cosine far enough). Falsifier on the
**upside**: a result > 0.9558 (above Cell C s=42's upper noise band)
would mean lr is the next high-leverage axis on autoaug and the matrix
should expand to A4=0.05 as a new column. Falsifier on the
**downside**: < 0.948 (below Cell C s=42's lower noise band) would
mean lr=0.1 is already optimal on this recipe and A4 is exhausted.
Mechanism check: train-vs-test gap should stay near Cell C s=42's
1.0 pp (autoaug regularizer dominates); a wider gap (> 2 pp) would
mean lr=0.05 broke the autoaug mechanism. After iter 010, the next
high-leverage probe is **Cell B + label smoothing or mixup** (a new
axis outside the program.md catalog, so requires a HARD-CONSTRAINT
review) since iter-007 §8 showed the cat↔dog residual is *not*
addressable by any catalog axis.
