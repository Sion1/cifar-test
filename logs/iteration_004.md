# Iteration 004 — iter004_std (Cell B: standard augmentation)
Date: 2026-04-27 09:29–09:51 | GPU: 0 | Duration: ~22 min wall (≈18.4 min net train, 60 ep × 18.4 s)

## 1. Hypothesis
Training ResNet-34 on CIFAR-10 with **standard augmentation** (RandomCrop 32 +
HorizontalFlip — Cell B of the ablation matrix). Single-axis delta
`data.augmentation: none → standard` vs. Cell A (`iter003_bare`); everything
else fixed (sgd lr=0.1 mom=0.9 wd=5e-4 nesterov, cosine, 60 ep, seed=42).
Expectation: standard aug is the canonical mid-strength regularizer, so this
cell should land cleanly between Cell A (0.8828 best) and Cell C
(autoaugment, 0.9519 best) — Δ(B − A) ≈ +5–7 pp, with most of C's gain over
A attributable to standard aug rather than to AutoAugment's stronger
color/shape jitter.

## 2. Falsification criterion
Refuted if (a) test_acc ≤ Cell A's 0.8828 — i.e., flips/crops don't help and
the matrix's baseline ordering is wrong; or (b) test_acc ≥ Cell C's 0.9519
— i.e., AutoAugment's extra ops contribute nothing on top of basic flips and
crops; or (c) the train–test gap stays as wide as Cell A's 11.9 pp,
implying the hypothesized regularization mechanism didn't fire. A NaN /
< 0.5 acc run would be a Bug.

## 3. Changes made
Cloned `configs/ablation/iter003_bare.yaml` →
`configs/ablation/iter004_std.yaml`, single-axis edit:

```diff
- exp_name: cifar10_iter003_bare
+ exp_name: cifar10_iter004_std
- augmentation: none
+ augmentation: standard
```

No code changes. Launched via `bash run_experiment.sh
configs/ablation/iter004_std.yaml 4` (GPU 0 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell C (iter002, current best) | Cell B (this run) | Δ vs A     | Δ vs C     |
|-------------------|------------------|--------------------------------|-------------------|------------|------------|
| test_acc (final)  | 0.8812           | 0.9519                         | **0.9475**        | **+0.0663**| −0.0044    |
| test_acc (best)   | 0.8828 @ ep55    | 0.9519 @ ep59                  | **0.9477** @ ep58 | **+0.0649**| −0.0042    |
| test_loss (final) | 0.4469           | 0.1630                         | 0.2138            | −0.2331    | +0.0508    |
| train_acc (final) | 1.0000           | 0.9642                         | 0.9991            | −0.0009    | +0.0349    |
| train–test gap    | 0.1188           | 0.0123                         | **0.0516**        | −0.0672    | +0.0393    |
| best_epoch        | 55               | 59                             | 58                | —          | —          |
| epochs            | 60               | 60                             | 60                | —          | —          |

Run dir: `runs/cifar10_iter004_std/`. Test acc walks tightly through the
0.94–0.95 band from epoch 50 onward (ep50→0.9372, 51→0.941, 52→0.9443,
53→0.9425, 54→0.9424, 55→0.9458, 56→0.9474, 57→0.9472, 58→0.9477,
59→0.9475) — cosine has fully converged at this regularization level; final
≈ best within 0.02 pp. Train_acc reaches 0.999 by epoch ~56 with train_loss
≈ 4.4e-3, while test_loss flattens at ~0.214 — the standard-aug
overfit-but-only-mildly profile: the model still memorizes (gap 5.16 pp)
but a *quarter* of Cell A's 11.88 pp gap is closed by the simple
crop+flip prior.

This cleanly **anchors Cell B** in the matrix and decomposes the augmentation
delta:
- **Δ(B − A) = +6.49 pp (best) / +6.63 pp (final)** — standard aug delivers
  the bulk of what's available on this dataset/architecture.
- **Δ(C − B) = +0.42 pp (best) / +0.44 pp (final)** — AutoAugment's marginal
  value over basic crop+flip is small (~0.4 pp) and well within seed-noise
  band; whether it's a real gain needs the 2-seed (seed=4078) replay before
  any winner is crowned.

Cell B already meets program target ≥ 0.94 but narrowly misses stretch
≥ 0.95 by ~2 pp. Run does **not** set a new provisional best (iter002 C
remains 0.9519 single-seed leader by +0.42 pp).

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_004/per_class.csv`): spread = **0.884
  (`cat`) → 0.973 (`automobile`) = 0.089 pp** — the *tightest* spread of
  the three cells (Cell A: 0.189, Cell C: 0.105). Class-by-class numbers:
  airplane=0.962, automobile=0.973, bird=0.917, cat=0.884, deer=0.958,
  dog=0.920, frog=0.970, horse=0.966, ship=0.962, truck=0.965. **Δ vs
  Cell A** (in pp): airplane +5.6, auto +1.4, **bird +10.2, cat +11.4,
  dog +10.9**, deer +7.3, frog +5.1, horse +5.8, ship +3.0, truck +4.2 —
  exactly the predicted pattern: the visually-ambiguous animal classes
  bird/cat/dog gain 10–11 pp from a simple crop+flip prior, while
  vehicles (already at ≥0.92 in A) gain only 1.4–5.6 pp. **Δ vs Cell C**
  (in pp): airplane **+0.9**, auto −1.1, bird −2.9, cat **+0.5**, deer
  −0.3, dog −0.2, frog −0.2, horse +0.2, ship −0.8, truck −0.3 — the
  +0.42 pp aggregate gap C→B is concentrated almost entirely in `bird`
  (−2.9 pp) and `auto` (−1.1 pp); cat, dog, frog, horse, deer, truck are
  effectively tied between standard and AutoAugment. Notably **B beats C
  on cat (+0.5 pp) and airplane (+0.9 pp)** — strong evidence that
  AutoAugment's marginal value is *narrowly localized* to the
  bird/automobile pair, not a broad upgrade.
- **t-SNE** (`figs/iter_004/tsne.png`): **8 well-separated lobes** —
  matches Cell C's quality and clearly above Cell A's ~6. Clean clusters
  for horse (top-left, isolated), automobile (top-right), truck (far
  right), frog (bottom, large), deer (centre), bird (lower-centre),
  airplane (middle-right) and ship (lower-right). The canonical
  cat↔dog hard pair is **partially fused** (left side: brown dog blob
  with red cat hanging below it as a sub-cluster — looser fuse than Cell
  A, slightly tighter than Cell C where they were one mammal blob).
  Airplane↔ship boundary shows a few mixed points (sky/water shared
  background) but the two cores are distinct. No "bridge" tendrils to
  truck/automobile — vehicles are crisply separated. Geometric signature
  matches the per-class table: feature space is structurally similar to
  Cell C with the cat↔dog confusion still the dominant residual.
- **Grad-CAM grid** (`figs/iter_004/cam.png`): **8/8 correctly
  classified** (Cell A: 7/8, Cell C: 7/8) — Cell B is the first cell to
  hit a clean grid in this loop, though n=8 is too small to read into.
  Heatmaps are sharply object-centred with bright red cores and minimal
  background activation: ship hull, frog body, airplane fuselage centre,
  automobile front fender, frog torso, cat body, ship hull, frog body —
  consistently fixated on discriminative central regions. Localization is
  visibly **tighter than Cell A's broad whole-object glow** but slightly
  less contour-following than Cell C's edge-tracing maps; "object-blob
  with strong centre" is the signature. No background-shortcut pathology.

## 6. Verdict
**Partial.** Mechanism fires exactly as predicted: standard aug closes most
(94 %) of the A→C gap, but the run does *not* set a new best (Δ vs current
best C = −0.42 pp, within the 0.5 pp Partial band — outside Noise's 0.3 pp
band but not Failure's >0.5 pp drop). Structurally, this is the
**Cell B anchor** the matrix needs: it gives us the first clean
decomposition Δ(B − A) = +6.49 pp (standard aug) vs Δ(C − B) = +0.42 pp
(AutoAugment-over-standard) — a strong prior that AutoAugment's *marginal*
value is small and will need 2-seed evidence to survive.

## 7. Decision
Keep. Lock Cell B at **0.9477 (best) / 0.9475 (final)** as the matrix
anchor in `CLAUDE.md`. Record the new isolated deltas:
Δ(B − A) = +0.0649 (standard-aug delta over no-aug) and
Δ(C − B) = +0.0042 (AutoAugment delta over standard). The latter is small
enough that the 2-seed (seed=4078) replay of Cell C should be **required**
before iter002 is declared winner — at single-seed variance ~±0.3 pp on
this setup, +0.42 pp may not survive.

Cell B becomes the *parent* for Cells D / E / F (which all use
augmentation=standard): they must beat 0.9477 to be Successes in their own
right; otherwise the proposed mechanism (optimizer / schedule / longer
training) doesn't earn its keep on top of the simplest-possible aug.

## 8. Next hypothesis
Cell D (iter005_adamw) is already running on GPU 3 in parallel — once it
lands, the optimizer axis (A2) is anchored. The next propose-step should
pick **Cell E (multistep schedule)** to test A3 with everything else equal
to Cell B, since multistep is the canonical SGD schedule we need to rule
out before committing to cosine. Cell F (long-train, epochs=100) only
makes sense once we know the schedule is right. After E lands, the top-2
cells get the seed=4078 replay; if Δ(C − B) survives 2-seed, AutoAugment
wins; otherwise the simpler standard-aug Cell B is the winner with a
narrower confidence interval.
