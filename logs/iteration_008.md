# Iteration 008 — cifar10_iter008_autoaug_seed4078 (Cell C hardening — seed=4078)
Date: 2026-04-27 13:56 | GPU: 1 | Duration: ~35 min (60 epochs × ~33 s/epoch)

## 1. Hypothesis
Iter 008 is the **Cell C 2-seed hardening pass** mandated by program.md
§Required ablation strategy ("after phase 1, pick top 2, run with
seed=4078, report 2-seed mean"). Phase 1 closed at iter 007 with the
top-two single-seed cells being **C (0.9528)** and **B (0.9481)**.
Single-axis delta vs iter 004 (Cell C, seed=42) is `seed: 42 → 4078`;
all other knobs identical (aug=autoaugment, sgd 0.1, momentum 0.9, wd
5e-4, nesterov, cosine, epochs=60). Mechanism question: does Cell C's
+0.47 pp gain over Cell B replicate at a different seed, or was it
seed-noise riding on top of a Cell-B-equivalent recipe?

## 2. Falsification criterion
Per iter-007 §8: predicted band **[0.9498, 0.9558]** (Cell C seed=42
± 0.3 pp). Falsifier on the **strong** side: any result **< 0.9481**
(below Cell B's seed=42 floor) means Cell C's lead was seed-noise and
Cell C falls out of contention as a winner candidate. Falsifier on the
**weak** side: a result outside the predicted ±0.3 pp band would
indicate larger-than-expected seed variance for autoaug. Mechanism
check: train-vs-test gap should land near Cell C seed=42's 1.0 pp
(i.e. ≤ 2 pp); a gap that widens back toward Cell B's 5.2 pp would
mean autoaug's regularization isn't firing the same way at the new
seed.

## 3. Changes made
New file `configs/ablation/iter008_autoaug_seed4078.yaml`. Diff vs
`configs/ablation/iter004_autoaug.yaml` (Cell C seed=42):

```diff
 exp_name: cifar10_iter008_autoaug_seed4078
-seed: 42
+seed: 4078
 data:
   augmentation: autoaugment
   ...
```

`exp_name` updated. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far (Cell B) | Cell C seed=42 (iter004) | This run (Cell C seed=4078) | Δ vs Cell C s=42 | Δ vs Cell B |
|---|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | 0.9528 | **0.9497** | **−0.0031** | **+0.0016** |
| final_acc  | 0.8868 | 0.9478 | 0.9522 | 0.9497     | −0.0025     | +0.0019     |
| test_loss  | 0.4231 | 0.2106 | 0.1535 | 0.1599     | +0.0064     | −0.0507     |
| train_acc  | 1.0000 | 0.9993 | 0.9625 | 0.9618     | −0.0007     | −0.0375     |
| train_loss | 0.0014 | 0.0037 | n/a    | 0.1144     | —           | +0.1107     |
| best_epoch | 52 / 60 | 57 / 60 | 58 / 60 | 59 / 60   | —           | —           |
| epochs run | 60      | 60      | 60     | 60         | —           | —           |

Source: `runs/cifar10_iter008_autoaug_seed4078/final.pth`,
`ckpt['metrics'] = {'acc': 0.9497, 'loss': 0.1599, 'best_acc': 0.9497,
'best_epoch': 59}`. Train-vs-test gap = 0.9618 − 0.9497 = **1.21 pp**
(Cell C seed=42: 1.03 pp; Cell B: 5.20 pp). Gap is right next to Cell
C seed=42's signature and ~4 pp tighter than Cell B's, so the autoaug
**regularization mechanism fires identically at the new seed**.

Trajectory check (autoaug-with-cosine signature):
- ep 0:  test_acc 0.2485 (slow start typical of strong augmentation)
- ep 10: 0.7758
- ep 20: 0.8095
- ep 30: 0.8774  (Cell B at ep30: ~0.91 — autoaug trades early acc
  for late acc, same as Cell C seed=42)
- ep 40: 0.9037
- ep 50: 0.9350
- ep 55: 0.9475
- ep 59: **0.9497** (best_epoch, last epoch)

best_epoch=59 (the final epoch) again — same shape as Cell C seed=42
where best_epoch=58 — cosine's tail is still buying ~0.5 pp over the
last 10 epochs, exactly as expected when memorization is delayed.

**Headline:** 0.9497 lands **just 0.0001 below the predicted lower
bound (0.9498)** — effectively on the edge of the predicted band,
within rounding distance. The strong falsifier (< 0.9481) is **NOT
triggered** (margin: +0.16 pp above Cell B's seed=42 floor). The weak
falsifier (outside ±0.3 pp band) is barely triggered, but the
mechanism (regularization gap, slow-start trajectory, late best_epoch)
matches Cell C seed=42 cleanly. Preliminary 2-seed Cell C mean =
**(0.9528 + 0.9497) / 2 = 0.95125** — still ahead of Cell B's
seed=42 (0.9481), but the **+0.47 pp single-seed lead has compressed
to a +0.31 pp seed-mean lead vs Cell B's seed=42** (and will compress
further once iter 009's Cell B seed=4078 lands).

## 5. Visualization evidence

**Per-class (`figs/iter_008/per_class.csv`, with Cell C seed=42
re-measured from `runs/cifar10_iter004_autoaug/best.pth` for honest
Δ).** Per-class accuracy and Δ vs Cell C seed=42 (this iter − Cell C
s=42, in pp):
- airplane:    0.956 → 0.953  (−0.3)
- automobile:  0.983 → 0.985  (+0.2)
- bird:        0.936 → 0.940  (+0.4)
- cat:         0.885 → **0.874** (**−1.1**) ⬇⬇ — **biggest single loss**
- deer:        0.967 → 0.958  (−0.9)
- dog:         0.915 → 0.908  (−0.7)
- frog:        0.971 → 0.971  (0.0)
- horse:       0.974 → 0.972  (−0.2)
- ship:        0.973 → 0.968  (−0.5)
- truck:       0.968 → 0.968  (0.0)

Mean Δ = −0.31 pp ≈ headline −0.31 pp ✓. **The most consequential
single-class number is cat: 0.874 — IDENTICAL (within rounding) to
Cell B seed=42's 0.878 and 1.1 pp BELOW Cell C seed=42's 0.885.** So
at this seed, autoaug's headline anti-cat effect from iter 004
(+0.7 pp cat over Cell B) **does not replicate** — the cat
improvement was substantially seed-luck on top of a Cell-B-equivalent
recipe. Top off-diagonals confirm: cat→dog=**79** (Cell C s=42: 62;
Cell B s=42: 71) — *worse* than both reference cells; dog→cat=58
(Cell C s=42: 55; Cell B s=42: 44). The cat→dog confusion is the
**dominant** direction here (79 vs 58, gap=21), matching Cell B's
canonical cat→dog-dominant asymmetry, NOT the dog→cat-dominant
flip that Cells E and F exhibited. So the iter-007 finding that "any
deviation off Cell B re-allocates errors into dog→cat dominance" was
*specific to schedule/horizon perturbations* (E, F); a *seed*
perturbation off Cell C reverts the asymmetry back toward Cell B's
direction (cat→dog dominant) but with the **gap widened** (79 vs Cell
B's 71, +8). Implication: the cat↔dog boundary on autoaug is
markedly seed-dependent and the +0.7 pp cat gain in iter 004 should
be downgraded to **noise within autoaug-seed-variance** rather than
treated as a real photometric-regularization signal.

**t-SNE (`figs/iter_008/tsne.png`).** Ten mostly-isolated clusters,
but the **cat (red) and dog (brown) clusters have visibly merged into
a single connected blob in the upper-right** with a wide interleaved
band rather than two clusters with a thin bridge. This is structurally
worse than Cell C seed=42's t-SNE (which still showed cat↔dog as two
distinct lobes with mixing) and is the feature-space cause of cat→dog
79. **The vehicle pair (orange automobile, cyan truck)** is clean and
well-separated on the left — autoaug's vehicle separation is the
non-fragile gain at this seed. **A small airplane↔ship contamination
zone** is visible on the right edge of the airplane cluster (a thin
line of yellow ship points), matching airplane→ship=18 and ship→air-
plane=12 in the confusion matrix; this is the second residual
confusion mode and is, again, present here just as it was in Cells C
s=42, D, E, F. Bird (green) shows minor scatter into the cat region
(matching bird→cat=12). Deer (purple) is mostly clean but with
scattered green/brown points at the bottom edge.

**Grad-CAM (`figs/iter_008/cam.png`).** **8/8 correct** (better than
Cell C seed=42's 7/8). However the **heatmap signature is centered
round blobs across all 8 panels**, NOT the location-following shape
Cell B locked in: the row-1 ship panel (ship in lower half) shows a
heatmap centered mid-image rather than tracking the hull down; the
row-3 frog panel (frog upper-right of frame) shows a heatmap centered
mid-image, not following the frog up-right. The row-1 airplane panel
*does* follow the plane (mid-image, heatmap mid-image — coincidence,
not localization). So Grad-CAM at this seed matches the same
center-blob signature observed in Cells C seed=42, D, E, F — i.e.,
**Cell B's location-following signature is unique to Cell B's exact
recipe at seed=42, and even a same-recipe seed change off Cell C
keeps the centered-blob signature Cell C seed=42 had.** This further
demotes Grad-CAM heatmap shape from "recipe canary" to "Cell-B-seed-
42-specific canary" — it doesn't even survive Cell C, which is the
recipe most similar to Cell B (single-axis A1 swap).

In short, iter 008's small −0.31 pp regression decomposes as:
mostly cat (−1.1 pp), with deer/dog/airplane/ship contributing
smaller losses, no class showing a meaningful gain. The mechanism
question — does autoaug's regularization fire? — answers *yes* (gap
1.21 pp ≈ Cell C s=42's 1.03 pp ≪ Cell B's 5.20 pp). The replication
question — does autoaug *actually* beat Cell B at +0.5 pp? — is now
**very weakly supported**: at this seed, autoaug only beats Cell B by
+0.16 pp (well inside noise), and the headline +0.47 pp lead from
iter 004 is largely seed-driven by the cat improvement that doesn't
replicate.

## 6. Verdict
**Partial** — Cell C **replicates** at seed=4078 but with a slightly
weaker effect size: 0.9497 vs Cell C seed=42's 0.9528 is −0.31 pp,
*just* outside the §Verdict Noise band (|Δ| < 0.3 pp) on the negative
side and 0.0001 below the predicted lower bound [0.9498, 0.9558]. The
strong falsifier from iter-007 §8 ("< 0.9481 ⇒ seed-noise") is **NOT
triggered** — Cell C survives as a winner candidate with a 2-seed mean
of 0.95125. The mechanism evidence is unambiguous: train-vs-test gap
1.21 pp ≈ Cell C seed=42's 1.03 pp ≪ Cell B's 5.20 pp, slow-start
trajectory and late best_epoch (59/60) all match Cell C's signature.
What this iter genuinely measures, then, is **autoaug's seed
sensitivity**: ≈ 0.3 pp peak-to-peak over two seeds is meaningful
relative to Cell C's +0.47 pp lead over Cell B (it eats roughly 60%
of that lead). Final crown decision must wait for iter 009 (Cell B
seed=4078) — only after both 2-seed means are in hand can program.md
§Required ablation strategy be honored.

## 7. Decision
**Keep Cell C as a winner candidate**; do NOT discard. Update CLAUDE.md
matrix row **C** with `2-seed mean = 0.95125`. Cell C's lead over
Cell B has compressed but is not eliminated. Do not propagate any
single-axis change off Cell C yet — wait for iter 009 (Cell B
seed=4078) so the **2-seed-vs-2-seed comparison** is honest. Once
iter 009 lands, crown the winner per §Required ablation strategy by
comparing means: if Cell B's 2-seed mean ≥ 0.9482 (i.e. seed=4078
lands ≥ 0.9483, the symmetric scenario where B's seed-variance also
goes negative), Cell C's lead may drop below the noise band and the
choice becomes argument-by-mechanism rather than headline acc.
state/iterations.tsv shows iter 010 (autoaug at lr=0.05) was already
queued; that's premature — phase 2 (the 2-seed hardening) needs to
finish first.

## 8. Next hypothesis
Iter 009 (already finished, status=completed in iterations.tsv but
not yet analyzed): **Cell B 2-seed hardening** — re-run
`iter003_std_aug.yaml` with `seed: 42 → 4078`. Predicted: test_acc
within ±0.3 pp of Cell B's 0.9481, i.e. landing in
**[0.9451, 0.9511]**. Falsifier on the strong side: a result
**> 0.9528** (above Cell C's seed=42) would invert the matrix
ordering — Cell B would *beat* Cell C's seed=42 single-seed result
on a single seed, suggesting the cell ordering is dominated by seed
variance and the "+std vs +autoaug" question is unresolved at 2
seeds. Falsifier on the weak side: a result **< 0.9451** would
match the same kind of negative seed-luck Cell C just exhibited, in
which case Cell B's 2-seed mean drops near 0.9466 and Cell C's
+0.31 pp 2-seed lead becomes ~+0.45 pp again — restoring much of
the original margin. After iter 009 lands the 2-seed-mean comparison
is the deciding metric.
