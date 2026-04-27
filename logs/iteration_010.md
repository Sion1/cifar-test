# Iteration 010 — cifar10_iter010_autoaug_lr05 (A4 sweep on Cell C: lr 0.1 → 0.05)
Date: 2026-04-27 14:25 | GPU: 0 | Duration: ~35 min (60 epochs × ~31 s/epoch)

## 1. Hypothesis
Phase 3 has opened (Cell C crowned phase-2 winner per iter 009). This iter is
the **A4 sweep on the winner**: single-axis delta vs iter 004 / Cell C is
`training.lr: 0.1 → 0.05` (stays inside catalog A4 = {0.1, 0.05, 0.01} for
SGD); everything else identical (aug=autoaugment, sgd, momentum 0.9, wd 5e-4,
nesterov, cosine, epochs=60, seed=42). Mechanism: Cell C had train_acc=0.9625
with only 1.0 pp train-vs-test gap — explicit under-fit headroom no completed
cell has consumed. Halving the base LR gives a smoother cosine trajectory
(smaller per-step noise → finer fitting in cosine's tail) which on an
*under-fit* recipe could consume that headroom without trading off autoaug's
regularization, unlike Cell F (long-train on the *saturated* Cell B recipe,
which failed). Predicted: best_acc ∈ [0.953, 0.960]; train_acc rises toward
0.97–0.98 (gap shrinks below Cell C's 1.0 pp ⇒ mechanism fired).

## 2. Falsification criterion
Strong falsifier on the **downside** (per the iter010 config header):
**best_acc ≤ 0.9528** (Cell C s=42's value) ⇒ LR=0.1 was already optimal for
autoaug too, and the under-fit signal in Cell C's history was not LR-actionable.
A drop > 0.5 pp would also pin A4=0.05 as a clean negative on this recipe.
Mechanism falsifier: gap stays ≥ 1.0 pp (or *widens*) ⇒ headroom is not
LR-consumable — the under-fit signal was a regularizer-strength artifact, not
a step-size issue. Iter-009 §8's softer prediction band was [0.945, 0.953]
with a downside falsifier at < 0.948.

## 3. Changes made
New file `configs/ablation/iter010_autoaug_lr05.yaml`. Diff vs
`configs/ablation/iter004_autoaug.yaml` (Cell C seed=42):

```diff
 exp_name: cifar10_iter010_autoaug_lr05
 training:
   optimizer: sgd
-  lr: 0.1
+  lr: 0.05
   momentum: 0.9
   weight_decay: 5.0e-4
   nesterov: true
   scheduler: cosine
   epochs: 60
```

`exp_name` updated. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Cell B s=42 | Cell C s=42 (iter004) | Cell C s=4078 (iter008) | This run (autoaug lr=0.05) | Δ vs Cell C s=42 | Δ vs Cell C 2-seed mean | Δ vs A |
|---|---|---|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | 0.9528 | 0.9497 | **0.9513** | **−0.0015** | **+0.00005** | **+0.0643** |
| final_acc  | 0.8868 | 0.9478 | 0.9522 | 0.9497 | 0.9513     | −0.0009     | +0.0001                 | +0.0645 |
| test_loss  | 0.4231 | 0.2106 | 0.1535 | 0.1599 | **0.1625** | +0.0090     | −0.0042                 | −0.2606 |
| train_acc  | 1.0000 | 0.9993 | 0.9625 | 0.9618 | **0.9671** | **+0.0046** | +0.0049                 | −0.0329 |
| best_epoch | 52/60  | 57/60  | 58/60  | 59/60  | **59/60**  | —           | —                        | — |
| epochs run | 60     | 60     | 60     | 60     | 60         | —           | —                        | — |

Source: `runs/cifar10_iter010_autoaug_lr05/final.pth`,
`ckpt['metrics'] = {'acc': 0.9513, 'loss': 0.1625, 'best_acc': 0.9513,
'best_epoch': 59}` (best is the LAST epoch). Train-vs-test gap =
0.9671 − 0.9513 = **1.58 pp** vs Cell C s=42's **1.03 pp** (Cell C s=4078:
1.21 pp). **Gap *widened* by +0.55 pp**, not shrunk — the mechanism prediction
fails directionally.

Trajectory check (autoaug + lr=0.05 + cosine):
- ep 0:  test_acc 0.3798 (Cell C s=42 ep0: ~0.34 — slightly faster start at
  half LR, sensible: less initial overshoot)
- ep 10: 0.8169 (Cell C s=42: ~0.78 — half-LR runs slightly *ahead* early)
- ep 20: 0.8659 (Cell C s=42: ~0.86 — converged to roughly equal)
- ep 30: 0.9003 (Cell C s=42: 0.8774 — half-LR ~2.3 pp ahead at midpoint)
- ep 40: 0.9226
- ep 50: 0.9434
- ep 55: 0.9493
- ep 56: 0.9502
- ep 57: 0.9510
- ep 58: 0.9505
- ep 59 (best): **0.9513**

Last 10 epochs: 0.9434, 0.9438, 0.9456, 0.9468, 0.9484, 0.9493, 0.9502,
0.9510, 0.9505, 0.9513 — late slope still mildly positive (+~0.4 pp over
last 5 epochs), so the tail is still earning a *little* acc, but gap widened
because train_acc rose faster (0.9558 → 0.9671 = +1.13 pp over the last 10
epochs while test_acc rose +0.79 pp). I.e., the extra fitting that lr=0.05
bought went **disproportionately into the training set** — the opposite of
the under-fit-headroom-consumption mechanism.

**Headline:** 0.9513 lands inside iter-009 §8's softer band [0.945, 0.953]
(downside falsifier < 0.948 NOT triggered; upside falsifier > 0.9558 NOT
triggered) but **fails the iter010 config's stricter downside falsifier**
(best_acc ≤ 0.9528 ⇒ "LR=0.1 was already optimal for autoaug; under-fit
signal in Cell C's history was not LR-actionable"). Δ vs Cell C s=42 =
−0.15 pp (inside §Verdict Noise band |Δ| < 0.3 pp); Δ vs Cell C 2-seed
mean (0.95125) = +0.005 pp (essentially identical). Mechanism prediction
(gap shrinks below 1.0 pp) **failed in the opposite direction** — gap
widened from 1.03 → 1.58 pp.

## 5. Visualization evidence

**Per-class (`figs/iter_010/per_class.csv`, with Cell C s=42 re-measured from
`runs/cifar10_iter004_autoaug/best.pth` for honest Δ).** Per-class accuracy
and Δ vs Cell C s=42 (this iter − Cell C s=42, in pp):
- airplane:    0.956 → 0.954  (−0.2)
- automobile:  0.983 → 0.985  (+0.2)
- bird:        0.936 → 0.932  (−0.4)
- cat:         0.885 → **0.889** (**+0.4**) ⬆ — small but real
- deer:        0.967 → 0.962  (−0.5)
- dog:         0.915 → 0.917  (+0.2) ⬆
- frog:        0.971 → 0.975  (+0.4)
- horse:       0.974 → 0.971  (−0.3)
- ship:        0.973 → **0.966** (**−0.7**) ⬇⬇ — biggest single loss
- truck:       0.968 → 0.962  (**−0.6**) ⬇

Mean Δ ≈ −0.15 pp ≈ headline −0.15 pp ✓. **Surprise**: lr=0.05 *did* nudge
the cat↔dog residual in the right direction (cat +0.4, dog +0.2; both up
together for the first time in any catalog cell), but the headline got eaten
by **ship −0.7 and truck −0.6** — i.e., **the vehicle subset (airplane/ship/
truck) softened**, which is unusual: the vehicle pair (automobile↔truck) and
ship had been the matrix's most robust separations. Top off-diagonals:
cat→dog=64 (Cell C s=42: 62), dog→cat=55 (s=42: 55). cat↔dog confusion
direction and magnitude are **basically pinned at Cell C s=42 levels** — the
+0.4 cat / +0.2 dog gains came from cat-as-bird / cat-as-deer corrections,
not from the cat↔dog boundary. airplane↔ship: 15 vs 13 (s=42: 14 vs 9) —
**ship is now leaking more both ways** (especially ship→airplane up from 9
to 13), consistent with ship's −0.7 pp loss. truck→automobile=23 (vs s=42's
typical ~12) is the bigger driver of truck's regression: the smoother lr=0.05
cosine *blurred the truck↔automobile boundary*, the matrix's most reliable
separation. **This is the actual mechanism finding**: lr=0.05 trades
slight cat↔dog gains for vehicle-pair erosion, net negative.

**t-SNE (`figs/iter_010/tsne.png`).** Ten well-separated clusters with the
characteristic Cell-C-style structure: **cat (red) and dog (brown) are
distinct lobes** with a small bridge of red points trailing into the dog
cluster from above — *visually equivalent* to Cell C s=42's cat↔dog
separation, no improvement. **Frog (pink) is its own clean isolated cluster
in the top-right** (consistent with frog's stable 0.971 → 0.975). **Automobile
(orange) shows a long tail extending toward the truck (cyan) cluster** — this
is the t-SNE signature of the truck→automobile=23 leakage observed in the
confusion matrix; the boundary is blurred in feature space, not just
instance-wise. **Ship (yellow) cluster has stragglers on its right edge near
the airplane (blue) cluster** and a few points dropping toward the lower-
right — matches the ship −0.7 pp loss. Bird (green) sits between cat (red),
deer (purple), and frog (pink) with a few scatters — bird's −0.4 pp is
distributed across these three neighbors, similar to Cell E's bird-
fragmentation pattern but milder. **No new structural anomalies**, but the
**automobile↔truck blurring is a *new* finding**: this boundary held under
every other catalog perturbation (B, C, D, E, F, B-s=4078, C-s=4078) and
breaks under lr=0.05.

**Grad-CAM (`figs/iter_010/cam.png`).** **7/8 correct** (1 miss: ship →
automobile — the heatmap correctly centers on the ship's metallic
superstructure but the model reads it as a compact mid-frame car shape; the
**identical failure mode** to Cell C s=42's 7/8 miss). All 8 heatmaps show
the **centered round blob signature** characteristic of every non-Cell-B-
seed=42 cell — confirming iter 009's verdict that heatmap shape is a
seed=42-specific artifact of Cell B alone, not a recipe canary. No new
information from Grad-CAM; the visualization adds zero diagnostic value on
this iter, consistent with iter 009's "drop heatmap shape as a mechanism
diagnostic" recommendation.

## 6. Verdict
**Failure** — by the program.md §Verdict criteria, "Failure: mechanism
doesn't fire OR acc drops > 0.5 pp on an expected-positive hypothesis."
Acc didn't drop > 0.5 pp (Δ vs Cell C s=42 is only −0.15 pp), but **the
mechanism (gap shrink consuming Cell C's 1 pp headroom) did not fire — it
fired in the *opposite* direction**, with the gap widening from 1.03 → 1.58
pp. The iter010 config's downside falsifier (best_acc ≤ 0.9528) is
triggered. Precedent: iter-006 (Cell F long-train) was verdicted Failure on
mechanism grounds with Δ_acc = −0.16 pp (essentially the same magnitude
as this iter), so this matches the established convention for
"small-Δ-but-mechanism-failed" cases.

**Crucially, this does NOT change the matrix winner**: vs the Cell C 2-seed
mean (0.95125), this run is +0.005 pp (statistically identical), and 0.9513
sits between Cell C s=42's 0.9528 and Cell C s=4078's 0.9497, comfortably
inside Cell C's known seed variance (peak-to-peak 0.31 pp). So lr=0.05 is
**effectively a third-seed replay of Cell C with a slightly different
optimization noise profile**, not a meaningfully different recipe at the
2-seed-mean level. **Cell C remains crowned (2-seed mean 0.95125).**

## 7. Decision
**Discard lr=0.05 as a Cell C improvement.** Update CLAUDE.md matrix to note
the A4 axis has been probed at 0.05 on Cell C with a Failure verdict (no
mechanism fire). Do NOT propagate this knob choice anywhere; do NOT run
A4=0.01 on Cell C (it would push under-fit further — predictable Failure).
Do NOT replay lr=0.05 at seed=4078 — the mechanism failed at seed=42 so a
seed-replay can't rescue it. The A4 axis is now exhausted on the autoaug
recipe within the catalog. **Phase 3 should pivot to A5 (weight decay) on
Cell C**: Cell C's 1.0 pp gap is the same regularization signal that lr=0.05
just failed to consume; a *stronger regularizer* (lower LR was the wrong tool;
higher wd is the right tool) is the next high-leverage probe. The A5 catalog
is {5e-4, 1e-4, 0} — only the wd=0 direction goes the wrong way; **wd=1e-4
is one of the two valid catalog moves and it goes the wrong direction
(weaker regularization, will *widen* the gap)**, so neither catalog A5 value
is predicted-positive on Cell C. **Catalog A5 axis is therefore exhausted on
the winner without a run** — both alternative values either weaken the
regularizer (predicted-Failure) or are the current value.

This puts phase 3 in a tight spot: A1, A2, A3, A6 are all pinned by phase 1
single-seed evidence; A4=0.05 just failed on the winner; A4=0.01 is
predicted-Failure (deeper under-fit); A5={1e-4, 0} are both predicted-Failure
(weaker regularization on a recipe with a 1 pp gap that's already mostly
generalization-bound, not memorization-bound). **The catalog is essentially
exhausted on Cell C.** The remaining options inside the catalog are: (a)
2-seed Cell C+lr=0.05 hardening — wasteful given mechanism failed, skip;
(b) exit early on the §Budget rule. The off-catalog probe (label smoothing /
mixup / dropout for the cat↔dog residual) flagged in iter-007 §8 and iter-009
§8 is the only remaining high-leverage experiment, but per program.md HARD
CONSTRAINTS §9 it requires a human editing program.md first. Flag this for
the human as the recommended pivot.

## 8. Next hypothesis
Within-catalog options are essentially exhausted on Cell C. The next iter
should either:
**(a)** run **A5 sweep wd=1e-4 on Cell B** (NOT Cell C) — Cell B has a 5.2
pp gap (real memorization-bound recipe), so weakening wd would *predictably*
widen it further (Failure), but the *opposite* direction (wd higher than
5e-4) is not in the catalog, so this just confirms A5 axis is exhausted.
Skip.
**(b)** run **A4=0.01 on Cell B** — Cell B's lr=0.1 fitted to memorization
(train_acc=0.9993); halving twice (0.025 then 0.0125) would slow that, but
0.01 is the catalog floor. With Cell B's 5.2 pp gap, smaller LR = LESS
memorization = potentially better generalization, the opposite logic of
this iter's failed prediction. This is a genuinely-novel probe that hasn't
been tried; predicted band [0.940, 0.952] with the upside being the only
remaining catalog way to potentially crack 0.953 without leaving Cell B's
known-stable seed profile. Pick this.
**(c)** flag the off-catalog need (label smoothing for cat↔dog) to the
human and await program.md amendment.

Default pick for iter 011 (loop tick): **(b) Cell B + lr=0.01** — single-axis
delta vs iter 003 (`training.lr: 0.1 → 0.01`); falsifier on the upside
> 0.9528 ⇒ A4 axis on Cell B has real headroom and matrix should expand;
falsifier on downside < 0.940 ⇒ A4=0.01 under-trains within 60 ep on this
recipe.
