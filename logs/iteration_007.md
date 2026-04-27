# Iteration 007 — cifar10_iter007_multistep (Cell E — multistep)
Date: 2026-04-27 13:49 | GPU: 0 | Duration: ~34 min (60 epochs × ~33 s/epoch)

## 1. Hypothesis
Iter 007 targets **Cell E — multistep** (program.md §Required ablation
strategy): single-axis delta vs Cell B (iter 003) is
`training.scheduler: cosine → multistep` with `milestones=[30, 45]`,
`gamma=0.1` (the canonical CIFAR ResNet multistep recipe inside the
A3 catalog). All other knobs identical to iter003 (aug=standard,
sgd 0.1, momentum 0.9, wd 5e-4, nesterov, epochs=60, seed=42).
Mechanism: multistep's discrete LR drops sometimes outperform cosine
at 60 ep on CIFAR ResNets when a milestone aligns with a generalization
plateau the loss curve hits around epoch 30; the LR drop "breaks
through" the plateau more abruptly than cosine's smooth decay can.
Predicted: **Partial** most likely (Δ within ±0.3 pp of Cell B), with
a small chance of Success (≥ 0.9531) if the milestone alignment is
favorable.

## 2. Falsification criterion
Per iter-006 §8: **"if iter 007 lands ≤ 0.9451 (Cell B −0.3 pp,
outside Noise band on the negative side), multistep is a clean
negative."** The complementary upside falsifier was
"≥ 0.9531 ⇒ joins 2-seed hardening candidates." Mechanism check:
the test_acc trajectory should show a *step-shaped* jump at
epochs 30 and 45 (the milestones); if the jumps are absent or the
ep-30 jump fails to cross Cell B's ep-30 checkpoint of ~0.9078, the
multistep schedule isn't doing the work it's supposed to. The
train-vs-test gap should land near Cell B's 5.2 pp ± 1 pp; a much
wider gap means multistep over-shoots into memorization, a much
narrower gap means it under-fits.

## 3. Changes made
New file `configs/ablation/iter007_multistep.yaml`. Diff vs
`configs/ablation/iter003_std_aug.yaml`:

```diff
 training:
   optimizer: sgd
   lr: 0.1
   momentum: 0.9
   weight_decay: 5.0e-4
   nesterov: true
-  scheduler: cosine
+  scheduler: multistep
+  milestones: [30, 45]
   epochs: 60
```

`exp_name` updated to `cifar10_iter007_multistep`. No code changes.

## 4. Results
| Metric     | Cell A (baseline) | Best so far (Cell B) | This run (Cell E) | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| best_acc   | 0.8870 | 0.9481 | **0.9431** | **−0.0050** | **+0.0561** |
| final_acc  | 0.8868 | 0.9478 | 0.9426     | −0.0052     | +0.0558     |
| test_loss  | 0.4231 | 0.2106 | 0.2434     | +0.0328     | −0.1797     |
| train_acc  | 1.0000 | 0.9993 | 0.9976     | −0.0017     | −0.0024     |
| train_loss | 0.0014 | 0.0037 | 0.0094     | +0.0057     | +0.0080     |
| best_epoch | 52 / 60 | 57 / 60 | 55 / 60   | —           | —           |
| epochs run | 60      | 60      | 60         | —           | —           |

Source: `runs/cifar10_iter007_multistep/final.pth`,
`ckpt['metrics'] = {'acc': 0.9426, 'loss': 0.2434, 'best_acc': 0.9431,
'best_epoch': 55}`. Train-vs-test gap = 0.9976 − 0.9426 = **5.50 pp**
(Cell B: 5.20 pp; close, within the ±1 pp mechanism-check window).

The trajectory is **textbook multistep** with milestones cleanly
visible in the history:
- **ep 0–29 (lr=0.1)**: train_acc plateaus at 0.88–0.89, test_acc at
  0.83–0.84 (ep29 = 0.8398). Cell B at the same epoch is materially
  ahead because cosine has already started decaying the LR.
- **ep 30 (lr→0.01, ×0.1 drop)**: dramatic step up — test_acc jumps
  0.8398 → **0.9253** (+8.55 pp in one epoch), train_acc 0.886 →
  0.938. The first milestone fires exactly as predicted: the LR drop
  breaks through the high-LR plateau abruptly.
- **ep 30–44 (lr=0.01)**: test_acc creeps from 0.9253 to 0.9282
  (+0.29 pp over 15 epochs), train_acc grinds toward 0.984.
- **ep 45 (lr→0.001, ×0.1 drop)**: smaller step — test_acc 0.9282 →
  0.9384 (+1.02 pp), train_acc 0.984 → 0.989.
- **ep 45–59 (lr=0.001)**: gentle plateau 0.94–0.943, best at ep 55
  (0.9431), final ep 59 = 0.9426.

So the **mechanism partially fires**: both LR drops produce visible
step-jumps and the ep-30 drop is the dominant gain (that's the
"break through the plateau" effect). But the falsifier
**≤ 0.9451 ⇒ clean negative** is **triggered** (0.9431 < 0.9451):
multistep's two discrete drops at [30, 45] cannot match cosine's
*continuous* refinement over the same 60 epochs on this recipe. The
upside falsifier (≥ 0.9531) is not even close.

This is a **Failure on direction** (clean negative per iter-006 §8)
even though the magnitude (−0.50 pp) sits *exactly* at the §Verdict
Failure threshold. Cell B (0.9481) remains "Best so far"; Cell E is
discarded as a winner candidate.

## 5. Visualization evidence

**Per-class (`figs/iter_007/per_class.csv`, with Cell B re-measured
from `runs/cifar10_iter003_std_aug/best.pth` for honest Δ).** Spread
0.892 (cat) → 0.977 (automobile) ≈ 8.5 pp, slightly tighter than
Cell B's 9.7 pp because cat improved and several saturated classes
regressed. Per-class Δ vs Cell B (this iter − Cell B, in pp):
- airplane:    0.962 → 0.957  (−0.5)
- automobile:  0.975 → 0.977  (+0.2)
- bird:        0.925 → 0.910  (**−1.5**) ⬇
- cat:         0.878 → 0.892  (**+1.4**) ⬆ — cat improved (again!)
- deer:        0.957 → 0.966  (+0.9)
- dog:         0.930 → 0.901  (**−2.9**) ⬇⬇ — biggest single loss
- frog:        0.956 → 0.953  (−0.3)
- horse:       0.966 → 0.958  (−0.8)
- ship:        0.967 → 0.956  (**−1.1**) ⬇
- truck:       0.964 → 0.961  (−0.3)

Mean Δ = −0.49 pp ≈ headline −0.50 pp ✓. **The cat-up-dog-down
asymmetry is the same pattern iter 006 (Cell F long-train)
exhibited**, but more pronounced: Cell F was cat +1.5 / dog −2.5,
this iter is cat +1.4 / dog −2.9. The top off-diagonal confusions
flipped: Cell B was **cat→dog dominant (71 vs 44)**; iter 007 Cell E
is **dog→cat dominant (64 vs 50)**. So multistep, like long-train,
re-allocates errors from cat-as-dog into dog-as-cat — polishing the
cat manifold at dog's expense rather than carving a cleaner
boundary. **Two non-Cell-B recipes have now produced this exact
asymmetry-flip** (long-train and multistep), strongly suggesting it
is *the* characteristic failure mode of the cat↔dog boundary on this
recipe whenever the optimization horizon or step structure deviates
from Cell B's smooth 60-ep cosine. Bird's −1.5 pp loss is broad
(bird→airplane=19, bird→deer=19, bird→frog=18, bird→cat=17), not
concentrated on one neighbour — the discrete LR drops appear to
fragment bird's boundary across multiple neighbours rather than
re-allocating it cleanly. Ship's −1.1 pp loss has a clear
counterpart in ship→airplane=27 (the third-largest off-diagonal),
matching the airplane↔ship contamination zone we expected to
reopen.

**t-SNE (`figs/iter_007/tsne.png`).** Ten clusters mostly
well-isolated, but two failure modes are visible. **(1) cat↔dog
bridge is wider than Cell B's**: the red (cat) and brown (dog)
clusters are *connected* by a substantial mixed-color strip in the
upper-middle of the plot, with both colors interleaving across a
visible band — this is the structural cause of the +1.4/−2.9
per-class swing. **(2) airplane↔ship contamination zone has
reopened on the right side**: a thin strip of yellow-green ship
points sits on the right edge of the blue airplane cluster (and a
couple of blue points sit inside the ship cluster), matching the
ship→airplane=27 confusion and the −1.1 pp ship regression. The
automobile↔truck pair (orange ↔ cyan) is touching but mostly clean
(one or two crossover points), so SGD-with-multistep preserves the
vehicle-pair separation that AdamW broke in Cell D. Bird (green) is
isolated but visibly contaminated with stray red/pink/yellow points
inside, consistent with bird's broad confusion pattern. Deer
(purple), horse (gray), and frog (pink) are clean isolated clusters.

**Grad-CAM (`figs/iter_007/cam.png`).** **8/8 correct** (matches
Cell B/F). However the **heatmap signature has drifted back toward
Cell A/D/F's center-bias**, NOT preserving Cell B's location-
following shape — peaks across all 8 panels are compact, roughly
circular red blobs near the image center. The clearest tells: the
row-3 frog panel (frog is upper-right in the frame) shows a
heatmap centered mid-image, not following the frog up-right; the
row-3 cat panel (cat is left-of-center) shows a heatmap centered
on the cat but with a symmetric blob shape rather than left-leaning;
even the row-1 airplane panel (plane nose in lower-left) has a
heatmap that drifts toward image-center. **This is now the third
non-Cell-B cell to lose Cell B's location-following signature**
(after Cell D AdamW and Cell F long-train), confirming iter 005's
hypothesis and iter 006's amendment: **heatmap shape is sensitive
to *any* knob that changes the optimization horizon or per-channel
update mass — not just A1, A2, or A6, but also A3.** The discrete
LR drops at [30, 45] plausibly cause the same effect as longer
schedules: with lr=0.01 for 15 epochs and lr=0.001 for 15 more,
the BN running stats and classifier head get the same kind of
extended low-LR refinement that center-dominant feature channels
benefit from, eroding the off-center prior std-aug initially
established. Cell B's location-following signature is now
*confirmed* as a recipe-fidelity canary that flips off whenever
the schedule shape deviates from cosine-60.

So Cell E's degradation is small (−0.50 pp) but visible at every
level: per-class regression concentrated on dog (−2.9 pp) with cat
trading places (the now-canonical cat↔dog asymmetry-flip), t-SNE
cat↔dog bridge widened and airplane↔ship boundary partially eroded,
and Grad-CAM heatmaps drifted back toward central blobs. Mechanism
prediction (multistep's discrete drops match cosine within ±0.3 pp,
small chance of break-through) falsified on the negative side.

## 6. Verdict
**Failure** — the falsifier from iter-006 §8 ("≤ 0.9451 ⇒ clean
negative") is **triggered**: best_acc 0.9431 is 0.50 pp below Cell B
(0.9481), exactly at the §Verdict Failure boundary
(`acc drops > 0.5 pp on an expected-positive hypothesis`) and well
outside the §Noise band (|Δ| < 0.3 pp). The mechanism partially fires
— the ep-30 milestone produces a clean +8.55 pp step that confirms
the "LR drop breaks the plateau" prior — but on this recipe two
discrete drops at [30, 45] cannot match cosine's continuous decay
over the same 60-ep horizon. Multistep is therefore the **worst of
the three A3 schedulers** tested in the high-LR-then-decay style on
this Cell-B base recipe. Cell B (0.9481) remains "Best so far"; Cell
E is discarded as a winner candidate. The structural finding worth
keeping is mechanical-not-just-numerical: **multistep at [30, 45]
spends the first 30 epochs at lr=0.1 with test_acc stuck in the
0.83–0.84 band**, while cosine at lr=0.1 (Cell B) had already
descended its LR enough by ep 30 to be at ~0.91. So multistep's
ep-30 step-up is impressive but only recovers what its high-LR first
phase had cost; over a 60-ep budget, smooth cosine wins the area
under the curve.

## 7. Decision
**Discard Cell E as a winner candidate; keep as the A3=multistep
negative datapoint.** Cell B (0.9481) remains "Best so far"; do not
propagate A3=multistep to any downstream cell. Update CLAUDE.md
matrix row **E** with `acc=0.9431, Best iter#=7, Verdict=Failure`.
**Phase 1 (Cells A–F) is now complete** — every single-axis catalog
cell has been measured at seed=42:
- A bare = 0.8870 (Success, locked floor)
- B +std = 0.9481 (Success, **Best so far**)
- C +AA  = 0.9528 (Partial, +0.47 pp over B)
- D AdamW = 0.9379 (Failure)
- E +ms  = 0.9431 (Failure, this iter)
- F long = 0.9465 (Failure)

The two strongest single-seed candidates are **C (0.9528)** and **B
(0.9481)** — they are the natural 2-seed hardening pair. Per
program.md §Required ablation strategy ("after phase 1: pick top 2,
run with seed=4078, report 2-seed mean"), the next iters should be
seed=4078 replays of Cell C and Cell B in some order. Note:
state/iterations.tsv already shows iter 008 (autoaug_seed4078) and
iter 009 (std_aug_seed4078) running in parallel — that exactly
matches phase 2's plan, so the loop is already correctly executing
the 2-seed pass.

## 8. Next hypothesis
Iter 008 (already running on GPU 1): **Cell C 2-seed hardening** —
re-run `iter004_autoaug.yaml` with `seed: 42 → 4078`. Predicted:
test_acc within ±0.3 pp of Cell C's 0.9528, i.e. landing in
[0.9498, 0.9558]. Falsifier: a result < 0.9481 (below Cell B's seed=42
floor) would mean Cell C's +0.47 pp gain over B was seed-noise, not
a real photometric-regularization win, and Cell C falls out of
contention. The sibling iter 009 will play the same role for Cell B
(seed=4078 replay of std_aug). After both finish, report the 2-seed
means and crown the winner per program.md §Required ablation strategy.
