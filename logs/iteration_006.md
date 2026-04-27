# Iteration 006 — iter006_multistep (Cell E: multistep LR schedule)
Date: 2026-04-27 10:12–10:33 | GPU: 2 | Duration: ~21 min wall (≈18.7 min net train, 60 ep × 18.7 s)

## 1. Hypothesis
Single-axis swap of A3 from `cosine → multistep` (Cell E of the ablation
matrix) on top of the Cell B recipe (standard aug, sgd 0.1 mom 0.9 nesterov
wd 5e-4, 60 ep, seed=42), with milestones `[30, 45]` (= 50 % / 75 % of the
60-ep budget) and γ=0.1. Folk-wisdom from the older CIFAR literature is
that multistep was the canonical default before cosine took over; the
prior is "lands within ±0.5 pp of Cell B's 0.9477". A Δ > +0.5 pp would
crown multistep as the A3 winner and propagate to Cell F; Δ < −0.5 pp
would confirm cosine's edge on this recipe. The mechanism we expect to
see is a sharp test-acc jump at ep30 (lr 0.1→0.01) and a smaller jump at
ep45 (lr 0.01→0.001).

## 2. Falsification criterion
Refuted if (a) test_acc ≤ 0.94 — multistep clearly underperforms cosine
on this recipe; or (b) test_acc ≥ 0.953 — multistep clearly beats both
Cell B and Cell C and becomes the new propagation parent for Cell F; or
(c) the train–test gap exceeds Cell A's 11.9 pp, indicating multistep
over-memorizes badly. NaN / divergence at high lr would be a Bug.

## 3. Changes made
Cloned `configs/ablation/iter004_std.yaml` →
`configs/ablation/iter006_multistep.yaml`, single-axis edit (A3 swap):

```diff
- exp_name: cifar10_iter004_std
+ exp_name: cifar10_iter006_multistep
  training:
    optimizer: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
    nesterov: true
-   scheduler: cosine
+   scheduler: multistep
+   milestones: [30, 45]
    epochs: 60
```

γ=0.1 is the trainer's hard-coded multistep gamma. No code changes.
Launched via `bash run_experiment.sh configs/ablation/iter006_multistep.yaml 6`
(GPU 2 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell B (iter004, parent) | Cell C (iter002, current best) | Cell E (this run) | Δ vs B      | Δ vs C     | Δ vs A     |
|-------------------|------------------|--------------------------|--------------------------------|-------------------|-------------|------------|------------|
| test_acc (final)  | 0.8812           | 0.9475                   | 0.9519                         | **0.9386**        | **−0.0089** | −0.0133    | +0.0574    |
| test_acc (best)   | 0.8828 @ ep55    | 0.9477 @ ep58            | 0.9519 @ ep59                  | **0.9394 @ ep49** | **−0.0083** | −0.0125    | +0.0566    |
| test_loss (final) | 0.4469           | 0.2138                   | 0.1630                         | **0.2459**        | +0.0321     | +0.0829    | −0.2010    |
| train_acc (final) | 1.0000           | 0.9991                   | 0.9642                         | 0.9981            | −0.0010     | +0.0339    | −0.0019    |
| train–test gap    | 0.1188           | 0.0516                   | 0.0123                         | **0.0595**        | +0.0079     | +0.0472    | −0.0593    |
| best_epoch        | 55               | 58                       | 59                             | 49                | —           | —          | —          |
| epochs            | 60               | 60                       | 60                             | 60                | —           | —          | —          |

Run dir: `runs/cifar10_iter006_multistep/`. The trajectory is exactly the
mechanism the hypothesis predicted, but quantitatively short of Cell B:

- **Pre-milestone-1 (ep0–29)**: test_acc climbs noisily to ~0.81, peaking
  at 0.8418 (ep20) and drifting back to 0.8155 at ep29 — very wide
  per-epoch swings (±2–4 pp epoch-to-epoch) characteristic of high-lr
  SGD with no decay.
- **Milestone 1 at ep30** (lr 0.1 → 0.01): single-epoch jump
  **0.8155 → 0.9191 (+10.4 pp)**, exactly the textbook multistep
  signature. Test acc smooths out and creeps from 0.919 (ep30) to 0.931
  (ep44).
- **Milestone 2 at ep45** (lr 0.01 → 0.001): smaller but visible jump
  **0.9314 → 0.9373 (+0.6 pp)**, then a tight plateau 0.937–0.939 from
  ep45 to ep59. Best lands at **ep49 (0.9394)** and the last 11 epochs
  drift inside 0.5 pp of that high-water-mark.
- **Train–test gap** = 0.9981 − 0.9386 = **5.95 pp**, slightly *wider*
  than Cell B's 5.16 pp despite identical aug, optimizer, and budget;
  test_loss saturates near 0.246 (vs B's 0.214), the loss penalty being
  larger than the acc penalty — more wrong-and-confident predictions.

This **anchors Cell E** in the matrix and yields the schedule-axis delta:

- **Δ(E − B) = −0.83 pp (best) / −0.89 pp (final)** — multistep
  `[30, 45], γ=0.1` *loses* to cosine on the Cell-B recipe. Mechanism
  fires cleanly (visible per-milestone jumps, stable training, no NaN),
  but the integrated improvement of cosine across all 60 epochs
  outweighs multistep's two discrete drops.
- **Δ(E − A) = +5.66 pp (best)** — Cell E still beats the bare-baseline
  floor by a healthy margin, so multistep isn't broken; it's just
  *suboptimal* on this dataset/architecture/budget combo.
- **Δ(E − D) = +0.40 pp (best)** — multistep on SGD beats AdamW-on-cosine
  (Cell D 0.9354), so on this recipe the *optimizer family* matters more
  than the *schedule family*.

Cell E clears the program target ≥ 0.94 cleanly on the *best* checkpoint
(0.9394) but the *final* (0.9386) misses by 0.14 pp; it falls 1.06–1.25
pp short of stretch ≥ 0.95. Run does **not** set a new provisional best;
iter002 Cell C remains 0.9519 leader.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_006/per_class.csv`): spread = **0.872
  (cat) → 0.978 (automobile) = 0.106** — *looser* than Cell B's 0.089
  but tighter than Cell A's 0.189. Class-by-class numbers:
  airplane=0.956, automobile=0.978, bird=0.913, cat=0.872, deer=0.953,
  dog=0.906, frog=0.946, horse=0.956, ship=0.960, truck=0.954.
  **Δ vs Cell B** (in pp, computed against a freshly-generated
  `figs/iter_004/per_class.csv`): airplane −0.6, **automobile +0.5**,
  bird −0.4, **cat −1.2**, deer −0.5, **dog −1.4**, **frog −2.4**,
  horse −1.0, ship −0.2, **truck −1.1**. The −0.83 pp aggregate Δ is
  **disproportionately animal-side** — the six animal classes
  (cat/dog/frog/horse/bird/deer) absorb 6.9 pp of the 8.3 pp net loss
  while the four vehicle classes give up only 1.4 pp (and automobile
  actually *gains* +0.5 pp). The single biggest hit is **frog −2.4 pp**,
  which the multistep schedule fails to lock in despite frog being one
  of Cell B's strongest classes (0.970). This rhymes with Cell D's
  signature (cat/dog/frog/horse also lost most of the AdamW penalty)
  and reinforces a consistent regularization story: when SGD+cosine's
  late-epoch tail of small-lr decreases is replaced — whether by AdamW
  or by multistep's two coarse steps — the animal-shape evidence is
  the first thing to slip, while vehicle classes (whose visual
  signatures are more rigid / linear) are robust.
- **t-SNE** (`figs/iter_006/tsne.png`): **8 visible lobes** — same
  count as Cell B / Cell D and clearly more separated than Cell A's
  ~6. Cleanly isolated clusters: horse (top-centre, grey), deer
  (mid-left, purple), truck (upper-right, cyan), automobile (right
  margin, orange — tightest single cluster on the plot), frog
  (bottom-left, pink), ship (bottom-right, yellow). Two structural
  problem zones consistent with the per-class deltas: (i) the
  **cat↔dog fuse** in the centre-left is a continuous brown→red mass
  rather than two separate lobes — visibly similar in geometry to
  Cell B but with a thicker bridge of mixed-class points along the
  cat/dog seam (consistent with Δcat=−1.2, Δdog=−1.4); (ii) the
  **airplane↔bird seam** in the centre — green bird points sit
  immediately adjacent to the blue airplane cluster, with a few
  cross-class points in between (consistent with Δbird=−0.4,
  Δairplane=−0.6). A small **deer↔frog/dog tendril** dangles off the
  bottom of the deer cluster, mixing pink/red into purple — a minor
  artefact, but absent from Cell B's t-SNE. Overall geometry: same
  lobe *count* as Cell B but with consistently looser boundaries —
  the feature space is similarly *organized* but less *separated*,
  matching the +0.79 pp gap-widening.
- **Grad-CAM grid** (`figs/iter_006/cam.png`): **8/8 correctly
  classified** (ship, frog, airplane, automobile, frog, cat, ship,
  frog). Heatmaps are sharply object-centred with bright red cores on
  every sample: ship hull, frog body in grass, airplane fuselage, car
  body, frog body, cat torso, ship hull, frog body. n=8 caveat applies
  (3× frog skews the panel — and ironically frog is the worst-regressed
  class on the per-class table, but the 3 frog samples shown all
  classify correctly). Localization quality is visually
  indistinguishable from Cell B and Cell D — the multistep schedule
  does **not** induce a qualitatively different saliency pattern; the
  regression is purely a *generalization-margin* problem (test_loss
  drift, looser t-SNE boundaries, animal-side per-class bleeds), not
  an attention-mechanism problem. No background-shortcut pathology.

## 6. Verdict
**Failure.** Mechanism fires cleanly (textbook multistep ladder visible
at ep30 and ep45, no NaN, schedule completes, checkpoint healthy), but
Δ vs the parent Cell B = **−0.83 pp**, well outside the Partial band's
[−0.5, +0.5] and the Noise band's ±0.3 pp. Per program.md §Verdict:
"acc drops > 0.5 pp on an expected-positive hypothesis" → **Failure**.
The schedule-family swap (A3: cosine → multistep) at the
`[30, 45], γ=0.1` config does **not** earn its keep on top of the Cell-B
recipe; cosine's smooth tail of small lr decreases evidently extracts
more accuracy than multistep's two-step decay over the same budget.
Cell E is *anchored* (the matrix has its first schedule data point), but
it's a negative result: multistep is dominated by cosine on this exact
recipe.

## 7. Decision
Discard as a propagation parent — downstream cell **F (long-train, A6 =
100)** should continue to build on **Cell B (sgd, cosine, 0.9477)**, not
on Cell E. Lock Cell E at **0.9394 (best) / 0.9386 (final)** in
`CLAUDE.md`'s matrix as the anchor for the A3 axis. The negative result
is informative: on Cell-B-style 60-ep recipes, the schedule family is
*not* a productive axis to spend further iterations on at the catalog's
current granularity (the only remaining A3 point, `none`, can only do
worse). Do **not** schedule a Cell E follow-up unless later cells leave
budget. After Cell F lands, the dominant remaining priority is the
2-seed (seed=4078) replay of Cells B and C to harden the C − B = +0.42
pp gap before crowning a winner.

## 8. Next hypothesis
Cell F (long-train) — single-axis delta from Cell B with
`training.epochs: 60 → 100`. Test A6: keeping the same standard-aug +
sgd 0.1 + cosine recipe but giving cosine 40 more epochs to anneal
should recover any unlocked headroom in the cosine-tail regime. Prior:
"lands at 0.9477 + 0.2 to 0.5 pp" (≈ 0.950 ± 0.5). Falsifier: Δ vs
Cell B in [−0.5, +0.3] pp = Partial / Noise → 60 ep is the right budget
and we don't need to spend longer; Δ ≥ +0.5 pp → 100-ep cosine becomes
the new propagation parent and should be paired with the seed=4078
replay; Δ ≥ +1.0 pp would supersede Cell C (0.9519) and reorder the
provisional-leader board.
