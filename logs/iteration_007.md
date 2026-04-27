# Iteration 007 — iter007_long (Cell F, long-train, A6=100)
Date: 2026-04-27 10:14–10:49 | GPU: 1 | Duration: ~34 min wall (≈29.8 min net train, 100 ep × 17.9 s)

## 1. Hypothesis
Single-axis swap of `training.epochs: 60 → 100` on the iter004 Cell B recipe
(standard aug, sgd 0.1 mom 0.9 nesterov wd 5e-4, cosine, seed=42). Cosine
T_max scales with the new budget, so this is a clean A6 test rather than a
schedule-shape change. Prior: Cell B's ep59 best (0.9477) was still in the
cosine tail — train_acc = 0.9991 / train_loss ≈ 4e-3 — i.e., the model
hadn't fully exploited the schedule yet. Adding 40 more ep at a slowly-
decaying lr should extract the residual generalization signal still on the
table, producing a B-side anchor in the **0.948–0.955** window. The
expected-positive threshold for a Success label is ≥ +0.5 pp vs the current
best (Cell C 2-seed mean = 0.9524).

## 2. Falsification criterion
Refuted if (a) test_acc < 0.945 — would mean the extra 40 ep overfit and
*lost* generalization (would relabel A6 as a regularization-bound axis on
this recipe); (b) train_acc reaches 1.0 well before ep100 *and* test_acc
plateaus or drops thereafter — overfit signature; (c) the run diverges/NaNs
— Bug. The "successful confirmation" outcome is test_acc ≥ 0.948 with the
Cell B trajectory clearly extending its cosine tail into the new budget.

## 3. Changes made
Cloned `configs/ablation/iter004_std.yaml` →
`configs/ablation/iter007_long.yaml`, single-axis edit (epochs):

```diff
- exp_name: cifar10_iter004_std
+ exp_name: cifar10_iter007_long
  data:
    augmentation: standard
  training:
    optimizer: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
    nesterov: true
    scheduler: cosine
-   epochs: 60
+   epochs: 100
```

No code changes. Launched via
`bash run_experiment.sh configs/ablation/iter007_long.yaml 7`
(GPU 1 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell B (iter004, std s=42) | Cell C s=42 (iter002) | Cell C 2-seed mean | Cell F long s=42 (this run) | Δ vs best (C 2-seed) | Δ vs B s=42 | Δ vs A     |
|-------------------|------------------|----------------------------|------------------------|--------------------|------------------------------|----------------------|-------------|------------|
| test_acc (final)  | 0.8812           | 0.9475                     | 0.9519                 | 0.9524             | **0.9522 @ ep99**            | −0.0002              | +0.0047     | +0.0710    |
| test_acc (best)   | 0.8828 @ ep55    | 0.9477 @ ep58              | 0.9519 @ ep59          | 0.9524             | **0.9531 @ ep98**            | **+0.0007**          | +0.0054     | +0.0703    |
| test_loss (final) | 0.4469           | 0.2138                     | 0.1630                 | —                  | **0.2012**                   | —                    | −0.0126     | −0.2457    |
| train_acc (final) | 1.0000           | 0.9991                     | 0.9642                 | —                  | **0.9999**                   | —                    | +0.0008     | −0.0001    |
| train–test gap    | 0.1188           | 0.0516                     | 0.0123                 | —                  | **0.0477**                   | —                    | −0.0039     | −0.0711    |
| best_epoch        | 55               | 58                         | 59                     | —                  | **98**                       | —                    | —           | —          |
| epochs            | 60               | 60                         | 60                     | —                  | **100**                      | —                    | —           | —          |

Run dir: `runs/cifar10_iter007_long/`. The trajectory walks the Cell B
recipe at the same lr-schedule shape (just stretched 100/60 ≈ 1.67× along
the time axis): ep0=0.271, ep5=0.689, ep10=0.797, ep20=0.830, ep30=0.855,
ep40=0.882, ep50=0.881, ep60=0.885, ep70=0.913, ep80=0.941, ep90=0.951,
ep95=0.953, ep98=0.9531, ep99=0.9522. Best == ep98, with the final epoch
giving back 0.09 pp — typical end-of-cosine micro-noise, not a structural
overfit slope. The early-mid plateau through ep30–60 (0.855→0.885, a
**3.0 pp** drift across 30 epochs at lr ≈ 0.07–0.04) is the cosine
"long-tail" cost: spending 1.67× the budget at near-baseline lr buys most
of its gain only in the last 30 ep, where lr drops below 0.02. The final
40-ep contribution (ep60→99) = +6.8 pp; the matched ep0→59 contribution
= +0.885 − 0.271 = +61.4 pp; **the marginal value per epoch in the tail
(0.17 pp/ep) is ~30× smaller than in the warm-up phase**, but still
non-zero and net-positive — the schedule is *not* saturated at ep59.

The train-test gap **tightens slightly** (4.77 pp at ep99 vs Cell B's
5.16 pp at ep59), and test_loss drops 0.214 → 0.201 — the longer cosine
tail has not introduced overfit; it has produced a marginally
*better-calibrated* solution at the same train_acc level (0.9999 vs
0.9991). This rules out falsifier (a) and (b).

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_007/per_class.csv`): spread = **0.887
  (cat) → 0.983 (automobile) = 0.096** — slightly looser than Cell B's
  0.089 but tighter than Cell C s=4078's 0.099. Class numbers:
  airplane=0.962, automobile=0.983, bird=0.941, cat=0.887, deer=0.962,
  dog=0.923, frog=0.972, horse=0.961, ship=0.967, truck=0.973. Deltas
  vs Cell B (iter004) sum to +5.4 / 1000 = +0.54 pp and decompose as:
  **bird +2.4** (biggest gain — recovers 83 % of the −2.9 pp gap Cell B
  had vs Cell C), automobile +1.0, truck +0.8, ship +0.5, deer +0.4,
  cat +0.3, dog +0.3, frog +0.2, airplane 0.0, **horse −0.5** (the only
  class that regresses, but well within the ±1 pp seed-noise band). The
  long-tail gain is **not uniformly distributed** — it concentrates in
  bird and the vehicle classes (auto/truck/ship +2.3 pp combined),
  exactly the classes where Cell B was weakest relative to Cell C; the
  cat↔dog fusion that bottlenecks both B and C *barely moves* (+0.3 /
  +0.3). Per-class deltas vs the Cell C 2-seed mean (using the s=4078
  numbers as a stand-in): bird −0.2, automobile −0.2, cat +0.1, dog +0.9
  (!), horse −1.4, ship −0.9, truck +0.7 — Cell F holds its own on
  cat/dog vs autoaug but loses ground on horse/ship; this confirms the
  long cosine tail is helping a *different* failure-mode than autoaug.
- **t-SNE** (`figs/iter_007/tsne.png`): **all 10 classes visible as
  named lobes**, with the canonical Cell-B fingerprint preserved: cat
  (red, centre-left) and dog (brown, immediately above cat) form a
  connected brown→red mass with a clear bridge of mixed-class points
  between them — *not* the separated cat/dog clusters Cell C s=4078
  produced. Bird (green, lower-left) is a clean compact lobe with a few
  cat/deer stray points at its top edge; airplane (blue, bottom-centre)
  has 1–2 ship-yellow stragglers at its right edge but is otherwise
  well-isolated. Frog (pink), horse (grey, top-centre), automobile
  (orange, upper-right), truck (cyan, right), deer (purple, far-left),
  ship (yellow, lower-right) are all tight, well-separated lobes. So
  the long tail produced **tighter intra-class density** than Cell B —
  every cluster is visibly more compact than iter004's lobes — but it
  did **not** induce the cat↔dog separation that AutoAugment did in
  Cell C. This is the structural signature of A6 vs A1: longer cosine
  refines the existing decision boundaries; AutoAugment *moves* them.
- **Grad-CAM grid** (`figs/iter_007/cam.png`): **8/8 correctly
  classified** — the cleanest grid this loop, no failures (ship ✓,
  frog ✓, airplane ✓ [Concorde], automobile ✓, frog ✓, cat ✓, ship ✓,
  frog ✓; n=8 caveat with 3× frog and 2× ship skews — the same sampling
  artefact iter002/iter008 hit). Heatmaps are sharply object-centred
  with bright red cores: frog body, Concorde fuselage, car chassis,
  cat torso, ship superstructure. Notably, **the canonical
  ship→automobile miss that reproduced across iter002 and iter008 does
  NOT appear here** — both ship samples in this grid classify
  correctly with attention on the hull-and-superstructure rather than
  fixating on dark hull alone. The 8/8 grid is a small-sample
  observation (n=8, this is a positive-side draw) but is consistent
  with the +0.5 pp per-class lift on ship/automobile/truck.

## 6. Verdict
**Partial.** Δ vs current best (Cell C 2-seed mean 0.9524) = **+0.07 pp
(best) / −0.02 pp (final)** — sits inside Noise band against the leader,
but the *real* comparison for an A6 single-axis swap is the parent **Cell
B s=42**: Δ = **+0.54 pp (best) / +0.47 pp (final)** which lands right
at the Success/Partial boundary (≥ +0.5 pp). Because the leader is Cell
C (different aug recipe) and we have not yet run a long-train+autoaug
twin, the absolute crowning gain is within noise; per program.md
§Verdict criteria, "Partial = mechanism fires but Δ small (0 to +0.5 pp)
[vs current best]" is the strict reading. Mechanism evidence is solid:
cosine's tail is genuinely productive past ep59, train-test gap actually
*tightens*, no NaN/divergence, no overfit slope. Importantly, this run
**clears stretch ≥ 0.95** on a single seed.

## 7. Decision
Keep — Cell F is anchored at single-seed best=0.9531 / final=0.9522.
This validates A6 as a **productive axis** on the Cell-B recipe (unlike
A2 and A3 which were Failures vs B). Two implications:
1. Cell F is now a *legitimate parent candidate* for downstream cells:
   if any future iter wants a "stronger Cell B", `epochs=100` is the
   default knob to inherit, not `epochs=60`.
2. The currently-uncovered combination **AutoAugment + 100 ep** (Cell C
   recipe + Cell F's epochs swap) is the most promising single-axis
   move left in the matrix — Cell C already extracts +0.42 pp over B at
   60 ep, and Cell F shows A6 is worth +0.54 pp on top of B; if the two
   gains compose even partially, an autoaug-100-ep run could push past
   0.955 single-seed. Do NOT crown Cell F as winner; it sits inside
   Noise of Cell C 2-seed mean. Crowning still gated on iter009 (Cell B
   s=4078) analysis to harden the C−B comparison.

## 8. Next hypothesis
The next single change is **Cell C × A6=100** — clone
`configs/ablation/iter002_autoaug.yaml`, swap `epochs: 60 → 100`,
everything else identical (autoaug, sgd 0.1 mom 0.9 nesterov wd 5e-4,
cosine, seed=42). Prior: "lands at 0.955 ± 0.4 pp — autoaug + long
cosine compose at least partially". Falsifier: test_acc < 0.951 would
mean autoaug already saturates the schedule at 60 ep and longer training
overfits the regularizer; test_acc ≥ 0.957 would crown it as the new
provisional leader pending a 2-seed replay; |Δ vs iter002| < 0.3 pp
would mean A6 and A1 don't compose on this backbone (A6's gain on B
came from B being unsaturated, but C is already at the cosine limit).
NOTE: also flag pending iter009 (Cell B s=4078) analysis as the
prerequisite for crowning any phase-1 winner; that run has already
completed per `state/iterations.tsv`.
