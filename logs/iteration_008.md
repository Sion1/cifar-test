# Iteration 008 — iter008_autoaug_s4078 (Cell C 2-seed replay, seed=4078)
Date: 2026-04-27 10:17–10:39 | GPU: 0 | Duration: ~22 min wall (≈19.1 min net train, 60 ep × 19.1 s)

## 1. Hypothesis
Single-axis swap of `seed: 42 → 4078` on the iter002 Cell C recipe (autoaugment,
sgd 0.1 mom 0.9 nesterov wd 5e-4, cosine, 60 ep). This is the **hardening
replay** required by `program.md`'s "After phase 1: pick top 2, run with
seed=4078, report 2-seed mean." Because iter004 Cell B (standard aug) ran
+0.42 pp under iter002 Cell C, the C−B gap currently sits inside the seed-noise
band; the only way to know whether AutoAugment is *really* the better aug or
whether C just happened to draw a lucky seed is a paired seed=4078 replay of
both cells. This iter takes the **first half** of that pair (Cell C). The
prior is "lands at 0.952 ± 0.4 pp on the new seed" — meaning Cell C is
robust to seed change.

## 2. Falsification criterion
Refuted if (a) test_acc < 0.945 — would confirm that iter002's 0.9519 was
a high-tail draw and the C-side seed-variance is wider than assumed; or
(b) test_acc > 0.957 — would suggest seed=42 was a low-tail draw and the
true Cell C mean is closer to 0.953; or (c) the run diverges/NaNs at high
lr — Bug. The "successful confirmation" outcome is |Δ vs iter002| < 0.5
pp, i.e., Cell C reproduces inside the seed-noise band.

## 3. Changes made
Cloned `configs/ablation/autoaugment.yaml` →
`configs/ablation/iter008_autoaug_s4078.yaml`, single-axis edit (seed):

```diff
- exp_name: cifar10_resnet34_autoaug
+ exp_name: cifar10_iter008_autoaug_s4078
- seed: 42
+ seed: 4078
  data:
    augmentation: autoaugment
  training:
    optimizer: sgd
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
    nesterov: true
    scheduler: cosine
    epochs: 60
```

No code changes. Launched via
`bash run_experiment.sh configs/ablation/iter008_autoaug_s4078.yaml 8`
(GPU 0 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell B (iter004, std s=42) | Cell C s=42 (iter002) | Cell C s=4078 (this run) | Δ vs C s=42 | Δ vs B s=42 | Δ vs A     |
|-------------------|------------------|----------------------------|------------------------|---------------------------|-------------|-------------|------------|
| test_acc (final)  | 0.8812           | 0.9475                     | 0.9519                 | **0.9529**                | **+0.0010** | +0.0054     | +0.0717    |
| test_acc (best)   | 0.8828 @ ep55    | 0.9477 @ ep58              | 0.9519 @ ep59          | **0.9529 @ ep58**         | **+0.0010** | +0.0052     | +0.0701    |
| test_loss (final) | 0.4469           | 0.2138                     | 0.1630                 | **0.1551**                | −0.0079     | −0.0587     | −0.2918    |
| train_acc (final) | 1.0000           | 0.9991                     | 0.9642                 | **0.9630**                | −0.0012     | −0.0361     | −0.0370    |
| train–test gap    | 0.1188           | 0.0516                     | 0.0123                 | **0.0101**                | −0.0022     | −0.0415     | −0.1087    |
| best_epoch        | 55               | 58                         | 59                     | 58                        | —           | —           | —          |
| epochs            | 60               | 60                         | 60                     | 60                        | —           | —           | —          |

Run dir: `runs/cifar10_iter008_autoaug_s4078/`. The trajectory tracks
iter002 closely throughout: ep0=0.192 (vs iter002 ~0.18–0.20 range), ep5=0.609,
ep20=0.828, ep30=0.867, ep45=0.929, ep55=0.950, ep58=0.9529, ep59=0.9529.
Best == final-epoch, exactly like iter002 — the cosine tail is still
extracting micro-gains right up to the very last epoch with no overfitting
plateau. The Δ from iter002 to iter008 is **+0.10 pp** — well inside both
the Noise band (|Δ| < 0.3 pp) and the Partial band (0 to +0.5 pp).

**2-seed mean for Cell C** (pending iter009 to mirror this for Cell B):
- Cell C: (0.9519 + 0.9529) / 2 = **0.9524**, σ ≈ 0.0007 (effectively the
  measurement floor — Cell C is **highly seed-stable** at this recipe).
- The C-side seed variance is ~0.10 pp peak-to-peak, far smaller than the
  +0.42 pp C-B single-seed gap from iter002 vs iter004 — so the C-B gap
  is **probably** real, but final hardening waits on iter009 (Cell B
  seed=4078) to compute the matched B-side variance.

The train-test gap actually *tightens* slightly (1.01 pp vs iter002's
1.23 pp) — a coincidental seed-level drift, not a structural change;
test_loss likewise drops 0.163 → 0.155 (−0.008) consistent with marginally
better-calibrated final predictions on this seed.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_008/per_class.csv`): spread = **0.886
  (cat) → 0.985 (automobile) = 0.099** — slightly *tighter* than iter002's
  0.105 spread. Class-by-class numbers: airplane=0.957, automobile=0.985,
  bird=0.943, cat=0.886, deer=0.958, dog=0.914, frog=0.969, horse=0.975,
  ship=0.976, truck=0.966. Deltas vs iter002 (Cell C s=42): airplane
  +0.4, **automobile +0.1**, bird −0.3, **cat +0.7**, deer −0.3, dog
  −0.8, frog −0.3, **horse +1.1**, ship +0.6, truck −0.2. Net = +0.10 pp.
  The single biggest movement (horse +1.1 pp) is well below the ±2 pp
  threshold that would have signalled a structural shift in the cat↔dog
  or bird↔airplane confusion budget — every per-class delta sits inside
  the seed-noise band, with no class swapping rank-order between strong
  and weak. cat remains the worst class on both seeds (0.879 → 0.886;
  −2.4 pp wider on bird than vehicle classes is *flat* across seeds).
  This is the textbook "uniform noise floor" signature.
- **t-SNE** (`figs/iter_008/tsne.png`): **9 visible lobes** — actually
  one *more* than iter002's 8. Cat (red, centre-left) and dog (brown,
  immediately below cat) are now visibly **separate clusters** with
  only a thin bridge of mixed-class points between them, rather than
  iter002's single fused brown→red mammal blob. The remaining structural
  fingerprint matches iter002: bird↔airplane still form a connected
  green↔blue mass with a clear bridge in the centre (the persistent
  sky-background co-occurrence), with a few stray green points landing
  inside the deer (purple) cluster at the top. Frog (pink, far-left),
  horse (grey, lower-centre), ship (yellow, right), truck (cyan,
  lower-right), automobile (orange, bottom) are all tight, well-isolated
  lobes. The seed flip has actually produced a marginally *cleaner*
  representation geometry — consistent with the small +0.10 pp aggregate
  gain and the cat +0.7 pp / dog −0.8 pp swap (cat now slightly easier,
  dog slightly harder, but the *separation* between them improved).
- **Grad-CAM grid** (`figs/iter_008/cam.png`): **7/8 correctly
  classified** (frog, airplane, automobile, frog, cat, ship, frog +
  ship→automobile miss). The miss is **the same failure mode iter002
  showed** — ship → automobile, with attention fixating on the dark
  hull/superstructure rather than masts or sails — so the qualitative
  attention pattern is essentially indistinguishable from iter002:
  sharp object-centred heatmaps with bright red cores on every sample
  (frog body, Concorde fuselage, car chassis, frog body, cat torso,
  ship hull, frog body), no background-shortcut pathology. n=8 caveat
  applies (3× frog skews — same as iter002 / iter006). The persistence
  of the ship→automobile miss across both seeds is informative: it's a
  property of the AutoAugment+ResNet-34 recipe at this dataset size,
  not a seed-specific anomaly, and identifies the most plausible single
  source of remaining headroom (better ship-vs-vehicle separation).

## 6. Verdict
**Noise (= successful hardening).** |Δ vs iter002 (current Cell C
single-seed best)| = +0.10 pp, which lands *inside* the explicit Noise
band (|Δacc| < 0.3 pp). Per program.md §Verdict criteria, Noise is the
ideal outcome for a 2-seed replay whose explicit purpose is *robustness
verification*: we wanted to confirm Cell C reproduces under a different
seed, and it does, with a peak-to-peak spread of just 0.10 pp. The
cell is **anchored** at a 2-seed mean of **0.9524**. No NaN, no
divergence; mechanism (cosine annealing + AutoAugment regularization)
fires exactly as in iter002. Note: this is a **good** Noise — calling it
Failure or Partial would misrepresent that the run *succeeded at its
stated goal* (replication), even though the absolute acc didn't move.

## 7. Decision
Keep — Cell C is now **hardened on the C side** at 2-seed mean 0.9524,
σ ≈ 0.0007. **Crowning is still gated** on iter009 (Cell B seed=4078)
finishing — only the matched B-side seed=4078 number can tell us
whether the +0.42 pp C-B gap shrinks, stays, or widens once both cells
are on the same seed-pair. Provisional leader remains Cell C; iter009 is
the deciding piece. Do not propagate iter008 to a downstream Cell — its
purpose is data quality, not a new parent recipe.

## 8. Next hypothesis
The next single change is **iter009 — Cell B 2-seed replay (seed=4078)**:
clone `configs/ablation/iter004_std.yaml`, swap `seed: 42 → 4078`,
everything else identical (standard aug, sgd 0.1 mom 0.9 nesterov wd
5e-4, cosine, 60 ep). Prior: "lands at 0.948 ± 0.4 pp on the new seed".
Falsifier: |Δ vs iter004| > 0.5 pp would mean Cell B has unexpectedly
high seed variance and the C-B comparison cannot yet be hardened
without a third seed; |Δ vs iter004| < 0.3 pp + the resulting 2-seed B
mean < the 2-seed C mean of 0.9524 by ≥ 0.3 pp would crown Cell C as
the phase-1 winner; reversed sign (B > C on the second seed) would
turn the +0.42 pp single-seed gap into a likely artefact and prompt a
3rd-seed tiebreak. NOTE: iter009 is already running per
`state/iterations.tsv` (GPU 3, completed 10:41:38) — analysis is the
next loop tick's responsibility, not this iter's launch step.
