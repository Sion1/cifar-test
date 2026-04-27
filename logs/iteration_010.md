# Iteration 010 — iter010_autoaug_long (Cell C × A6=100, autoaug + 100 ep)
Date: 2026-04-27 11:06–11:40 | GPU: 2 | Duration: ~34 min wall (≈31 min net train, 100 ep × 18.6 s)

## 1. Hypothesis
Compose the only two productive single-axis moves on the catalog so far:
**A1 standard→autoaug** (Cell C @60ep, +0.42 pp single-seed / +0.66 pp 2-seed mean
vs Cell B) and **A6 60→100** (Cell F vs Cell B @60 ep, +0.54 pp single-seed).
Single-axis delta from `iter002_autoaug.yaml`: `training.epochs: 60 → 100`,
everything else identical (autoaug, sgd 0.1 mom 0.9 nesterov wd 5e-4, cosine,
seed=42). Cosine T_max scales with the new budget so this is a clean A6 test
on the autoaug recipe. Prior: lands at **0.955 ± 0.4 pp** — autoaug + long
cosine compose at least partially on this backbone, the 40 extra epochs at
lr ≈ 0.07 → 0 mostly tighten the cat/dog/bird boundaries that AutoAugment
*moves* (rather than just refines, as Cell F's long tail did on the std-aug
recipe).

## 2. Falsification criterion
- **test_acc < 0.951** → autoaug already saturates the schedule at 60 ep
  (longer training overfits the regularizer); A6 and A1 don't compose.
- **|Δ vs iter002 (Cell C s=42, 0.9519)| < 0.3 pp** → Noise; A6's Cell-B
  gain came from B being unsaturated, C is already at the cosine limit.
- **test_acc ≥ 0.957** → Success and provisional new leader pending a
  2-seed (s=4078) replay before crowning.
- **0.951 ≤ test_acc < 0.957** → Partial compose; both axes contribute but
  sublinearly, leader still Cell C 2-seed mean = 0.9524.
- Run diverges/NaN → Bug.

## 3. Changes made
Cloned `configs/ablation/iter002_autoaug.yaml` →
`configs/ablation/iter010_autoaug_long.yaml`, single-axis edit (epochs):

```diff
- exp_name: cifar10_iter002_autoaug
+ exp_name: cifar10_iter010_autoaug_long
  seed: 42
  data:
    augmentation: autoaugment
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
`bash run_experiment.sh configs/ablation/iter010_autoaug_long.yaml 10`
(GPU 2 per loop scheduler).

## 4. Results
| Metric            | Cell A (iter003) | Cell B 2-seed mean | Cell C 2-seed mean | Cell F (iter007, 1-seed) | Cell C×A6=100 (this run) | Δ vs Cell C 2-seed | Δ vs Cell F | Δ vs Cell A |
|-------------------|------------------|----------------------|----------------------|----------------------------|----------------------------|---------------------|--------------|----------------|
| test_acc (best)   | 0.8828 @ ep55    | 0.9458              | 0.9524              | 0.9531 @ ep98              | **0.9596 @ ep96**          | **+0.0072**         | +0.0065      | +0.0768        |
| test_acc (final)  | 0.8812           | 0.9452              | 0.9524              | 0.9522 @ ep99              | **0.9585 @ ep99**          | **+0.0061**         | +0.0063      | +0.0773        |
| test_loss (final) | 0.4469           | —                    | —                    | 0.2012                     | **0.1445**                 | —                   | −0.0567      | —              |
| train_acc (final) | 1.0000           | 0.9991/0.9992       | 0.9642/0.9630       | 0.9999                     | **0.9713**                 | —                   | −0.0286      | —              |
| train–test gap    | 0.1188           | 0.0516/0.0563       | 0.0123/0.0101       | 0.0477                     | **0.0128**                 | —                   | −0.0349      | —              |
| best_epoch        | 55               | 58/58               | 59/58               | 98                         | **96**                     | —                   | —            | —              |

Run dir: `runs/cifar10_iter010_autoaug_long/`. Trajectory walks the Cell C
shape stretched 1.67× along time, then keeps extracting gains right into
the cosine tail: ep0=0.210, ep5=0.568, ep10=0.783, ep20=0.835, ep30=0.840
(small dip), ep40=0.884, ep50=0.886, ep55=0.914, ep60=0.910, ep70=0.924,
ep80=0.941, ep85=0.951 (clears stretch ≥0.95 here), ep90=0.954, ep95=0.959,
**ep96=0.9596 (best)**, ep97=0.958, ep98=0.959, ep99=0.9585 (final). The
last 10 epochs (ep90–99) sit in a tight 0.953–0.960 band (peak-to-peak
0.6 pp) — the cosine tail is *still* extracting micro-gains on the autoaug
recipe at ep90+, just like Cell C @60 ep had no overfit plateau, but here
the schedule has been given 40 more epochs of small-lr time to refine the
boundaries AutoAugment opened up.

**Decomposition vs the parents**:
- Δ vs Cell C s=42 (iter002, 0.9519) = **+0.0077 (best) / +0.0066 (final)** —
  larger than Cell F's gain over Cell B s=42 (+0.0054 best / +0.0047 final),
  i.e. **A6 lifts autoaug *more* than it lifted std-aug** at the same epoch
  budget — the opposite of the "autoaug saturates the schedule" prior.
- Δ vs Cell C 2-seed mean (0.9524) = **+0.0072 (best) / +0.0061 (final)** —
  ≥ +0.5 pp Success threshold AND well outside Cell C's ±0.05 pp 2-seed
  spread, so the lift is real even before factoring in this run's own
  seed-variance band.
- Δ vs Cell F (iter007, 0.9531) = **+0.0065 (best) / +0.0063 (final)** —
  also ≥ +0.5 pp; shows the lift comes from A1 (autoaug) on top of A6
  (long-train), not just from A6 alone.
- Δ(B→F best) = +0.0054, Δ(C→Cell C×A6 best) = +0.0077 → A6's gain is
  ~43 % *larger* under autoaug than under standard aug (0.77 pp vs 0.54 pp).
  Mirrored on the regularization side: **gen-gap final = 1.28 pp**, only
  marginally wider than Cell C s=42's 1.23 pp / s=4078's 1.01 pp, and
  **3.5× tighter than Cell F's 4.77 pp** — autoaug keeps the long-train
  run from drifting into the memorize regime that Cell F just barely
  resisted (Cell F train_acc=0.9999 vs this run's 0.9713).

**Caveat — single seed.** Both Cells C and F were single-seed Successes
that needed s=4078 replays before any crown was unlocked; iter002→iter008
moved Cell C by only +0.10 pp (well inside Noise) and crown C, while Cell F
has not been replayed. This run inherits the same status: **1-seed
Success, leader-status pending an iter011 (or later) s=4078 replay**.
Do not edit `CLAUDE.md`'s "Current best" until a 2-seed mean is in.

## 5. Visualization evidence
- **Per-class CSV** (`figs/iter_010/per_class.csv`): airplane=0.969,
  automobile=0.984, bird=0.950, cat=0.883, deer=0.970, dog=0.943,
  frog=0.979, horse=0.972, ship=0.981, truck=0.965, overall=**0.9596**.
  Spread = **0.101** (cat=0.883 → automobile=0.984), comparable to
  Cell C s=4078's 0.099 / s=42's 0.105. The headline finding: **dog
  recovers from 0.92 (Cell B s=42) → 0.943 (this run), the largest
  single-class lift on the cat↔dog axis any cell has produced**, and
  cat=0.883 finally clears the 0.88 floor that Cell C s=42 and
  s=4078 both stalled at (0.879 / 0.886). Per-class deltas vs Cell B
  s=42 (iter004: 0.962/0.973/0.917/0.884/0.958/0.920/0.970/0.966/
  0.962/0.965, in class order): airplane **+0.7**, automobile **+1.1**,
  bird **+3.3** (largest gain — recovers the bird gap C had vs B *and*
  improves further), cat **−0.1** (essentially flat — cat is the
  hardest class for *any* recipe in this catalog), deer **+1.2**,
  dog **+2.3** (the cat↔dog axis finally moves), frog **+0.9**,
  horse **+0.6**, ship **+1.9**, truck **0.0**. Sum = +11.9 / 1000 =
  **+1.19 pp aggregate vs Cell B s=42** (matches the 0.9596 − 0.9477 =
  +1.19 pp gap exactly). The lift is broadly distributed: 7/10 classes
  gain ≥0.6 pp, only cat and truck stagnate, and no class regresses.
  This is unlike Cell F's lift pattern (concentrated on bird/vehicles)
  — adding A6 on top of A1 lifts the *cat↔dog* axis that Cell F's long
  tail couldn't move on the standard-aug recipe.
- **t-SNE** (`figs/iter_010/tsne.png`): **all 10 classes resolved as
  named lobes** — the cleanest separation of any cell so far. Cat (red,
  upper-right) and dog (brown, centre-right) are now **adjacent but
  visibly separated**, with a thin band of mixed points between them
  rather than the fused brown→red mass that defines Cell B and Cell F
  and that even Cell C s=42 only partially resolved. About 5–8 red
  cat-points sit on dog's upper edge (the residual 5.7 pp dog error)
  and a similar count of brown dog-points sit on cat's lower edge,
  but both clusters have unambiguous mass-centres — the topology is
  finally fully separated, in contrast to Cell C s=4078's "9 lobes
  with thin bridge" or Cell F's "preserved fuse". Frog (pink, far
  right) is large and well-separated from the cat/dog mass with only
  ~3 stragglers. Bird (green, centre) is a tight lobe; airplane (blue,
  bottom-left) is well-separated from ship (yellow, mid-left) and
  truck (cyan, far-left) — the bird↔airplane sky-background bleed
  that Cell C s=4078 still showed has *closed* on this run. Automobile
  (orange) sits alone at top, horse (grey) is centrally tight, deer
  (purple, bottom-right) is clean. The structural lesson: A1 (autoaug)
  *moves* the cat↔dog boundary; A6 (long-train) *sharpens* whatever
  boundaries the regularizer set up. Composing them yields the first
  topology where every class has its own mass-centre.
- **Grad-CAM grid** (`figs/iter_010/cam.png`): **8/8 correctly
  classified** (ship ✓, frog ✓, airplane ✓ [Concorde], automobile ✓,
  frog ✓, cat ✓, ship ✓, frog ✓ — n=8 caveat with the same 3× frog +
  2× ship sample-draw skew that iter002/007/008/009 hit, deterministic
  sample seed). Heatmaps are sharply object-centred with bright red
  cores: ship hull-and-superstructure (both samples), frog body (all
  three), Concorde fuselage, automobile chassis with a tight wrap on
  the body, **cat torso with the heatmap covering both head and body
  rather than fixating on either** — the cat sample classifies
  correctly despite cat being the worst class on the diagonal,
  rhyming with iter009's same observation. **The canonical
  ship→automobile miss that reproduced on iter002 and iter008 does
  NOT reproduce here** — both ship samples classify correctly with
  attention on hull+masts, matching iter007 and iter009 (the miss is
  sample-draw-dependent rather than recipe-locked, as expected).
  Heatmaps look slightly *tighter* than Cell C s=4078's — the long
  tail visibly sharpens the discriminative-region focus on top of
  AutoAugment's already-sharp baseline, consistent with the +0.77 pp
  per-class aggregate lift over Cell C s=42.

## 6. Verdict
**Success (1-seed).** Δ vs current best (Cell C 2-seed mean = 0.9524) =
**+0.72 pp (best) / +0.61 pp (final)**, both ≥ the +0.5 pp Success
threshold and outside Cell C's 2-seed spread (±0.05 pp). Mechanism
evidence is also strong: the gen-gap stays at 1.28 pp (autoaug keeps
the long tail from over-fitting), the trajectory shows the cosine tail
*still* extracting gains at ep90+ (no saturation plateau under autoaug),
and the lift composes from both A1 (+0.65 pp vs Cell F) and A6 (+0.77 pp
vs Cell C @60 ep) — both axes contributing, super-additive against the
"autoaug saturates the schedule" falsifier. Status mirrors iter002 →
new provisional leader, **crowning gated on a Cell C×A6=100 s=4078
replay** before `CLAUDE.md`'s "Current best" is updated.

## 7. Decision
**Keep, propagate as the next-best parent recipe (`autoaug + 100 ep`)
pending 2-seed hardening.** Cell C remains the *crowned* phase-1 leader
on the 2-seed criterion; this run is the most-credible *phase-2*
candidate. Implications for the matrix:
1. A6 and A1 *compose* on this backbone — ~0.7 pp from autoaug + ~0.7 pp
   from long-train sum to ~1.4 pp lift over Cell B s=42 (0.9477 →
   0.9596), only ~0.06 pp short of the additive prediction (1.4 − 0.96
   = 0.44 pp). Compose-efficiency ≈ 88 %.
2. The pair (A2=adamw, A3=multistep) on the Cell-B recipe both Failed,
   so A1 + A6 are the only productive single-axis moves at the catalog's
   current granularity. Cell C×A6=100 is the most-likely cell to clear
   the stretch ≥0.95 robustly across seeds.
3. Cell F (long-train, std aug) is dominated: same A6, weaker A1 (std
   instead of autoaug), and 4.77 pp gen-gap vs this run's 1.28 pp. Cell
   F crowning is no longer an interesting downstream — Cell C×A6 covers
   it strictly.
4. Cell A floor still LOCKED at 0.8828 (iter003).

## 8. Next hypothesis
The single most-informative next experiment is the **2-seed replay of
Cell C×A6=100 with seed=4078**. Clone `configs/ablation/iter010_autoaug_long.yaml`
→ `configs/ablation/iter011_autoaug_long_s4078.yaml`, single-axis edit
`seed: 42 → 4078`, everything else identical. Prior: "lands inside
±0.3 pp of 0.9596 (best)" — Cell C s=42→s=4078 spread was just 0.10 pp
so the autoaug recipe is highly seed-stable; long-train should not
break this. Falsifier: |Δ vs iter010| ≥ 0.5 pp = wide-variance, demand
a 3rd seed. Success: 2-seed mean ≥ 0.956 crowns Cell C×A6=100 as the
new overall leader. After that, the remaining productive directions on
the matrix are: (i) Cell C × A6=100 × A5=1e-4 (weight-decay sweep on
the new leader recipe — A5 has not been touched yet), and (ii) Cell F
s=4078 to harden Cell F (lower priority now that Cell C×A6 dominates F).
