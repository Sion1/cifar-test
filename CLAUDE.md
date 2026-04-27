# Working Memory · template

> Long-term scratchpad the framework reloads on every iteration. The agent
> appends here as it learns; you (human) edit it whenever you want to inject
> guidance, lock a finding, or correct a mistake.
>
> Read `program.md` first — that's the immutable contract.

---

## Baseline numbers (LOCKED — don't edit after they're recorded)

Fill in after the first run finishes. For the CIFAR-10 demo:

```
Pure baseline (cell A — augmentation=none, sgd 0.1, cosine, 60 ep, seed=42):
- test_acc:   0.8812 (final, ep59) / 0.8828 (best, ep55)
- test_loss:  0.4469 (final)
- run_dir:    runs/cifar10_iter003_bare/
```

Reference: cell A is the floor. Anything in cells B-F should beat this.

---

## Module catalog (mirror of program.md — DO NOT edit, reference only)

| Symbol | Axis | Values |
|---|---|---|
| A1 | data augmentation       | none / standard / autoaugment |
| A2 | optimizer               | sgd / adamw / adam |
| A3 | LR schedule             | cosine / multistep / none |
| A4 | base learning rate      | sweep |
| A5 | weight decay            | sweep |
| A6 | epochs                  | 30 / 60 / 100 |

---

## Ablation matrix progress (UPDATE as cells fill)

| Cell | A1 | A2 | A3 | Best iter# | acc | Verdict | 2-seed mean |
|---|---|---|---|---|---|---|---|
| **A** bare | none      | sgd   | cosine    | 003 | 0.8828 | baseline (1-seed, floor) | TBD |
| **B** +std | standard  | sgd   | cosine    | 004 | 0.9477 | partial (1-seed) → 009 s=4078 = 0.9438, **2-seed mean = 0.9458** (σ ≈ 0.0028) | **0.9458** |
| **C** +AA  | autoaug   | sgd   | cosine    | 002 | 0.9519 | success (1-seed) → 008 s=4078 = 0.9529, **2-seed mean = 0.9524** (σ ≈ 0.0007); CROWNED phase-1 leader | **0.9524** |
| **D** AdamW| standard  | adamw | cosine    | 005 | 0.9354 | failure (1-seed, −1.23 pp vs B) | TBD |
| **E** ms   | standard  | sgd   | multistep | 006 | 0.9394 | failure (1-seed, −0.83 pp vs B) | TBD |
| **F** long | standard  | sgd   | cosine    | 007 | 0.9531 | partial (1-seed, +0.54 pp vs B; clears stretch ≥0.95) | TBD |
| **C×A6=100** (compose) | autoaug | sgd | cosine | 010 | **0.9596** | success (1-seed, +0.72 pp vs Cell C 2-seed mean; clears stretch ≥0.95 robustly; gen-gap 1.28 pp) — provisional new leader pending s=4078 replay | TBD |

## Current best (UPDATE only after 2-seed evidence)
- **PHASE-1 WINNER CROWNED: Cell C (AutoAugment) at 2-seed mean = 0.9524
  (σ ≈ 0.0007).** Cell C s=42 = 0.9519 (iter002), Cell C s=4078 = 0.9529
  (iter008), peak-to-peak spread just 0.10 pp. Cell B 2-seed mean = 0.9458
  (σ ≈ 0.0028): Cell B s=42 = 0.9477 (iter004), Cell B s=4078 = 0.9438
  (iter009), peak-to-peak spread 0.39 pp — **~4× wider than Cell C's**.
  **Δ(C − B) 2-seed mean = +0.66 pp** — *larger* than the +0.42 pp
  single-seed gap, and well outside both cells' within-seed spreads;
  AutoAugment is robustly better than standard aug on this backbone, both
  in mean and in seed-stability. Cell A floor LOCKED at **iter 003 =
  0.8828 (best) / 0.8812 (final)**. Augmentation deltas decomposed
  (2-seed-hardened where available): Δ(B − A) = **+0.0630 (mean − A
  best)** — most of the gain comes from the simple crop+flip prior;
  Δ(C − B) = **+0.0066 (2-seed mean)** — AutoAugment's marginal value
  over standard aug is small but seed-robust. **Cell D anchored** at
  iter 005 = **0.9354 (best, ep51) / 0.9350 (final)** — AdamW
  (lr=1e-3, wd=1e-4) with Cell B's recipe loses **−1.23 pp** vs Cell B;
  optimizer-family swap is a Failure on this setup, do **not** propagate
  Cell D as parent for E/F. **Cell E anchored** at iter 006 =
  **0.9394 (best, ep49) / 0.9386 (final)** — multistep `[30, 45], γ=0.1`
  with Cell B's recipe loses **−0.83 pp** vs Cell B; the textbook ladder
  fires (ep30 jump +10.4 pp, ep45 jump +0.6 pp) but cosine's smooth tail
  still wins, so A3 is a Failure axis on this recipe. **Cell F anchored
  (1-seed only)** at iter 007 = **0.9531 (best, ep98) / 0.9522 (final)**
  — Δ vs Cell C 2-seed mean = +0.07 pp (best) / −0.02 pp (final), well
  inside Cell C's spread band; Cell F is *not* unlocked as winner without
  a Cell F s=4078 replay. **Most promising next experiment: Cell C ×
  A6=100** (autoaug + 100 ep) — composes the only two productive single-
  axis moves (A1 standard→autoaug, A6 60→100); if gains compose even
  partially, target window is 0.955–0.958 single-seed. **iter 010
  ran this experiment** at **0.9596 (best, ep96) / 0.9585 (final, ep99)**
  — overshoots the predicted window by +0.16 pp on best, with
  super-additive composition (A6 lifts +0.77 pp under autoaug vs
  +0.54 pp under std-aug, ~143 % efficiency on A6). Δ vs Cell C
  2-seed mean = +0.72 pp (best) / +0.61 pp (final), both ≥ Success
  threshold (+0.5 pp) and outside Cell C's ±0.05 pp 2-seed spread.
  Gen-gap 1.28 pp ≈ Cell C s=42's 1.23 pp (autoaug keeps the long
  tail from drifting), 3.5× tighter than Cell F's 4.77 pp. **Status:
  provisional phase-2 leader pending a Cell C×A6=100 s=4078 replay**;
  Cell C remains the *crowned* phase-1 leader on the 2-seed
  criterion. The most promising next experiment is the **iter 011
  s=4078 replay** of `iter010_autoaug_long.yaml` to harden the +0.72 pp
  gap before crowning.

---

## Documented findings (agent appends here per iter)

<!-- The agent will add `### Iteration NNN — {cell} {short_name}` blocks
     below per `program.md` §5. Do NOT touch this section yourself unless
     you're correcting a mistake; the agent reads back its own past notes
     to build context for the next iteration. -->

### Iteration 000 — framework smoketest (cifar10_resnet34, EPOCHS_OVERRIDE=1)
The first launch ran the standard-aug baseline config but with `EPOCHS_OVERRIDE=1`,
producing test_acc=0.3045 / test_loss=1.871 after a single 17.9 s epoch. The
training loop, data pipeline, optimizer, checkpointing and viz scripts all
executed cleanly, so the framework itself is healthy — but the config's stated
`epochs: 60` was overridden, so this run does **not** fill any cell of the
ablation matrix. Per-class accuracy after 1 epoch is wildly uneven
(cat=0%, bird=0.6%, deer=13.9% vs. dog=55.5%, automobile=52.4%, frog=47.6%),
t-SNE shows only a vehicle-vs-animal split with no per-class clusters, and
Grad-CAM blobs are object-centred but loose and occasionally fixate on
background bands — all expected for an under-trained model. Lesson: iter 1
must launch the **real Cell A bare baseline** (`configs/ablation/no_aug.yaml`,
augmentation=none, 60 epochs, no `EPOCHS_OVERRIDE`) to actually establish the
floor. Verdict: **Noise** (smoketest, not an ablation point).

### Iteration 010 — Cell C × A6=100 compose (cifar10_iter010_autoaug_long, full 100 ep)
Single-axis delta from `iter002_autoaug.yaml` — `training.epochs:
60 → 100`, everything else identical (autoaugment, sgd 0.1 mom 0.9
nesterov wd 5e-4, cosine, seed=42). The hypothesis was that the only
two productive single-axis moves on the catalog (A1 standard→autoaug
worth +0.66 pp 2-seed mean, and A6 60→100 worth +0.54 pp 1-seed on
Cell B) would compose at least partially on the autoaug recipe; the
falsifier was test_acc < 0.951 (autoaug saturates the schedule). The
full 100-epoch run reached **test_acc=0.9596 (best, ep96) / 0.9585
(final, ep99) / test_loss=0.1445**, with train_acc=0.9713 /
train_loss=0.083 — a **gen-gap of 1.28 pp**, comparable to Cell C
s=42's 1.23 pp / s=4078's 1.01 pp and **3.5× tighter than Cell F's
4.77 pp**: AutoAugment keeps the long-train run from drifting into
the memorize regime that Cell F just barely resisted. Trajectory
walks the Cell C shape stretched 1.67×: ep0=0.21, ep20=0.835,
ep40=0.884, ep55=0.914, ep70=0.924, ep80=0.941, **ep85=0.951 (clears
stretch ≥0.95)**, ep90=0.954, ep95=0.959, **ep96=0.9596 (best)**,
ep99=0.9585. The last 10 epochs sit in a tight 0.953–0.960 band
(peak-to-peak 0.6 pp) — the cosine tail is *still* extracting
micro-gains at ep90+ on the autoaug recipe, no saturation plateau.
**Decomposition**: Δ vs Cell C s=42 (iter002, 0.9519) = **+0.77 pp
(best) / +0.66 pp (final)**; Δ vs Cell F (iter007, 0.9531) =
**+0.65 pp (best) / +0.63 pp (final)**; Δ vs Cell C 2-seed mean
(0.9524) = **+0.72 pp (best) / +0.61 pp (final)** — both axes
contribute, ≥ Success threshold (+0.5 pp) on every comparator, and
outside Cell C's ±0.05 pp 2-seed spread. **Super-additive
composition**: A6's gain under autoaug (+0.77 pp Cell C @60 → Cell
C×A6=100) is ~143 % of A6's gain under std-aug (+0.54 pp Cell B →
Cell F), so far from "autoaug already saturates the schedule" the
inverse holds — autoaug *enables* the long tail to keep working.
Compose-efficiency (A1 + A6 vs predicted sum on Cell B baseline):
0.9477 + 0.0042 + 0.0054 = 0.9573 predicted, 0.9596 actual →
+0.23 pp super-additive. Per-class spread = 0.101 (cat=0.883 →
auto=0.984), comparable to Cell C s=4078's 0.099; per-class deltas
vs Cell B s=42 sum to **+1.19 pp** and decompose: bird **+3.3**
(largest single gain — closes Cell C's bird gap and improves further),
dog **+2.3** (the cat↔dog axis finally moves, recovering from 0.92 →
0.943), ship **+1.9**, deer **+1.2**, automobile **+1.1**, frog
**+0.9**, airplane **+0.7**, horse **+0.6**, cat −0.1, truck 0.0 —
broadly distributed lift, 7/10 classes gain ≥0.6 pp, no class
regresses. Unlike Cell F's bird-and-vehicle-only lift, A6-on-autoaug
*moves* the cat↔dog axis Cell F couldn't touch. t-SNE shows **all 10
classes resolved as named lobes** — the cleanest topology of any
cell so far; cat (red, upper-right) and dog (brown, centre-right)
are now adjacent but **visibly separated** with a thin mixed-points
band rather than the fused brown→red mass that defines Cell B and
Cell F (and that even Cell C s=42 only partially resolved). The
bird↔airplane sky-background bleed visible in Cell C s=4078 has
*closed* on this run; every class has its own mass-centre. Grad-CAM
**8/8 correct** with the same 3× frog + 2× ship deterministic
sample-skew; both ship samples classify correctly (the canonical
ship→automobile miss does NOT reproduce, matching iter007/iter009);
heatmaps slightly tighter than Cell C s=4078's. Verdict: **Success
(1-seed)** — Δ ≥ +0.5 pp on every comparator, mechanism evidence is
clean (super-additive A6 lift, no overfit plateau, t-SNE topology
fully resolved). Status mirrors iter002: **provisional phase-2
leader, crowning gated on a Cell C×A6=100 s=4078 replay** before
`CLAUDE.md`'s "Current best" is updated past Cell C. Lesson: **A1
and A6 are not just both productive, they are super-additively
productive on this backbone** — autoaug does not saturate the cosine
schedule, it *creates* schedule headroom by keeping the regularizer
active through ep90+. This rules Cell F out as a downstream — Cell
C×A6=100 dominates F strictly (same A6, better A1, 3.5× tighter
gen-gap). Continue priorities: iter 011 s=4078 replay of
`iter010_autoaug_long.yaml` to harden the gap; if 2-seed mean ≥
0.956, crown Cell C×A6=100 as the new overall leader. After that,
the remaining productive single-axis moves are A5 (weight decay, not
yet swept) on the new leader recipe, and a Cell F s=4078 replay
(lower priority since C×A6 dominates F).

### Iteration 009 — Cell B 2-seed replay s=4078 (cifar10_iter009_std_s4078, full 60 ep)
Single-axis delta from `iter004_std.yaml` — `seed: 42 → 4078`,
everything else identical (standard aug, sgd 0.1 mom 0.9 nesterov wd
5e-4, cosine, 60 ep). Pair-partner of iter008 (Cell C s=4078); together
the two replays close the program.md "After phase 1: pick top 2, run
with seed=4078, report 2-seed mean" requirement. The full 60-epoch run
reached **test_acc=0.9438 (best, ep58) / 0.9429 (final, ep59) /
test_loss=0.2355**, with train_acc=0.9992 — a **5.63 pp generalization
gap**, very close to Cell B s=42's 5.16 pp (within ±0.5 pp; same
overfit signature). Trajectory tracks iter004 throughout with normal
seed-driven micro-divergence (one −1.4 pp dip at ep35 vs none in
iter004; cosine-tail plateau ep55–59 settles at 0.941–0.944). **Δ vs
iter004 (Cell B s=42) = −0.39 pp (best) / −0.46 pp (final)** — just
outside the formal Noise band (|Δ|<0.3 pp) but well inside the ±0.5 pp
wide-variance threshold. **Pair-partner readout (the actual
scientific output of this run):** Cell B 2-seed mean = **0.9458** with
peak-to-peak spread **0.39 pp**; Cell C 2-seed mean = 0.9524 with
spread 0.10 pp; **Δ(C − B) 2-seed mean = +0.66 pp (best) / +0.72 pp
(final)** — *larger* than the single-seed +0.42 pp gap, and well
outside Cell B's own 0.39-pp seed-spread. Cell B's seed-variance is
**~4× wider than Cell C's** — AutoAugment is more seed-stable on this
backbone, an additional argument in C's favour beyond mean
performance. Per-class spread = 0.105 (cat=0.876 → auto=0.981) vs
iter004's 0.089; per-class deltas vs Cell B s=42 sum to −0.39 pp
and decompose mostly through the mammal triplet — **dog −2.5**
(largest single regression, >50 % of the net loss), **bird +1.7**
(counter-intuitive gain, but at cat/dog's expense — see t-SNE), cat
−0.8, frog −1.3, airplane −1.1, truck −1.1, ship +1.0, automobile
+0.8, deer −0.4, horse −0.2. Cell B's seed-driven per-class jitter is
**~2× Cell C's** and concentrated on the cat↔dog↔bird mammal triplet.
t-SNE shows the canonical Cell-B fingerprint reproducing cleanly:
all 10 classes as named lobes, but cat (red, centre-left) and dog
(brown, lower-left) form a connected brown→red mass with a clear
mixed-class bridge, and bird (green, upper-centre) has 5–8 cat-red
and brown-dog stragglers along its lower edge — these visible
feature-space leaks into bird are the structural correlate of bird's
+1.7 pp gain (bird is *receiving* misclassified mammals from a
less-separated cat/dog region, inflating its diagonal at cat/dog's
expense). The seed only re-routes a few hundred mammal points along
the existing bridge — not a topology change. Grad-CAM **8/8 correct**
(n=8 caveat, 3× frog + 2× ship sampling skew); the canonical ship→
automobile miss that reproduced across iter002/iter008 does NOT
appear (matching iter007's observation that this miss is sample-
dependent rather than recipe-locked). Verdict: **Noise (hardening
successful)** — |Δ|=0.39 pp is just outside the formal Noise band,
but the run's *role* is hardening (a seed replay is not an
expected-positive hypothesis), and that role is fully achieved.
**This run unlocks the phase-1 crowning: Cell C is the winner at
2-seed mean 0.9524, beating Cell B's 0.9458 by +0.66 pp with margin
beyond both within-cell spreads.** Lesson: Cell B is the less
stable cell — for any future "stronger Cell B" parent, cosine +
standard aug carries seed-variance that AutoAugment damps out;
this is a structural argument for inheriting the AutoAugment
recipe rather than the standard one in downstream cells. Continue
priorities: **Cell C × A6=100** (autoaug + 100 ep) is the next
single-axis experiment — composes the only two productive moves on
the catalog (A1 standard→autoaug, A6 60→100); if gains compose
partially, target window 0.955–0.958. Cell F crowning would also
require an s=4078 replay (currently 1-seed only).

### Iteration 007 — Cell F long-train (cifar10_iter007_long, full 100 ep)
Single-axis delta from `iter004_std.yaml` — `training.epochs: 60 → 100`,
everything else identical (standard aug, sgd 0.1 mom 0.9 nesterov wd
5e-4, cosine, seed=42). Cosine T_max scales with the new budget so it's
a clean A6 test. The full 100-epoch run reached **test_acc=0.9531 (best,
ep98) / 0.9522 (final, ep99) / test_loss=0.2012**, train_acc=0.9999 /
train_loss=1.5e-3 — a **4.77 pp gen-gap**, slightly *tighter* than Cell
B's 5.16 pp at 60 ep. Trajectory walks the Cell B shape stretched
1.67× along time: ep0=0.271, ep10=0.797, ep30=0.855, ep50=0.881,
ep60=0.885, ep80=0.941, ep90=0.951, ep98=0.9531. The first 60 ep deliver
+61.4 pp; the extra 40 ep at lr ≈ 0.07→0 deliver +6.8 pp — marginal
value drops ~30× into the tail but stays positive throughout, no overfit
plateau. Δ(F − B) = **+0.54 pp (best) / +0.47 pp (final)** — sits right
at the Success/Partial boundary; Δ(F − Cell C 2-seed mean 0.9524) =
**+0.07 pp (best) / −0.02 pp (final)** — well inside the Noise band
against the leader. Per program.md §Verdict, leader-relative reading
governs: **Partial**. Per-class spread = 0.096 (cat=0.887 → auto=0.983),
between Cell B's 0.089 and Cell C s=4078's 0.099. Per-class deltas vs
Cell B decompose unevenly: **bird +2.4** (recovers 83 % of the −2.9 pp
gap Cell B had vs Cell C), automobile +1.0, truck +0.8, ship +0.5,
deer/cat/dog/frog all +0.2–+0.4, airplane 0.0, **horse −0.5** (sole
regression, within ±1 pp seed-noise). The long-tail gain concentrates
in bird + vehicles — *not* the cat/dog axis that bottlenecks both B and
C. t-SNE shows all 10 classes as named lobes with **tighter intra-class
density than Cell B**, but the canonical brown→red cat↔dog fusion
**persists** — the long cosine tail refines existing decision boundaries
rather than moving them; AutoAugment in Cell C *moves* the cat/dog
boundary, the long tail does not. Grad-CAM 8/8 correct (n=8 caveat,
3× frog + 2× ship skews) — the cleanest grid this loop. Notably the
ship→automobile miss that reproduced across iter002 and iter008 does
NOT reproduce here (both ship samples classify correctly with attention
on hull+superstructure rather than dark hull alone), small-sample but
consistent with the +0.5 pp ship lift. Verdict: **Partial** (Δ vs
current best = +0.07 pp, mechanism fires cleanly, run clears stretch
≥0.95). Lesson: A6 is a **productive axis** on the Cell-B recipe (unlike
A2/A3 which were Failures) — the schedule is genuinely not saturated at
ep59. Two implications for next moves: (1) Cell F is now a legitimate
parent candidate — any future "stronger Cell B" should inherit
`epochs=100`. (2) The **AutoAugment + 100 ep** combination (Cell C
recipe + Cell F's epoch swap) is the most promising single-axis move
left in the matrix at this granularity; if A6 and A1 gains compose even
partially, an autoaug-100-ep run could push past 0.955 single-seed.
Continue priorities: iter 009 (Cell B s=4078) analysis is the
prerequisite for crowning any phase-1 winner; after that, autoaug+100ep
is the natural next experiment.

### Iteration 008 — Cell C 2-seed replay s=4078 (cifar10_iter008_autoaug_s4078, full 60 ep)
Single-axis delta from `iter002_autoaug.yaml` — `seed: 42 → 4078`,
everything else identical (autoaug, sgd 0.1 mom 0.9 nesterov wd 5e-4,
cosine, 60 ep). The full 60-epoch run reached **test_acc=0.9529 (final,
ep59) / 0.9529 (best, ep58) / test_loss=0.1551**, with train_acc=0.9630
/ train_loss=0.1102 — a **1.01 pp generalization gap**, slightly *tighter*
than iter002's 1.23 pp. Trajectory tracks iter002 throughout (ep5=0.609,
ep20=0.828, ep30=0.867, ep45=0.929, ep55=0.950, ep58=0.9529) — cosine
tail still extracting micro-gains right up to the last epoch with no
overfit plateau. **Δ vs iter002 (Cell C s=42) = +0.10 pp** — well inside
the Noise band (|Δ|<0.3 pp); **Cell C 2-seed mean = (0.9519+0.9529)/2 =
0.9524, peak-to-peak spread just 0.10 pp** (σ ≈ 0.0007). This is a
**high-quality replication**: Cell C is highly seed-stable and clears
stretch ≥ 0.95 on both seeds. Per-class spread = 0.099 (cat=0.886 worst
→ automobile=0.985 best), even *tighter* than iter002's 0.105; per-class
deltas vs iter002 are uniform-noise (largest movement: horse +1.1 pp;
cat +0.7 pp, dog −0.8 pp — net cat↔dog confusion budget barely shifts).
t-SNE actually shows **9 visible lobes** (one *more* than iter002's 8)
— cat and dog are now *separate* clusters with only a thin bridge,
rather than iter002's fused brown→red mammal blob; bird↔airplane mass
persists in the centre (the canonical sky-background co-occurrence).
Grad-CAM 8-image grid shows **7/8 correct** with the same single failure
mode as iter002: **ship → automobile** (attention on dark hull instead
of masts) — confirming this miss is a property of the
AutoAugment+ResNet-34 recipe, not a seed-specific anomaly, and pinpoints
the most plausible source of remaining headroom (better ship-vs-vehicle
separation). Verdict: **Noise** = successful hardening — replication
inside the explicit |Δ|<0.3 pp band is the *ideal* outcome for this kind
of run, even though Noise is normally a "did nothing" label. Lesson:
the C-side seed-variance is ≪ the +0.42 pp single-seed C−B gap, so the
gap is likely real, but **crowning is gated** on iter 009 (Cell B s=4078,
already completed, awaiting analysis) — only the matched B-side number
hardens the cross-cell comparison. Continue priorities: iter 009
analysis (Cell B s=4078) → if the 2-seed B mean lands ≥ 0.3 pp under
0.9524, crown Cell C; if not, schedule a 3rd-seed tiebreak. iter 007
(Cell F long-train) is also running and should complete this loop.

### Iteration 006 — Cell E multistep (cifar10_iter006_multistep, full 60 ep)
Single-axis delta from `iter004_std.yaml` — `training.scheduler: cosine →
multistep`, `milestones: [30, 45]` (= 50 % / 75 % of 60 ep), γ=0.1 (trainer
default). Standard aug, sgd 0.1 mom 0.9 nesterov wd 5e-4, 60 ep, seed=42 held
fixed. The full 60-epoch run reached **test_acc=0.9386 (final, ep59) /
0.9394 (best, ep49) / test_loss=0.2459**, with train_acc=0.9981 /
train_loss=0.0084 by ep~55 — a **5.95 pp generalization gap**, slightly
wider than Cell B's 5.16 pp. The trajectory is *exactly* the textbook
multistep ladder: noisy ~0.81 plateau through ep0–29 (peak 0.8418 at
ep20, drift back to 0.8155 at ep29), single-epoch jump **+10.4 pp at
ep30** when lr=0.1→0.01 (0.8155 → 0.9191), creep to 0.931 by ep44,
**+0.6 pp jump at ep45** when lr=0.01→0.001 (0.9314 → 0.9373), tight
plateau 0.937–0.939 ep45–59, peak 0.9394 at ep49. Δ(E − B) = **−0.83 pp
(best) / −0.89 pp (final)** — multistep loses to cosine on the Cell-B
recipe; Δ(E − A) = +5.66 pp — still beats the floor; Δ(E − D) = +0.40 pp
— SGD+multistep beats AdamW+cosine, so on this recipe the optimizer
family matters more than the schedule family. Per-class spread = 0.106
(cat=0.872 → auto=0.978), looser than B's 0.089 but tighter than A's
0.189; the −0.83 pp aggregate Δ vs Cell B is **disproportionately
animal-side** — frog (−2.4), dog (−1.4), cat (−1.2), horse (−1.0)
absorb 6.0 pp of the 8.3 pp net loss while vehicles only give up 1.4 pp
(automobile actually *gains* +0.5 pp). This rhymes with Cell D's
animal-side bleed pattern: when SGD+cosine's small-lr tail is replaced —
whether by AdamW or by multistep's two coarse drops — animal-shape
evidence slips first. t-SNE shows the same 8 lobes as Cell B with
looser boundaries: the cat↔dog fuse is present as a continuous
brown→red mass with a thicker mixed-class bridge; airplane↔bird seam in
the centre with cross-class points; small deer↔frog/dog tendril off the
deer cluster. Grad-CAM 8/8 correct (n=8 caveat, 3× frog skews — and
ironically frog is the worst-regressed class on the per-class table)
and visually indistinguishable from Cell B / Cell D — multistep does NOT
induce a different saliency pattern, regression is purely a
generalization-margin problem. Verdict: **Failure** (acc drops > 0.5 pp
on an expected-positive hypothesis vs the parent Cell B). Lesson: on
this Cell-B recipe, A3 is **not** a productive axis to spend more
iterations on at the catalog's current granularity — the only remaining
A3 point (`none`) can only do worse than multistep. **Cell E is a
dead-end parent**: Cell F (long-train, A6=100) should continue to build
on **Cell B (sgd, cosine, 0.9477)**, not on Cell E. Continue priorities:
Cell F (long-train, A6=100), then 2-seed (seed=4078) replay of Cells B
and C to harden the C−B = +0.42 pp gap before crowning a winner.

### Iteration 005 — Cell D AdamW (cifar10_iter005_adamw, full 60 ep)
Single-axis delta from `iter004_std.yaml` — `training.optimizer: sgd →
adamw` with A4/A5 moved into AdamW's catalog brackets (`lr 0.1 → 1.0e-3`,
`wd 5.0e-4 → 1.0e-4`; `momentum`/`nesterov` keys dropped). Standard aug,
cosine, 60 ep, seed=42 held fixed. The full 60-epoch run reached
**test_acc=0.9350 (final, ep59) / 0.9354 (best, ep51) / test_loss=0.4368**,
with train_acc=0.9995 / train_loss=0.0015 by ep~55 — a **6.45 pp
generalization gap**, *wider* than Cell B's 5.16 pp despite identical aug.
The trajectory is unusual: test_acc peaks at ep51 (0.9354) and then
**flattens / drifts down to 0.9350** for the last 9 epochs while
test_loss rises monotonically from ~0.30 (ep20) to 0.44 (ep59) — the model
becomes more confident on memorized errors rather than gaining
generalizable signal. Δ(D − B) = **−1.23 pp (best) / −1.25 pp (final)** —
AdamW with catalog-bracket defaults *loses* to SGD+nesterov on this Cell-B
recipe; Δ(D − A) = +5.26 pp — still beats the floor by a healthy margin.
Per-class spread = 0.108 (cat=0.863 → auto=0.971), looser than B's 0.089
but tighter than A's 0.189; the −1.23 pp aggregate Δ vs Cell B is
**disproportionately animal-side** — cat (−2.1), dog (−3.5), frog (−2.6),
horse (−1.2) absorb 9.4 pp of the 12.3 pp drop while vehicles
(auto/truck/ship) collectively give up only 0.8 pp. t-SNE shows the same
8 lobes as Cell B but with looser boundaries: cat↔dog fuse persists, plus
new bleed at bird↔airplane↔ship and auto↔truck — feature space similarly
*organized* but less *separated*, consistent with the wider gap. Grad-CAM
8/8 correct (n=8 caveat, 3× frog skews) and visually indistinguishable
from Cell B — AdamW does NOT induce a different saliency pattern, the
regression is purely a generalization-margin problem. Verdict:
**Failure** (acc drops > 0.5 pp on an expected-positive hypothesis vs the
parent Cell B). Lesson: on this Cell-B recipe, A2 is **not** a productive
axis to spend more iterations on — the only remaining AdamW catalog point
is lr=5e-4 which is unlikely to recover 1.23 pp on its own. **Cell D is a
dead-end parent**: downstream cells E/F should continue to build on
**Cell B (sgd, 0.9477)**, not Cell D. Continue priorities: Cell E
(multistep, A3), Cell F (long-train, A6=100), then 2-seed (seed=4078)
replay of the top-2 to harden the C−B = +0.42 pp gap.

### Iteration 004 — Cell B standard aug (cifar10_iter004_std, full 60 ep)
Single-axis delta from `iter003_bare.yaml` — `data.augmentation: none →
standard` (RandomCrop 32 + HorizontalFlip), everything else fixed (sgd
lr=0.1 mom=0.9 wd=5e-4 nesterov, cosine, 60 ep, seed=42). The full 60-epoch
run reached **test_acc=0.9475 (final, ep59) / 0.9477 (best, ep58) /
test_loss=0.2138**, with train_acc=0.9991 by ep~56 and train_loss ≈ 4.4e-3
— a *mild* overfit profile, **5.16 pp gap** that sits cleanly between Cell
A's 11.88 pp and Cell C's 1.23 pp. This anchors Cell B and yields the first
clean **decomposition of the augmentation axis**: Δ(B − A) = **+6.49 pp
(best)** — standard aug delivers ~94% of what's available; Δ(C − B) = **+0.42
pp (best)** — AutoAugment's marginal value over crop+flip is small and
within seed-noise band, so the iter002 single-seed lead may not survive a
2-seed replay (now scheduled as required before any winner is crowned).
Per-class spread = 0.089 (cat=0.884 → auto=0.973), the *tightest* of the
three cells; the +0.42 pp aggregate gap to Cell C is concentrated almost
entirely in bird (−2.9 pp) and automobile (−1.1 pp), while Cell B actually
*beats* Cell C on cat (+0.5 pp) and airplane (+0.9 pp). t-SNE shows 8 clean
lobes (matches Cell C, beats Cell A's 6); cat+dog still partially fused on
the left but as a brown-blob-with-red-sub-cluster rather than a single
mammal mass. Grad-CAM: 8/8 correct (a clean grid for the first time this
loop, n=8 caveat); heatmaps sharp and object-centred — tighter than Cell
A's broad whole-object glow, slightly less contour-faithful than Cell C's
edge-tracing maps. Verdict: **Partial** (Δ vs current best C = −0.42 pp,
within the 0.5 pp Partial band but outside Noise's 0.3 pp; the cell's
structural role as the standard-aug anchor of the matrix is fully achieved).
Lesson: AutoAugment's contribution on top of crop+flip is narrow (mostly
bird) — for downstream cells D/E/F that build on `augmentation=standard`,
**0.9477 is the threshold to beat**, not 0.9519.

### Iteration 003 — Cell A bare baseline (cifar10_iter003_bare, full 60 ep)
Single-axis delta from `configs/cifar10_resnet34.yaml` — `data.augmentation:
standard → none`, everything else fixed (sgd lr=0.1 mom=0.9 wd=5e-4 nesterov,
cosine, 60 ep, seed=42). The full 60-epoch run reached **test_acc=0.8812
(final, ep59) / 0.8828 (best, ep55) / test_loss=0.4469**, with train_acc=1.0
hit by ep~50 and train_loss collapsing to 1.4e-3 — a textbook overfit
profile, **11.9 pp generalization gap** vs. Cell C's 1.2 pp. Test acc
plateaus across ep50–59 (0.881–0.883), so cosine has fully converged at this
regularization level. This LOCKS the program's floor and yields the first
clean isolated augmentation delta: **Δ(C − A) = +0.0691 (best) / +0.0707
(final)** — AutoAugment is delivering ~7 pp on top of no-aug ResNet-34.
Per-class spread widens to 0.770–0.959 (Cell C: 0.879–0.984); the loss
concentrates in the visually-ambiguous animal classes — `bird=0.815`
(−0.131), `dog=0.811` (−0.111), `cat=0.770` (−0.109) — while vehicles
lose only 2.5–5.3 pp. t-SNE collapses to ~6 clean lobes (vs. C's 8); cat+dog
form a single fused mammal blob, airplane bleeds into ship, and multiple
"bridge" tendrils between classes show that without aug the feature space is
memorized rather than invariant. Grad-CAM: 7/8 correct (frog→deer the lone
miss); heatmaps remain object-centred but are visibly **blobbier and broader**
than iter002's tight contour-following maps — coarse whole-object glow rather
than discriminative parts. No background-shortcut pathology. Verdict:
**baseline** (the cell's purpose is to be beaten; not labelled Success since
acc didn't rise vs best, but it's exactly the floor the matrix demands).
Caveat: still single-seed; 2-seed hardening only after Cell B lands so we
can pair the two replays.

### Iteration 002 — Cell C +autoaug (cifar10_iter002_autoaug, full 60 ep)
Single-axis delta from `configs/cifar10_resnet34.yaml` — `data.augmentation:
standard → autoaugment`, everything else fixed (sgd lr=0.1 mom=0.9 wd=5e-4
nesterov, cosine, 60 ep, seed=42). The full 60-epoch run reached
**test_acc=0.9519 / test_loss=0.163** at epoch 59 (best == final epoch — cosine
schedule is still extracting gains at the very end, no overfitting plateau).
Train_acc=0.9642 leaves only ~1.2 pp generalization gap, the regularization
signature of strong on-the-fly augmentation. Per-class accuracies are tightly
banded (0.879–0.984) with `cat=0.879` as the lone weak class — the canonical
cat–dog confusion that AutoAugment alleviates but cannot solve at 32×32. t-SNE
shows 8 of 10 classes as their own clusters, with cat+dog fused into one mammal
blob and bird+airplane partially overlapping along their shared sky background
— both consistent with the per-class table. Grad-CAM: 7/8 correctly classified
test samples, attention sharply object-centred (fuselage, frog body, horse
torso, ship hull), a clear advance over iter000's loose maps; the lone miss
(ship→automobile) fixates on the dark hull rather than masts. Already meets
the program target ≥0.94 and clears the stretch ≥0.95 on a single seed. **Caveat**:
no Cell A or Cell B run has finished yet, so the *isolated* AutoAugment Δ vs.
standard aug is unquantified — must wait for iter004_std. Verdict: **Success
(1-seed)**. Provisional best; scheduled for 2-seed (seed=4078) replay only
after Cell B lands.

---

## Operating rules per iteration (framework-supplied; keep as-is)

1. Read `state/iterations.tsv` first — what cell are we on, what's pending
2. Read the 3 most recent `logs/iteration_*.md` for context
3. Pick ONE change — single-axis delta, no bundling
4. Use `run_experiment.sh` to launch — never invoke `python3 train.py` directly
5. Don't skip the mandatory visualizations (program.md §Mandatory)
6. Update this file's matrix + findings after each iteration

---

## Tools cheat-sheet

```bash
# Launch one experiment manually:
bash run_experiment.sh configs/ablation/no_aug.yaml 1

# After it finishes — visualize:
python3 scripts/visualize_tsne.py --ckpt runs/cifar10_iter001_no_aug/best.pth \
        --out figs/iter_001/tsne.png
python3 scripts/visualize_cam.py  --ckpt runs/cifar10_iter001_no_aug/best.pth \
        --out figs/iter_001/cam.png

# Regenerate the dashboard webpage:
bash scripts/generate_experiment_tree_web.sh
```
