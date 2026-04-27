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
| **B** +std | standard  | sgd   | cosine    | 004 | 0.9477 | partial (1-seed, anchor) | TBD |
| **C** +AA  | autoaug   | sgd   | cosine    | 002 | 0.9519 | success (1-seed) | TBD |
| **D** AdamW| standard  | adamw | cosine    | 005 | 0.9354 | failure (1-seed, −1.23 pp vs B) | TBD |
| **E** ms   | standard  | sgd   | multistep | 006 | 0.9394 | failure (1-seed, −0.83 pp vs B) | TBD |
| **F** long | standard  | sgd   | cosine    | — | — | — | — |

## Current best (UPDATE only after 2-seed evidence)
- None yet — phase 1 in progress. Provisional 1-seed leader: **iter 002 Cell C
  (AutoAugment) at test_acc=0.9519** (seed=42), but Cell B (iter 004,
  standard aug, 0.9477) lands within 0.42 pp and the gap is concentrated in
  just bird (−2.9 pp) and automobile (−1.1 pp); on cat/airplane Cell B
  *beats* Cell C. Cell A floor LOCKED at **iter 003 = 0.8828 (best) / 0.8812
  (final)**. Augmentation deltas now decomposed:
  Δ(B − A) = **+0.0649 (best)** — most of the gain comes from the simple
  crop+flip prior; Δ(C − B) = **+0.0042 (best)** — AutoAugment's marginal
  value over standard aug is small and within seed-noise band. **2-seed
  (seed=4078) replay of Cell C is now required before crowning** — at this
  variance, +0.42 pp may not survive. **Cell D anchored** at iter 005 =
  **0.9354 (best, ep51) / 0.9350 (final)** — AdamW (lr=1e-3, wd=1e-4) with
  Cell B's recipe loses **−1.23 pp** vs Cell B; optimizer-family swap is a
  Failure on this setup, do **not** propagate Cell D as parent for E/F.
  **Cell E anchored** at iter 006 = **0.9394 (best, ep49) / 0.9386 (final)**
  — multistep `[30, 45], γ=0.1` with Cell B's recipe loses **−0.83 pp**
  vs Cell B; the textbook ladder fires (ep30 jump +10.4 pp, ep45 jump
  +0.6 pp) but cosine's smooth tail still wins, so A3 is a Failure axis
  on this recipe and Cell F should continue to build on Cell B.

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
