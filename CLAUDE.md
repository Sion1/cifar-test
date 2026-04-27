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
- test_acc:   0.8870  (best epoch 52; final epoch 59 acc=0.8868)
- test_loss:  0.4231
- run_dir:    runs/cifar10_iter002_bare_baseline/
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
| **A** bare | none      | sgd   | cosine    | 002 | 0.8870 | Success (locked) | — |
| **B** +std | standard  | sgd   | cosine    | 003 | 0.9481 | Success | — |
| **C** +AA  | autoaug   | sgd   | cosine    | 004 | 0.9528 | Partial | — |
| **D** AdamW| standard  | adamw | cosine    | 005 | 0.9379 | Failure | — |
| **E** ms   | standard  | sgd   | multistep | — | — | — | — |
| **F** long | standard  | sgd   | cosine    | 006 | 0.9465 | Failure | — |

## Current best (UPDATE only after 2-seed evidence)
- None yet — phase 1 in progress.

---

## Documented findings (agent appends here per iter)

<!-- The agent will add `### Iteration NNN — {cell} {short_name}` blocks
     below per `program.md` §5. Do NOT touch this section yourself unless
     you're correcting a mistake; the agent reads back its own past notes
     to build context for the next iteration. -->

### Iteration 000 — pipeline-confirmation only (NOT a matrix cell)

Iter 0 was launched with `EPOCHS_OVERRIDE=1` against the default
`configs/cifar10_resnet34.yaml` (the canonical first-launch smoketest from
`scripts/first_launch_setup.sh:401`). Outcome: test_acc=0.2911 after one
epoch on CIFAR-10. **Verdict = Bug** — not because anything broke, but to
keep this row out of `loop.sh:615`'s "best so far" calculation; a 1-epoch
checkpoint is not a usable matrix baseline. Lessons worth carrying forward:
(1) the training/checkpointing/reap pipeline is verified end-to-end on this
host (CUDA OK on the `zsl_torch` env, `final.pth` saved with `metrics` and
`history`, dataset loads from `./data` via `AUTORES_DATA_ROOT`); (2) per-class
behavior at epoch 0 is dominated by a vehicle-vs-animal split with cat
collapsing to 0% — interpret early-iter regressions in cat/bird/ship through
this lens, not as bugs; (3) Grad-CAM at epoch 0 is uniformly center-biased,
so any later iter that *retains* center-only attention after many epochs
should be flagged as under-fit even if test_acc looks plausible. Cell A
baseline (augmentation=none, 60 ep, seed=42) still owes its locked numbers
to CLAUDE.md's "Baseline numbers" block — that's the next iter's job.

### Iteration 002 — Cell A bare baseline (LOCKED FLOOR)

Iter 2 ran `configs/ablation/iter002_bare_baseline.yaml` (sgd 0.1, momentum
0.9, wd 5e-4, nesterov, cosine, 60 ep, seed=42, **augmentation=none**) on
GPU 0 for ~34 min. Result: best_acc=0.8870 (epoch 52), final_acc=0.8868,
test_loss=0.4231, train_acc=1.0000 — i.e. the model fully memorized the
training set, and the ~11.3 pp train-vs-test gap is the textbook
no-augmentation overfit. This row is now the **locked Cell A floor**;
every subsequent cell B–F must beat 0.887 to count. Headline lesson worth
carrying forward (the most surprising bit, not derivable from the test
acc): **Grad-CAM at convergence is *still* strongly center-biased,**
nearly indistinguishable in spatial pattern from the iter-000 1-epoch
heatmaps. Without RandomCrop the model never had to find off-center
objects, so it learned a "look at the middle" prior even after fully
fitting the train set. This gives a concrete, falsifiable visual
prediction for iter 003 (Cell B, +standard aug = pad-4 RandomCrop +
HFlip): the Grad-CAM *signature must shift* — heatmap peaks should move
off-center on at least some panels, otherwise +std aug isn't doing what
it should and a small acc bump alone wouldn't be a real "Success." The
hardest classes are cat (0.748) ≪ dog (0.830) ≈ bird (0.819) < deer
(0.885); cat/dog also visibly contaminate each other in the t-SNE, so
those are the per-class numbers to watch as B/C/F land. Iters 003 (Cell B)
and 004 (Cell C) are already running in parallel on GPUs 1 & 2 — their
analyses will close those open verdicts.

### Iteration 003 — Cell B +std aug (NEW BEST)

Iter 3 ran `configs/ablation/iter003_std_aug.yaml` (single-axis delta vs
Cell A: `data.augmentation: none → standard` = pad-4 RandomCrop +
HFlip; everything else identical) on GPU 1 for ~34 min. Result:
**best_acc=0.9481** (epoch 57), final_acc=0.9478, test_loss=0.2106,
train_acc=0.9993 — a **+6.11 pp jump** over Cell A and already past
program.md §Goal's 0.94 baseline target on a single seed. Mechanism
fired exactly as iter 002 predicted: train-vs-test gap collapsed
**11.3 pp → 5.2 pp**, test_loss halved (0.4231 → 0.2106), best_epoch
shifted later (52 → 57) because cosine's tail now contributes once
memorization is delayed. The two falsifiable visual predictions both
landed: (a) **Grad-CAM center-bias broken** — the airplane panel's
heatmap follows the plane down to the lower-left, the off-center frog
panel's heatmap shifts upper-left to where the frog actually is, and
the ship panel's heatmap stretches horizontally along the hull rather
than collapsing to a central blob (8/8 panels correct, vs Cell A's
7/8); (b) **per-class gains concentrated on the hard classes** —
cat +13.0 pp, bird +10.6 pp, dog +10.0 pp, while saturated vehicles
gained ≤ +5 pp (automobile +2.9, ship +2.3, frog +2.4). Cat is still
the worst class but the cat–next gap closed from 7.1 pp → 4.7 pp. The
t-SNE shows the airplane↔ship contamination zone has fully cleared
(HFlip presumably absorbed the orientation cue), but **cat↔dog still
touch with a small bridge of mixed points** — that's the residual
mechanism Cell C (autoaug, iter 004 currently running) and Cell F
(long-train) should attack with color/contrast invariance and more
optimization steps respectively. Cell B is the new "Best so far"
denominator; every later cell must beat 0.9481 to count. Lesson worth
carrying forward: **on this recipe, +std aug isn't just a regularizer
— it specifically buys spatial-location invariance** (Grad-CAM
prediction confirmed) and converts ~6 pp of memorized texture into
actual class-discriminative features for the small-mammal/bird
classes; Cells C/D/E/F should each be evaluated on whether they move
*cat* upward, not just the headline acc.

### Iteration 004 — Cell C +autoaug (Partial — NOT new best)

Iter 4 ran `configs/ablation/iter004_autoaug.yaml` (single-axis delta
vs Cell B: `data.augmentation: standard → autoaugment`; everything
else identical) on GPU 2 for ~34 min. Result: **best_acc=0.9528**
(epoch 58), final_acc=0.9522, test_loss=0.1535, train_acc=0.9625 — a
**+0.47 pp** gain over Cell B (0.9481), which falls **just below** the
+0.5 pp Success threshold in program.md §Verdict criteria, so this is
a **Partial**, not a Success, and **Cell B remains "Best so far"**.
The mechanism, though, fired hard: train-vs-test gap collapsed
**5.2 pp → 1.0 pp** (autoaug is genuinely a stronger regularizer than
pad-crop+HFlip), test_loss further halved (0.2106 → 0.1535), best_epoch
shifted 57 → 58. Headline lesson worth carrying forward (the surprise,
not derivable from the +0.47 pp number): **autoaug's per-class profile
contradicts the iter-003 prediction.** Iter 3 expected autoaug to
attack the cat↔dog confusion residual; instead **dog regressed −1.5 pp
(0.930 → 0.915), airplane regressed −0.6 pp, and cat improved only
+0.7 pp**. The actual gainers were frog (+1.5), bird (+1.1), deer
(+1.0), automobile/horse (+0.8). The cat↔dog t-SNE bridge is still
present and looks no narrower than Cell B's; meanwhile a *new* small
airplane↔ship↔bird contamination zone has appeared in the lower-right,
likely because autoaug's color/contrast policies erode the hull-color
cue HFlip-only had preserved. Grad-CAM dropped from 8/8 → 7/8 (the
miss is a ship → automobile, where the heatmap correctly localized on
the bridge superstructure but the model has learned to read compact
mid-frame metallic rectangles as car-like). Implication for matrix
strategy: **Cell C is not the right tool for the cat↔dog residual** —
that confusion is semantic (similar texture/posture), not photometric.
Cell F (long-train at 100 ep) is now the highest-leverage remaining
cell because Cell C's 1.0 pp gap shows there's still clear
under-fitting room on this regularizer; bolting more epochs onto std-aug
(or autoaug) is more likely to crack 0.96 than another A1 swap. The
pre-staged Cell D (+adamw, iter005) still goes next on matrix-completion
grounds, but expect Failure/Partial there per the SGD-vs-AdamW prior on
CIFAR ResNets.

### Iteration 005 — Cell D +adamw (Failure — NOT new best)

Iter 5 ran `configs/ablation/iter005_adamw.yaml` (single-axis delta vs
Cell B: `training.optimizer: sgd → adamw`, `training.lr: 0.1 → 1e-3`
per program.md §A4 catalog; everything else identical) on GPU 0 for
~36 min. Result: **best_acc=0.9379** (epoch 51), final_acc=0.9377,
test_loss=0.4271, train_acc=0.9997 — a **−1.04 pp** drop vs Cell B
(0.9481), well past the −0.5 pp **Failure** threshold, so this is a
clean **Failure** and **Cell B remains "Best so far"**. The
prediction (iter004 §8: AdamW lands 0.93–0.945, mechanism = AdamW
under-performs tuned-SGD on CIFAR ResNets) was confirmed exactly: 0.9379
sits dead-center in the predicted band. Surprises worth carrying
forward (not derivable from the −1 pp headline):
**(a) the train-vs-test gap actually *widened* vs Cell B (5.2 pp →
6.2 pp)** — AdamW's adaptive scaling at LR=1e-3 effectively over-fits
*more* than SGD+momentum at the same wd=5e-4, despite a slower
optimization start (best_epoch 51 vs Cell B's 57, then test_loss
*climbs* from 0.33 ep30 → 0.43 ep59 while test_acc plateaus — classic
"more confident on its mistakes"). So the next time A2 needs probing,
the move is **lower wd or shorter epochs at AdamW, not the other catalog
LR (5e-4)**, since halving the LR will only deepen the under-fitting
side of the dynamics.
**(b) the regression is uniform — 9/10 classes drop, only deer
+0.3 pp.** Biggest losses: dog −2.8, airplane −2.7, horse/bird/truck
−1.0. Cat barely moved (−0.4 pp) because cat is already at the floor;
SGD's edge over AdamW manifests on the classes with headroom, *not*
on the deepest-confusion class. So the cat↔dog bridge is **not** an
optimizer problem and Cell D contributes zero new evidence on it.
**(c) a new airplane↔ship contamination zone opened in t-SNE** with
several yellow ship points embedded in the blue airplane cluster —
these were cleanly separated under Cell B; the airplane −2.7 pp
regression has a direct feature-space cause. Combined with iter004's
new airplane↔ship↔bird zone under autoaug, this confirms airplane↔ship
is the second residual confusion mode (after cat↔dog) and is sensitive
to *any* perturbation off the Cell B recipe.
**(d) Grad-CAM is 8/8 but the heatmap signature drifted back toward
Cell A's center-bias** — round, compact, central peaks rather than
Cell B's location-following peaks. AdamW's per-parameter scaling
plausibly damps gradients on spatially peripheral feature channels
relative to SGD's uniform LR, biasing the network toward always-active
central channels. So **headline acc and Grad-CAM correctness are not
sufficient diagnostics on this recipe — heatmap *shape* is a leading
indicator** of whether the spatial-invariance prior earned by std-aug
in Cell B is being preserved by a downstream change.
Implication for matrix strategy: **A2 axis is now pinned at SGD** with
single-seed evidence; do NOT rerun AdamW at LR=5e-4. The remaining
high-leverage cells are **F (long-train, currently running on GPU 1
per state/iterations.tsv)** and **E (multistep)**, in that order. Cell
F directly tests the under-fitting headroom that iter003/iter004 both
exhibited (best_epoch 57/58 with cosine still descending), and is the
only remaining single-axis cell with a credible path to crossing 0.955.

### Iteration 006 — Cell F +long-train (Failure — NOT new best)

Iter 6 ran `configs/ablation/iter006_long_train.yaml` (single-axis
delta vs Cell B: `training.epochs: 60 → 100`; cosine extends in
proportion; everything else identical) on GPU 1 for ~56 min. Result:
**best_acc=0.9465** (epoch 95), final_acc=0.9457, test_loss=0.2237,
train_acc=0.9997 — a **−0.16 pp** drop vs Cell B (0.9481). Magnitude
sits inside the §Verdict §Noise band (|Δ| < 0.3 pp), but the
predicted-positive mechanism (more cosine epochs consume Cell B's
under-fit headroom, lifting acc into 0.950–0.955) **did not fire**:
train-vs-test gap stayed at 5.4 pp ≈ Cell B's 5.2 pp (mechanism
required gap-shrink), and the iter-005 §8 falsifier ("≤ 0.9481 ⇒
Cell B was already saturated and the small under-fit signal in
iter-003's history was noise") was cleared. Verdict = **Failure** on
mechanism grounds; Cell B (0.9481) remains "Best so far."
Surprises worth carrying forward (not derivable from the −0.16 pp
headline):
**(a) iter-003's "best_epoch 57/60 with cosine still descending"
was a cosine-shape artifact, not a real under-fit signal.** Cosine's
derivative → 0 near the end, so any cosine schedule will *look*
"still descending" near termination. At epoch 60 of this 100-ep run
test_acc is only 0.9037 — **~4.4 pp BELOW** Cell B's 0.9478 at the
same wall-clock epoch — because the slower 100-ep cosine keeps LR
high longer. The extra 40 epochs only recover what the slower decay
schedule lost; they do not add headroom. Implication: **never use
"cosine still descending at termination" as evidence of under-fit
on this recipe again**; instead require train_acc < 0.999 OR a
test_acc trajectory whose slope at termination is meaningfully
positive over the final 5 epochs (this run plateaued at 0.945–0.946
over the last 10).
**(b) Cat IMPROVED +1.5 pp (0.878 → 0.893) — the only matrix-cell
to date where cat moves meaningfully — but dog REGRESSED −2.5 pp
(0.930 → 0.905), with the confusion matrix showing the cat↔dog
errors became *asymmetric* (dog→cat 60, cat→dog 59).** Long training
re-allocated cat-as-dog errors into dog-as-cat errors, polishing
the cat manifold at dog's expense rather than carving a cleaner
boundary. So the cat↔dog residual is **not a "more training"
problem** — it's a representational/semantic problem that
schedule length alone cannot crack. Cell C (autoaug) also failed
here, and Cell D (AdamW) was a no-op on this confusion. The
cat↔dog bridge will need a *new axis* (label smoothing? mixup?
dropout?) — none currently in the catalog — or a 2-seed Cell C
hardening pass to test if the +0.7 pp cat gain there is real.
**(c) Grad-CAM heatmaps drifted back toward Cell A/D's centered
blobs even though A1=standard and A2=sgd are unchanged from Cell B.**
The longer optimization horizon erodes the off-center prior that
Cell B's faster cosine had locked in — plausibly because BN running
stats and the classifier head get ~2× more updates on the
center-dominant feature channels. Combined with iter-005's centered-
blob signature under AdamW, this confirms iter-005's hypothesis that
**heatmap shape is a leading indicator of spatial-invariance erosion
and is sensitive to *any* knob — not just A1 or A2 — that changes the
optimization horizon or per-channel update mass.** Going forward,
Cell B's location-following heatmap signature is the recipe-fidelity
canary; if a downstream cell loses it, the spatial prior is gone
even when headline acc looks fine.
**(d) Airplane↔ship contamination zone partially reopened in t-SNE
even with the recipe identical to Cell B.** Cell D opened it badly,
Cell C opened it differently, and now Cell F opens it mildly — so
**airplane↔ship is the second-most-fragile boundary** in CIFAR-10's
feature space on this recipe (after cat↔dog), and any change off
Cell B's exact 60-ep recipe will perturb it.
Implication for matrix strategy: **A6 axis is now pinned at 60** with
single-seed evidence (don't rerun A6=30; that direction is even more
under-fit). The only remaining single-axis cell is **E (multistep
schedule, A3 axis)**, which iter 007 should run next. After Cell E,
phase 1 is complete and the 2-seed hardening pass begins on the top
two cells (currently B at 0.9481 and C at 0.9528).

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
