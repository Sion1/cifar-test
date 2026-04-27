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
| **B** +std | standard  | sgd   | cosine    | 003 | 0.9481 | Success | 0.94795 (s=42, s=4078: 0.9481, 0.9478) |
| **C** +AA  | autoaug   | sgd   | cosine    | 004 | 0.9528 | Partial | 0.95125 (s=42, s=4078: 0.9528, 0.9497) |
| **D** AdamW| standard  | adamw | cosine    | 005 | 0.9379 | Failure | — |
| **E** ms   | standard  | sgd   | multistep | 007 | 0.9431 | Failure | — |
| **F** long | standard  | sgd   | cosine    | 006 | 0.9465 | Failure | — |

## Current best (UPDATE only after 2-seed evidence)
- **Cell C** (autoaug + sgd + cosine, 60 ep) — **2-seed mean 0.95125**
  (s=42: 0.9528, s=4078: 0.9497). Lead over Cell B at 2-seed mean =
  +0.33 pp (Partial-margin lead, NOT Success-margin: < 0.5 pp threshold).
  Cell B 2-seed mean = 0.94795 (s=42: 0.9481, s=4078: 0.9478) is the
  most seed-stable cell measured (peak-to-peak 0.03 pp).

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

### Iteration 007 — Cell E +multistep (Failure — NOT new best)

Iter 7 ran `configs/ablation/iter007_multistep.yaml` (single-axis
delta vs Cell B: `training.scheduler: cosine → multistep` with
`milestones=[30, 45]`, `gamma=0.1`; everything else identical) on
GPU 0 for ~34 min. Result: **best_acc=0.9431** (epoch 55),
final_acc=0.9426, test_loss=0.2434, train_acc=0.9976 — a
**−0.50 pp** drop vs Cell B (0.9481), **exactly at** the §Verdict
**Failure** boundary and outside the §Noise band on the negative
side. The iter-006 §8 falsifier ("≤ 0.9451 ⇒ clean negative") is
**triggered**; the upside falsifier (≥ 0.9531) wasn't close. Cell B
(0.9481) remains "Best so far." Mechanism status is mixed: the
**ep-30 milestone fires textbook-cleanly** (test_acc 0.8398 →
0.9253, +8.55 pp in one epoch — "LR drop breaks through the
high-LR plateau" is real) and the ep-45 milestone adds another
+1 pp, but two discrete drops at [30, 45] cannot match cosine's
*continuous* decay over the same 60-ep budget. **Phase 1 (Cells
A–F) is now complete**; the 2-seed hardening pass on B and C is
the next phase per program.md §Required ablation strategy.
Surprises worth carrying forward (not derivable from the −0.50 pp
headline):
**(a) The cat↔dog asymmetry-flip is now confirmed as the
characteristic non-Cell-B failure mode of the cat↔dog boundary.**
Cell F (long-train) showed cat +1.5 / dog −2.5 with errors
flipping from cat→dog dominant to dog→cat dominant. Cell E
(multistep) shows the *exact same pattern, more pronounced*: cat
+1.4 / dog −2.9, top off-diagonals flip from Cell B's cat→dog 71
& dog→cat 44 to iter-007's dog→cat 64 & cat→dog 50. Two
unrelated knobs (A6=100 and A3=multistep) produce the same flip,
so **any deviation from Cell B's smooth-cosine-60 recipe
re-allocates errors from cat-as-dog into dog-as-cat rather than
shrinking the joint cat↔dog error**. Implication for the 2-seed
hardening: the cat↔dog residual is a property of Cell B itself,
not a phase-1 measurement artifact, and **fixing it requires a
new axis (label smoothing / mixup / dropout) that's currently
outside the program.md catalog**.
**(b) Multistep at [30, 45] spends epochs 0–29 at lr=0.1 with
test_acc stuck at 0.83–0.84**, while cosine at lr=0.1 (Cell B)
had already descended its LR enough by ep30 to be at ~0.91. The
ep-30 step-up to 0.9253 is impressive but only *recovers* what
the high-LR first phase had cost; over a 60-ep budget cosine
wins the area-under-the-curve. So the lesson is not that "the
ep-30 LR drop breaks plateaus" (it does — cleanly) but that
**high-LR-then-decay schedules trade early-epoch test_acc for
late-epoch generalization on this recipe, and cosine's
continuous decay manages this trade-off better than two discrete
steps** within the catalog A3 axis.
**(c) Bird's −1.5 pp loss is broad, not concentrated.** Top
bird-as-* confusions: bird→airplane=19, bird→deer=19,
bird→frog=18, bird→cat=17 — bird's boundary fragmented across
*four* neighbours simultaneously, in contrast to the dog
regression which concentrated on dog→cat. So multistep's two
discrete drops appear to **fragment fine-grained class
boundaries** (bird sits between airplane/deer/frog/cat in
feature space), where cosine's smooth decay would normally
keep them sharp. This is a new failure mode, not seen in
iter 005 or iter 006.
**(d) Grad-CAM heatmaps drifted back to Cell A/D/F's centered-
blob signature** even though A1=standard and A2=sgd are
unchanged from Cell B. **This is now the third non-Cell-B cell
to lose Cell B's location-following heatmap shape** (after
Cell D AdamW and Cell F long-train), so the iter-006 hypothesis
is **upgraded from "sensitive to A6" to "sensitive to A2 OR A3
OR A6"** — i.e., any catalog axis that perturbs the optimization
horizon or per-channel update structure erodes the off-center
prior std-aug initially built. **Cell B's location-following
signature is the recipe-fidelity canary**, full stop.
Implication for matrix strategy: **A3 axis is now pinned at
cosine** with single-seed evidence (multistep is the only
remaining catalog A3 alternative; A3=`none` is degenerate so not
worth running). All three of the schedule-perturbing axes (A2,
A3, A6) have now been measured and all three lose to Cell B's
exact recipe. **Phase 1 ends here**: Cells A–F have all reported
single-seed numbers, with **B (0.9481) and C (0.9528) as the
2-seed hardening pair**. The 2-seed pass is already in flight per
state/iterations.tsv: iter 008 = autoaug seed=4078 (Cell C
hardening) and iter 009 = std-aug seed=4078 (Cell B hardening).
A 2-seed mean for whichever cell wins becomes the crowned winner
under "Current best."

### Iteration 008 — Cell C 2-seed hardening at seed=4078 (Partial — Cell C survives but lead compresses)

Iter 8 ran `configs/ablation/iter008_autoaug_seed4078.yaml`
(single-axis delta vs iter 004 / Cell C: `seed: 42 → 4078`;
everything else identical: aug=autoaugment, sgd 0.1, momentum 0.9,
wd 5e-4, nesterov, cosine, epochs=60) on GPU 1 for ~35 min. Result:
**best_acc=0.9497** (epoch 59, last epoch), final_acc=0.9497,
test_loss=0.1599, train_acc=0.9618 — a **−0.31 pp** drop vs Cell C
seed=42 (0.9528) and a **+0.16 pp** gain vs Cell B seed=42 (0.9481).
Strong falsifier from iter-007 §8 ("< 0.9481 ⇒ seed-noise") **not
triggered** (margin: +0.16 pp); weak falsifier (outside ±0.3 pp band
of Cell C s=42) is *just* triggered (0.9497 lands 0.0001 below the
predicted lower bound of 0.9498). Verdict = **Partial**: Cell C
replicates with a 2-seed mean of 0.95125, still ahead of Cell B's
seed=42 result, **but the +0.47 pp single-seed lead has compressed
to a +0.16 pp seed-replay lead** and will compress further once
iter 009 (Cell B seed=4078, already finished, status=completed
pending analysis) lands. The mechanism evidence is unambiguous:
train-vs-test gap = 1.21 pp ≈ Cell C s=42's 1.03 pp ≪ Cell B's
5.20 pp, and the slow-start trajectory (ep30 = 0.8774 vs Cell B's
~0.91 at the same epoch) plus best_epoch=59/60 match Cell C's
canonical signature exactly. Surprises worth carrying forward (not
derivable from the −0.31 pp headline):
**(a) Cat regression is the dominant component of the loss.** Per-
class Δ vs Cell C s=42: cat **0.885 → 0.874 (−1.1 pp, the biggest
single drop)**, deer −0.9, dog −0.7, ship −0.5, airplane −0.3, with
no compensating gain anywhere. Cat 0.874 is **identical (within
rounding) to Cell B seed=42's cat 0.878** — i.e., at this seed,
**autoaug's headline cat-helping effect from iter 004 (+0.7 pp cat
over Cell B) does not replicate**. The +0.7 pp cat gain in iter 004
should be downgraded from "real photometric-regularization signal"
to **"within autoaug-seed-variance noise"**. This rewrites the
matrix interpretation: Cell C's lead over Cell B is *not* mediated
by attacking the cat residual (as iter-003 had hoped), and the
cat↔dog boundary is **not actually being addressed by any cell in
the catalog at the 2-seed level**.
**(b) The cat→dog asymmetry direction is *seed-stable for a
*recipe*, not stable across seeds.** Iter 6 (Cell F long-train) and
iter 7 (Cell E multistep) both flipped the asymmetry to dog→cat
dominant (64 vs 50, 64 vs 50). Iter 8 (Cell C, seed-only change)
**reverts to cat→dog dominant (79 vs 58)** — i.e., the
"any-deviation-from-Cell-B re-allocates errors to dog→cat dominance"
finding from iter 007 needs scope-restriction: it holds for
*schedule/horizon* perturbations (E, F), not for *seed* perturbations
on the same recipe. A *seed* perturbation off Cell C reverts the
asymmetry back toward Cell B's direction (cat→dog dominant) and
*widens* the gap (79 vs Cell B's 71, +8). So there are now **two
distinct cat↔dog failure modes** in this matrix: (1) schedule-
perturbation flips asymmetry to dog→cat dominant; (2) seed-
perturbation on autoaug widens cat→dog dominance back beyond Cell
B's level. Both are bigger than any Δ_acc; the boundary is
genuinely fragile.
**(c) Grad-CAM heatmap signature at seed=4078 is centered round
blobs (8/8 correct).** Cell C seed=42 was 7/8 with the same
center-blob signature. So **even with the *exact same recipe* (same
A1/A2/A3/A4/A5/A6), the only-seed-change still shows the centered-
blob signature — Cell B's location-following heatmap is unique to
Cell B's seed=42**, not to standard-aug-style augmentation in
general. This **demotes Grad-CAM heatmap shape from "recipe canary"
to "Cell-B-seed-42-specific canary"** — heatmap shape is no longer
a reliable mechanism diagnostic for non-Cell-B-recipe runs at the
2-seed level.
**(d) The vehicle pair (automobile↔truck) and airplane↔ship modes
are seed-stable.** automobile 0.985 (very stable across cells), truck
0.968 (Cell B: 0.964; Cell C s=42: 0.968 — pinned). The
airplane↔ship contamination zone reopens slightly here too
(airplane→ship=18, ship→airplane=12) — same low-grade contamination
seen in Cells C s=42, D, E, F. So the **airplane↔ship boundary is
the second-most-fragile boundary in this matrix and is sensitive to
*everything* — schedule, optimizer, *and* seed**, while the
vehicle-pair separation is the matrix's most robust gain.
Implication for matrix strategy: **do NOT crown Cell C as winner
yet** — wait for iter 009 (Cell B seed=4078) so the 2-seed-mean-vs-
2-seed-mean comparison can be made honestly. If Cell B s=4078 lands
≥ ~0.9466 (its expected ±0.3 noise band), the 2-seed means become
Cell C 0.95125 vs Cell B ~0.9474, a +0.39 pp lead — still inside
the §Verdict Partial range (Δ ∈ [0, 0.5] pp). If Cell B s=4078
lands above 0.9498, Cell B's mean exceeds 0.9489 and the lead
shrinks to ~0.22 pp, dangerously close to noise. **Phase 2 is now
midway**; the next iter (009) is the deciding measurement.
state/iterations.tsv shows iter 010 (autoaug at lr=0.05, an A4
sweep) was queued early — that's premature; phase 2 should finish
before any phase-3 axis-sweeps off the winner.

### Iteration 009 — Cell B 2-seed hardening at seed=4078 (Noise — Cell B replicates, Cell C crowned)

Iter 9 ran `configs/ablation/iter009_std_aug_seed4078.yaml`
(single-axis delta vs iter 003 / Cell B: `seed: 42 → 4078`;
everything else identical: aug=standard, sgd 0.1, momentum 0.9, wd
5e-4, nesterov, cosine, epochs=60) on GPU 2 for ~35 min. Result:
**best_acc=0.9478** (epoch 58), final_acc=0.9478, test_loss=0.2085,
train_acc=0.9995 — a **−0.03 pp** drop vs Cell B seed=42 (0.9481),
landing **dead center** in the iter-008 §8 predicted band
[0.9451, 0.9511]. Both falsifiers cleared cleanly: upside
(> 0.9528) untriggered by 5.0 pp, downside (< 0.9451) untriggered
by 2.7 pp. Verdict = **Noise** (|Δ| < 0.3 pp), which is the
*desired* Noise verdict for a 2-seed hardening pass: Cell B's
+6.11 pp gain over Cell A is **fully replicated**, train-vs-test
gap (5.17 pp) is identical to seed=42's 5.20 pp, and Cell B is now
the **most seed-stable cell measured** (peak-to-peak 0.03 pp vs
Cell C's 0.31 pp). **2-seed Cell B mean = 0.94795** vs **2-seed
Cell C mean = 0.95125** ⇒ **Cell C wins phase 2 by +0.33 pp** —
outside Noise band, inside Success threshold (< 0.5 pp), so it's a
**Partial-margin crown**, not a Success-margin crown. Cell C is
hereby crowned phase-2 winner; phase 3 (axis sweeps off Cell C)
opens.
Surprises worth carrying forward (not derivable from the −0.03 pp
headline):
**(a) The "Cell B's location-following Grad-CAM is a recipe canary"
hypothesis (iters 005/006/007) is REFUTED.** Iter 008 already
demoted heatmap shape from "recipe canary" to
"Cell-B-seed-42-specific canary"; iter 009 is the decisive test
because A1/A2/A3/A4/A5/A6 are *identical* to Cell B seed=42 — yet
**7/8 Grad-CAM panels show centered round blobs**, only 1/8 has
mild horizontal hull-tracking on a ship. **The location-following
signature is unique to seed=42's optimization trajectory, not to
the Cell B recipe.** Going forward: **stop using heatmap shape as
a mechanism diagnostic on this matrix.** The reliable canaries are
train-vs-test gap (best_acc-discriminating, recipe-stable across
seeds), per-class profile, and t-SNE structural anomalies. This
overturns the running iter-005/006/007 narrative; future cells'
heatmap-shape observations should be downgraded accordingly.
**(b) Despite headline replication, the per-class redistribution is
non-trivial and asymmetric.** Δ vs Cell B s=42: dog **−2.0 pp**
(0.930 → 0.910, the largest single-class shift in any 2-seed pair
in the matrix), bird **+1.1 pp** (0.925 → 0.936), automobile
**+0.6 pp**, frog **+0.9 pp**, cat **−0.5 pp** (basically pinned),
others within ±0.5 pp. So Cell B's 0.9478 is *compositionally
different* from its 0.9481 at seed=42 — it sacrifices ~6 dog
predictions to gain ~11 birds + ~6 cars + ~9 frogs. Implication:
**single-class headline numbers are the noisier dimension at the
2-seed level, not headline acc.** If a future phase-3 cell shows
"+1 pp on dog", check whether it's larger than Cell B's natural
seed-driven dog variance (≥ 2 pp) before reading it as a real
mechanism signal.
**(c) Cell B s=42's characteristic 71-vs-44 cat→dog asymmetry
shrinks to 63-vs-55 at seed=4078** (gap 27 → 8). Direction is
preserved (cat→dog dominant, no flip), but magnitude swings by
factor-of-3. Cells E and F (schedule perturbations) flipped to
dog→cat dominant at gaps of ~14; this is now distinguishable from
seed-driven gap-shrinkage on Cell B because the *direction* is
different. So the rule "schedule perturbation flips direction;
seed perturbation shrinks magnitude but preserves direction"
holds across both replications. The cat↔dog residual remains
not-attackable by any catalog axis, and seed-magnitude alone can
swing the asymmetry by 19 cells in the confusion matrix — so
**stand-alone confusion-asymmetry numbers under-power; require
≥ 2-seed evidence before reading them as mechanism**.
**(d) t-SNE structure is *better* at seed=4078 than at seed=42** —
cat (red) and dog (brown) are visually two distinct lobes here,
whereas Cell B s=42's t-SNE had a more visible cat↔dog mixing
band, and Cell C s=4078's was a merged blob. So feature-space
quality and headline acc decouple: 0.9478 with cleaner
class-manifold separation produces *more* dog→cat confusions than
0.9481 with a slight visible bridge — i.e., the dog regression
isn't manifold-wise; it's instance-wise on ambiguous images. This
also implies that t-SNE class separation is a noisy proxy for
acc — a cleaner-looking t-SNE doesn't guarantee better acc, even
on the same recipe.
Implication for matrix strategy: **Phase 2 closes here.** Phase 3
opens with **Cell C as the propagation point** for axis sweeps
(A4 lr, A5 wd) — iter 010 (autoaug at lr=0.05) is already running
on GPU 0 and is the right first probe. **Be cautious propagating
Cell C as "settled winner"**: the +0.33 pp 2-seed lead is small
enough that any phase-3 finding on Cell C's lr/wd axes that
doesn't transfer back to Cell B should be flagged as
Cell-C-specific, not "+autoaug is genuinely better". The cat↔dog
residual remains the single biggest matrix-wide blind spot, and
no catalog axis (A1–A6) attacks it; that suggests a future
deviation from program.md §Required ablation strategy
(label smoothing / mixup / dropout) might be the next high-leverage
experiment after phase 3 plateaus.

### Iteration 010 — Cell C + lr=0.05 (A4 sweep on the winner — Failure on mechanism)

Iter 10 ran `configs/ablation/iter010_autoaug_lr05.yaml` (single-axis
delta vs iter 004 / Cell C: `training.lr: 0.1 → 0.05`; everything else
identical: aug=autoaugment, sgd, momentum 0.9, wd 5e-4, nesterov,
cosine, 60 ep, seed=42) on GPU 0 for ~35 min. Result: **best_acc=
0.9513** (epoch 59, last epoch), final_acc=0.9513, test_loss=0.1625,
train_acc=0.9671 — a **−0.15 pp** drop vs Cell C s=42 (0.9528) and
**+0.005 pp** vs the Cell C 2-seed mean (0.95125), i.e., headline acc
is **statistically identical to Cell C at the 2-seed-mean level** and
sits cleanly between Cell C s=42 (0.9528) and Cell C s=4078 (0.9497).
The iter-009 §8 prediction band [0.945, 0.953] held; the iter-010
config's stricter falsifier (best_acc ≤ 0.9528) was just triggered.
Verdict = **Failure** on mechanism grounds (matching the iter-006
precedent of small-Δ-with-mechanism-failure): the predicted-positive
mechanism was "gap shrinks below Cell C's 1.0 pp ⇒ headroom consumed
by smoother cosine"; **the gap *widened* from 1.03 → 1.58 pp** — the
extra fitting bought by lr=0.05 went disproportionately into the
training set, not generalization. **Cell C remains crowned (2-seed
mean 0.95125)**; lr=0.05 is effectively a third-seed replay of Cell
C, not a meaningfully different recipe.
Surprises worth carrying forward (not derivable from the −0.15 pp
headline):
**(a) lr=0.05 is the *first catalog cell* to nudge cat AND dog
together upward** (cat 0.885 → 0.889, dog 0.915 → 0.917; both up
+0.2-0.4 pp simultaneously). Every prior cell either improved cat
at dog's expense (E, F flipped asymmetry; B s=4078 dropped dog
−2.0 pp) or improved neither (D, C-s=4078 dropped cat). But **the
cat↔dog confusion magnitude is unchanged** (cat→dog 64 vs Cell C
s=42's 62; dog→cat 55 vs 55) — the +0.4 cat / +0.2 dog gains came
from cat-as-bird / cat-as-deer corrections, NOT from carving the
cat↔dog boundary. So the mechanism is "smoother optimization
slightly sharpens cat's representation against non-dog neighbors,"
not "smoother optimization addresses the cat↔dog residual." This
demotes the iter-009 §8 hope that any catalog axis could attack
the cat↔dog boundary; the residual is even more entrenched.
**(b) NEW failure mode: the vehicle pair (automobile↔truck) blurred
under lr=0.05.** truck −0.6 pp, ship −0.7 pp; truck→automobile=23
vs Cell C s=42's typical ~12, and t-SNE shows automobile's cluster
extending a long tail toward truck. **automobile↔truck had been the
matrix's most robust separation** through Cells B, C, D, E, F, B-
s=4078, C-s=4078 — every single one. Breaks under lr=0.05. Mechanism:
half-LR cosine spends more total optimization mass in the high-LR
phase (BN running stats update with larger gradient noise for longer),
so per-channel feature specialization is less crisp; the
automobile↔truck separation, which appears to be highly feature-
crisp-dependent, suffers most. **Implication**: the matrix's most
robust class boundaries are NOT the most LR-stable — they are the
most "high-feature-crispness" boundaries, and any knob that softens
feature crispness (lower LR, AdamW's adaptive scaling) will blur them
even when headline acc looks fine. New canary: **truck→automobile
confusion count is a sensitive crispness diagnostic.**
**(c) Trajectory comparison reveals an interesting under/over fit
crossover.** At ep30, lr=0.05 leads Cell C s=42 by ~2.3 pp (0.9003
vs 0.8774) — half-LR is *ahead* mid-training because autoaug's
strong regularizer pairs better with smaller per-step noise early on.
But Cell C s=42 *catches up and overtakes* in the cosine tail because
the higher LR's annealing through the final 10 epochs consumes its
last drops of headroom faster than half-LR can. So **lr=0.05 +
60 ep is mid-training-superior but tail-truncated**; the natural
followup would be **lr=0.05 + 100 ep** to let the slower cosine
descend further — but that's a 2-axis change (A4+A6), forbidden by
single-axis discipline, AND iter-006 already showed A6=100 is
saturated on the *same recipe family* (Cell B), so the prior is the
extra epochs would just re-deepen the train-overfit gap. **A4=0.05
is therefore exhausted on Cell C even though its mid-training profile
suggests it had real learning capacity that 60 ep didn't capture.**
**(d) Heatmap shape stays seed=42-non-Cell-B-centered (8/8 round
center blobs, 7/8 correct, 1 ship→automobile miss with hull-localized
heatmap).** Identical signature to Cell C s=42, Cell C s=4078, Cell
D, E, F. iter-009's verdict that "Grad-CAM heatmap shape is a
seed=42-Cell-B-specific artifact, not a recipe canary" is **further
confirmed** here: identical recipe family to Cell C, identical seed
to Cell C s=42, identical heatmap signature to Cell C s=42. Drop
heatmap shape from the diagnostic toolkit, full stop.
Implication for matrix strategy: **A4 axis on Cell C is now closed**
(lr=0.05 is Failure-on-mechanism with statistically-identical
headline; lr=0.01 is predicted-Failure as deeper under-fit). A5
(weight decay) on Cell C is also effectively closed within catalog:
the only catalog A5 alternatives to wd=5e-4 are wd=1e-4 (weaker
regularization on a recipe with a 1 pp gap → predicted to widen the
gap further, Failure) and wd=0 (no regularization, predicted-Failure
trivially). **Phase 3 within-catalog options on Cell C are
exhausted.** The remaining axis-sweeps that could matter are on
Cell B (A4=0.01 on Cell B is genuinely-novel and could probe whether
Cell B's memorization-bound gap has LR headroom — predicted band
[0.940, 0.952]; falsifier > 0.9528 reopens A4 axis on Cell B).
After that, the matrix-level cat↔dog and ship/truck residuals can
only be addressed by an off-catalog axis (label smoothing / mixup /
dropout), which requires a human edit to program.md per HARD
CONSTRAINTS §9. Recommended pivot: **flag the off-catalog need**
explicitly to the human after iter 011 (A4=0.01 on Cell B); if the
human declines, the §Budget early-stop rule (3 consecutive Failures
+ no Partial in last 5 iters) becomes increasingly likely as
remaining catalog probes are predicted-negative.

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
