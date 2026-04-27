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
| **C** +AA  | autoaug   | sgd   | cosine    | — | — | — | — |
| **D** AdamW| standard  | adamw | cosine    | — | — | — | — |
| **E** ms   | standard  | sgd   | multistep | — | — | — | — |
| **F** long | standard  | sgd   | cosine    | — | — | — | — |

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
