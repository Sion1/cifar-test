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
| **B** +std | standard  | sgd   | cosine    | — | — | — | — |
| **C** +AA  | autoaug   | sgd   | cosine    | 002 | 0.9519 | success (1-seed) | TBD |
| **D** AdamW| standard  | adamw | cosine    | — | — | — | — |
| **E** ms   | standard  | sgd   | multistep | — | — | — | — |
| **F** long | standard  | sgd   | cosine    | — | — | — | — |

## Current best (UPDATE only after 2-seed evidence)
- None yet — phase 1 in progress. Provisional 1-seed leader: **iter 002 Cell C
  (AutoAugment) at test_acc=0.9519** (seed=42). Cell A floor now LOCKED at
  **iter 003 = 0.8828 (best) / 0.8812 (final)** — Δ(C − A) = +0.0691 (best) /
  +0.0707 (final), the first clean 1-seed augmentation delta. Awaiting Cell B
  (iter004_std) and Cell D (iter005_adamw) before crowning a winner or
  scheduling the 2-seed replay (seed=4078).

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
