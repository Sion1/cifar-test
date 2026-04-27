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
- test_acc:   <fill in>
- test_loss:  <fill in>
- run_dir:    runs/cifar10_iter001_no_aug/
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
| **A** bare | none      | sgd   | cosine    | — | — | — | — |
| **B** +std | standard  | sgd   | cosine    | — | — | — | — |
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
