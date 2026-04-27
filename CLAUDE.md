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
