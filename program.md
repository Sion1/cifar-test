# Research Program · template

> **This file is part of the Agentic Research Assistant framework.**
> The framework reads it on every iteration. Replace the contents of every
> section below with your own research's rules; keep the **section
> structure** intact so agents can parse it consistently.
>
> The example here is for the bundled CIFAR-10 + ResNet-34 demo.

---

## Goal
Maximize **CIFAR-10 test accuracy** while keeping a single ResNet-34 backbone
fixed. Baseline: ResNet-34 trained with SGD + cosine schedule + standard
augmentation.

- Baseline target acc:    ≥ 0.94
- Stretch goal:           ≥ 0.95
- Hard ceiling for STOP:  ≥ 0.96 (loop exits early once reached)

---

## Module catalog (this loop's frozen knobs)

The bot operates ONLY on these axes. Adding a new backbone, switching dataset,
or grafting a method from a paper is a HARD-CONSTRAINT violation (see #8/#9).

| Symbol | Axis | Values | Where |
|---|---|---|---|
| **A1** | data augmentation       | `none` / `standard` / `autoaugment`     | `data.augmentation` in YAML |
| **A2** | optimizer               | `sgd` / `adamw` / `adam`                 | `training.optimizer`        |
| **A3** | learning-rate schedule  | `cosine` / `multistep` / `none`          | `training.scheduler`        |
| **A4** | base learning rate      | `{0.1, 0.05, 0.01}` for SGD; `{1e-3, 5e-4}` for AdamW | `training.lr` |
| **A5** | weight decay            | sweep `{5e-4, 1e-4, 0}`                   | `training.weight_decay`     |
| **A6** | epochs                  | `{30, 60, 100}`                           | `training.epochs`           |

> Replace this catalog with **your** ablation axes — the framework doesn't
> care what they are, only that you list them and the bot only touches them.

## Required ablation strategy

Cover this matrix systematically before declaring any winner. Each cell ⇒ one
single-seed run; high-impact cells then get a 2nd-seed replay for hardening.

| Cell | A1 aug | A2 opt | A3 sched | Notes |
|---|---|---|---|---|
| **A — bare baseline**  | none      | sgd   | cosine    | floor; everything must beat this |
| **B — +std aug**       | standard  | sgd   | cosine    | tests A1 alone |
| **C — +autoaug**       | autoaug   | sgd   | cosine    | stronger A1 |
| **D — +adamw**         | standard  | adamw | cosine    | tests A2 |
| **E — multistep**      | standard  | sgd   | multistep | tests A3 |
| **F — long-train**     | standard  | sgd   | cosine    | epochs=100 |

After phase 1 (cells A–F): pick top 2, run with seed=4078, report 2-seed mean.
Only then is a winner crowned in `CLAUDE.md`.

---

## What you (the agent) can do each iteration

1. Read `program.md`, `CLAUDE.md`, last 3 `logs/iteration_*.md`,
   `state/iterations.tsv`.
2. Modify ONLY:
   - `configs/ablation/*.yaml` (clone an existing one, single-axis edit)
   - knobs already exposed in `src/cifar_demo/*.py` (no new files)
3. Launch ONE training via `bash run_experiment.sh <config> <iter_num>`.
4. After it finishes, analyze the resulting `runs/<exp>/final.pth`.
5. Generate **mandatory** visualizations (see below).
6. Write `logs/iteration_NNN.md` — auto-committed by `git_iter_commit.sh`.

---

## Mandatory per-iteration visualizations

Every completed experiment must produce, before the verdict is written:

1. **Feature t-SNE** (`figs/iter_NNN/tsne.png`) — `scripts/visualize_tsne.py`.
   Reports class separation in the penultimate feature space.
2. **Grad-CAM grid** (`figs/iter_NNN/cam.png`) — `scripts/visualize_cam.py`.
   8 random test images with overlay; visually highlights which regions the
   model uses to classify, including failure cases.
3. **Per-class accuracy table** (`figs/iter_NNN/per_class.csv`) — diagonal
   of the confusion matrix; spot the regressed classes.

---

## Per-iteration log format (`logs/iteration_NNN.md`)

```markdown
# Iteration NNN — {short_name}
Date: YYYY-MM-DD HH:MM | GPU: {id} | Duration: {h}

## 1. Hypothesis
2-3 sentences. Which cell are you targeting? What single config delta vs the
prior cell?

## 2. Falsification criterion
What numeric outcome would refute the hypothesis?

## 3. Changes made
The YAML diff and any code change.

## 4. Results
| Metric | Cell A (baseline) | Best so far | This run | Δ vs best | Δ vs A |
|---|---|---|---|---|---|
| acc | 0.84 | X | X | X | X |
| ... |

## 5. Visualization evidence
What the t-SNE / Grad-CAM / per-class CSV say.

## 6. Verdict
**Success** / **Partial** / **Failure** / **Noise** / **Bug**

## 7. Decision
Keep / discard / propagate to which downstream cells?

## 8. Next hypothesis
The single config delta you'll try in iter NNN+1.
```

---

## Budget

- Total iterations: **30** (HARD stop)
- Stop early if 3 consecutive Failures AND no Partial in last 5 iters.
- Stop early if test_acc ≥ 0.96.

## Verdict criteria

| Label | Criteria |
|---|---|
| **Success** | acc rises ≥ +0.5 pp vs current best AND mechanism evidence supports the hypothesis |
| **Partial** | mechanism fires but Δ small (0 to +0.5 pp) |
| **Failure** | mechanism doesn't fire OR acc drops > 0.5 pp on an expected-positive hypothesis |
| **Noise** | \|Δacc\| < 0.3 pp both ways — within seed variance |
| **Bug** | sanity baselines broken — halt and debug |

---

## HARD CONSTRAINTS — never violate

1. **Don't touch other projects** outside this repo's working tree.
2. **Don't modify the dataset** under `/data/cifar10/`.
3. **Don't edit `program.md`** — these rules are immutable to you.
4. **Don't delete prior iters' artifacts** under `runs/` or `logs/`.
5. **One experiment per loop tick.**
6. **No `--dangerously-skip-permissions` ever.**
7. **Git policy.** `bash scripts/git_iter_commit.sh <iter>` is auto-invoked at
   end of analyze. Branch = `autoresearch/iter-NNN` off main, auto-pushed,
   PR opened. You are FORBIDDEN from: `git push --force`, `git merge`,
   `git rebase`, modifying `main` directly, deleting branches.
8. **No module lifting.** You may NOT graft modules / losses / heads from
   published image-classification papers (cutout, mixup variants from
   specific papers, attention mechanisms beyond the bare ResNet, etc.).
   Original mechanisms only — derived from this loop's findings, ≤ 10 lines
   from elementary math.
9. **Stay inside the module catalog.** Toggle/sweep listed axes only. Adding
   a new axis (e.g. label smoothing, dropout, cutmix) requires a human
   editing `program.md` first.
