# Deep-dive report template

For Deep-dive analyses (per `SKILL.md` §Analysis depth), structure the
write-up using this template. Copy it to `logs/iteration_NNN.md` and
fill each section. The framework's `git_iter_commit.sh` parses headings
to build the per-iter commit message, so **keep the section numbers and
titles intact**.

```markdown
# Iteration NNN — {short_name}
Date: YYYY-MM-DD HH:MM | GPU: {id} | Duration: {h}

## 1. Hypothesis
Two or three sentences. State precisely:
- What issue you're trying to resolve.
- Which ablation cell you're targeting.
- The exact config delta vs the prior cell (single axis preferred).

Example: "Switch optimizer from SGD (lr=0.1, cosine) to AdamW (lr=1e-3,
cosine, weight_decay=0.05) while keeping `augmentation: standard` and
`epochs: 60`. Hypothesis: AdamW with appropriate weight decay matches
SGD on CIFAR-10 within ±0.5 pp test accuracy and shows faster early
convergence (epoch 5 acc ≥ 0.65 vs SGD's typical 0.55)."

## 2. Falsification criterion
What numeric or qualitative outcome would refute the hypothesis? Be
specific — "accuracy doesn't go up" is not falsifiable enough.

Example: "Falsified if (a) test_acc < SGD baseline by > 0.5 pp at epoch
60, OR (b) test_acc at epoch 5 is below 0.6 (slower than SGD), OR (c)
train loss at epoch 5 is higher than SGD's despite AdamW's reputation
for fast early convergence."

## 3. Changes made
The YAML diff and any code changes. Show the actual diff:

```diff
-training:
-  optimizer: sgd
-  lr: 0.1
-  momentum: 0.9
-  weight_decay: 5.0e-4
+training:
+  optimizer: adamw
+  lr: 1.0e-3
+  weight_decay: 0.05
   scheduler: cosine
   epochs: 60
```

If you edited `src/`, link to the commit / diff.

## 4. Results

Headline numbers in a comparison table:

| Metric | Baseline (cell A) | Best so far | This run | Δ vs best | Δ vs cell A |
|---|---|---|---|---|---|
| test_acc  | 0.____  | 0.____  | 0.____  | ±0.___ | ±0.___ |
| test_loss | _.___   | _.___   | _.___   | ±_.___ | ±_.___ |
| top5_acc  | 0.____  | 0.____  | 0.____  | ±0.___ | ±0.___ |
| train_acc | 0.____  | 0.____  | 0.____  | ±0.___ | ±0.___ |
| best_epoch | __     | __      | __      |        |        |

Per-class summary if available:

| Class | Baseline | This run | Δ |
|---|---|---|---|
| airplane    | 0.___ | 0.___ | ±0.___ |
| automobile  | 0.___ | 0.___ | ±0.___ |
| ...         | ...   | ...   | ...    |

Note: signals worth highlighting, e.g. "hard pair (cat/dog) gained +2.5
pp each while easy pair (truck/automobile) was within ±0.3 pp" — this
is the shape of the result, not the average.

## 5. Visualization evidence

For each viz produced (per `program.md` §Mandatory):

**Grad-CAM** (`figs/iter_NNN/cam.png`):
- One sentence on what changed in the attention pattern relative to the
  baseline. Did it move toward the object? Toward background?
- Highlight 1–2 specific samples where the attention shift is most
  visually clear.

**Feature t-SNE** (`figs/iter_NNN/tsne.png`):
- Did class clusters tighten? Spread? Did the cat/dog overlap shrink?
- Were there outlier samples that the baseline placed near the wrong
  cluster but the new run placed correctly?

**Per-class table** (`figs/iter_NNN/per_class.csv`):
- Mean per-class accuracy.
- Variance / spread (std across classes — narrowing is good).
- Worst class accuracy (a "weakest link" indicator).

## 6. Verdict
**Success** / **Partial** / **Failure** / **Noise** / **Bug**

One paragraph defending the choice. Cite the three axes from the SKILL:
- Did the targeted issue resolve? (yes / partially / no)
- Which direction did the metric move? (up / flat / down)
- Did the predicted mechanism appear in the visualizations? (yes / no /
  unclear)

Combine these per the SKILL's verdict-labels table.

Confidence level (one of: high / medium / low) with reason.

## 7. Decision

Based on the verdict, what to do next:

- **Keep the change?** (commit to the per-iter branch and propagate to
  downstream cells, OR discard and revert)
- **Which downstream cells does this affect?** (e.g. "if AdamW Success,
  drop SGD from cells E and F as well")
- **What follow-up is needed?** (a 2nd-seed replay, a wider sweep, a
  different visualization)

## 8. Next hypothesis

The single config delta you'll try in iter NNN+1. Be as concrete as in
§1 — another loop tick will read this and act on it.

Example: "Sweep AdamW weight decay {0.01, 0.05, 0.1, 0.2}. Hypothesis:
optimum sits at 0.05 (current setting); going up reduces overfitting
but costs raw acc, going down restores SGD-like train-test gap.
Falsifies if 0.1 gives strictly higher test_acc than 0.05 by > 0.3 pp."
```

## How the framework uses this report

1. The agent's analyze step writes this file directly to
   `logs/iteration_NNN.md`.
2. `scripts/git_iter_commit.sh` parses §1, §4, §6, §7 to build the
   commit message + PR description.
3. `scripts/parse_consensus.py` parses §6 (verdict) and §8 (next-step)
   to build the consensus output.
4. The propose phase of the next loop tick reads §8 verbatim as the
   binding hypothesis for the next experiment.

If you change the section structure, update those three scripts in
lockstep — otherwise the agent will silently produce malformed reports.

## Length guidance

| Section | Quick | Standard | Deep dive |
|---|---|---|---|
| §1 Hypothesis             | 1 sentence | 2-3 sentences  | 1 paragraph + falsification |
| §2 Falsification          | (skip)     | 1 sentence     | 1 paragraph |
| §3 Changes                | 1 line     | YAML diff      | YAML + code diff + rationale |
| §4 Results table          | 1 row      | full table     | full + per-class |
| §5 Visualizations         | (skip)     | 1 sentence × 2 | 1 paragraph × 3 |
| §6 Verdict                | 1 sentence | 1 paragraph    | 1-2 paragraphs + confidence |
| §7 Decision               | (skip)     | bullet list    | full reasoning |
| §8 Next hypothesis        | (skip)     | 1 sentence     | sharpened, ready-to-execute |

The agent should match the depth to the user's request. Don't pad
sections when the data doesn't support the depth.
