---
name: experiment-analysis
description: Structured, hypothesis-driven analysis of experiment results. Use this skill whenever the user finishes a training run and asks to analyze, interpret, compare, or write up results — including when they mention accuracy changes, ablation results, Grad-CAM / feature visualizations, confusion matrices, regression cases, comparing to baseline, or deciding whether a code change "worked". Also trigger on requests to write an experiment report, debug why a change made things worse, or decide next steps after seeing numbers. Trigger proactively even when the user just says "results are in, what do you think?" — don't just dump metrics back, apply this protocol.
---

# Experiment analysis

This skill governs how to analyze experiment results after a code change. The
goal is not to decide "did the number go up?" but **"did we fix the issue we
were trying to fix, and what do the results tell us about next steps?"**

The methodology below is task-agnostic. Worked examples reference the bundled
CIFAR-10 + ResNet-34 demo (test accuracy, per-class breakdown, Grad-CAM,
feature t-SNE). Adapt the examples to your project's metrics and viz outputs.

## Core principle — the non-negotiable one

**A drop in the headline metric is not automatically a failure.** If the
change was designed to fix a specific issue (e.g. "model overfits to a few
common classes", "Grad-CAM focuses on background rather than the object"),
and the qualitative evidence shows the issue is resolved, the run is a
partial or full success even with lower raw accuracy.

Why: accuracy is a compound metric entangling many things. A change that
correctly redirects attention to the object but hasn't yet learned the new
representation well will look bad on accuracy but good on Grad-CAM. Dropping
that change on accuracy alone throws away a partially-working correct
direction.

**Conversely, accuracy going up is not automatically a success.** If the
issue isn't resolved and accuracy went up for unrelated reasons (training
longer, lucky seed, test-set leakage through aug), the run doesn't validate
the hypothesis.

Judge by three axes, all together: **(issue resolved?) × (metric direction)
× (expected mechanism observed?)** — not one alone. The `## Verdict labels`
section below defines the combinations.

## Minimum inputs

For a strong verdict, gather as many of the following as possible:

- Dataset and split (e.g. "CIFAR-10 default test split").
- Baseline run's name and headline metric.
- New run's name and headline metric.
- Whether results are single-seed or multi-seed (with std if multi).
- Per-class accuracy breakdown if available.
- Ablation knob that changed (e.g. `augmentation: standard → autoaugment`,
  `optimizer: sgd → adamw`).
- Visualization evidence if the hypothesis is mechanistic (Grad-CAM,
  feature t-SNE, confusion matrix).
- Per-sample predictions from both runs if quadrant regression analysis is
  needed.

**Missing-data principle** (applies to every step): if required inputs are
absent, do not stop by default. Give a provisional analysis, explicitly
state which evidence is missing, and lower the confidence level of the
verdict accordingly. Ask a follow-up question only when the missing
information would materially change the verdict (e.g. you genuinely cannot
tell success from noise without a second seed or per-class breakdown).

When a step requires data the user doesn't have, **give them the command to
produce it** rather than approximating:

> I can give a provisional verdict from the headline accuracy, but the
> regression case analysis needs sample-level predictions from both runs.
> Dump them with: `torch.save({'y_true': ..., 'y_base': ..., 'y_new': ...,
> 'paths': ...}, 'preds.pt')`. Once that's available I can run the full
> quadrant analysis.

## Analysis depth

Pick the depth based on the user's request and available evidence:

- **Quick review** — user asks "how does this look?" with just headline
  numbers. Minimum output (see `## Output structure`).
- **Standard analysis** — user wants to understand the result properly.
  Output: hypothesis reconstruction, accuracy table with per-class patterns,
  2–3 key visualizations tied back to hypothesis, caveats, three-part
  verdict with confidence.
- **Deep dive** — user is writing up the experiment or making a decision
  that matters. Output: full protocol (Steps 0–4 below) + formal report
  from `report-template.md`.

Default to **Standard** unless the user signals otherwise ("just a quick
check" → Quick; "write up the report" → Deep dive). Follow the protocol in
order; deviate only when explicitly asked for a shorter or narrower
analysis.

## Output structure

Every analysis, regardless of depth, must at minimum contain these five
blocks in this order:

1. **Reconstructed hypothesis** — what the change was supposed to fix, one
   or two sentences.
2. **Headline verdict** — one sentence stating the label (from `## Verdict
   labels`) and direction.
3. **Evidence for the verdict** — keyed to the three axes (issue resolved /
   accuracy / mechanism); length scales with depth.
4. **Missing evidence and confidence** — what's not available, resulting
   confidence level (from `## Confidence`).
5. **Next step** — one concrete action.

Quick review can do this in ~150 words. Standard expands block 3 with
tables and per-class patterns. Deep dive expands all blocks per the
`report-template.md` structure.

## The analysis protocol

### Step 0 — reconstruct the hypothesis before interpreting results

Before analyzing the numbers, pin down:

1. What issue the change was supposed to fix.
2. What evidence should appear if it works (quantitative + qualitative,
   with specific subsets or classes if possible).
3. What result would falsify the hypothesis.

If the user did not state these clearly, reconstruct them from motivation +
code change, state your assumptions explicitly, and proceed with a
provisional analysis (per the Missing-data principle above).

Examples of good predictions (CIFAR-10):

- "Switched augmentation `standard → autoaugment` → test accuracy should
  rise by ~1–2 pp; per-class accuracy on hard classes (cat/dog/bird) should
  rise faster than easy classes (truck/ship); train-test acc gap should
  shrink."
- "Switched optimizer `sgd → adamw` with same effective LR → test acc
  should be within ±0.5 pp; loss curve should converge faster early but may
  generalize slightly worse late; weight norms should grow slower."
- "Replaced random-crop augmentation with cutmix → Grad-CAM should localize
  more diffusely (less single-region peak); per-class accuracy on
  classes-with-distinctive-parts (e.g. truck wheels) should drop slightly,
  while accuracy on full-object classes should rise."

Reject vague predictions and sharpen them:

- "Accuracy should go up." → ask: by how much, on which classes, driven by
  reduced train-test gap or by raw representation gain.
- "The model should be better." → unfalsifiable, reject.
- "Augmentation helps." → on which classes, by how much, with what
  trade-off in train accuracy.

### Step 1 — accuracy-level analysis

Don't stop at "test_acc went from 0.823 to 0.851". Do all of these when
data allows:

**1a. Headline metric in context.**

```
              Baseline    New        Δ
test_acc      0.____      0.____     +/-_.___
test_loss     _.___       _.___      +/-_.___
top5_acc      0.____      0.____     +/-_.___
train_acc     0.____      0.____     +/-_.___   (overfitting check)
```

**1b. Direction check.** Is the Δ consistent with the prediction from Step
0? "Accuracy went up" doesn't count as consistent if you predicted "test
acc up driven by hard classes" and saw "test acc up driven by an extra
+5pp on the already-best class" — that's a different mechanism firing.

**1c. Train-test gap.** If train acc rose more than test acc, the change
*overfits* even when it improves headline accuracy. Common with longer
training, weaker weight decay, removed augmentation.

**1d. Per-class breakdown.** Average accuracy hides the signal. Ask:

- Which classes improved? Which regressed?
- Is the set of improved classes the one the hypothesis predicted (e.g.
  classes where the issue was strongest)?
- Did per-class variance narrow or widen?

For CIFAR-10, `cat` and `dog` are the canonical hard pair (visually
similar); `truck` and `automobile` are another. Improvements on hard
classes are higher-signal than improvements on easy classes.

**1e. Seed variance.** Treat single-seed gains as suggestive rather than
confirmatory unless the gain is clearly larger than the run-to-run
variance historically observed for that setup. As a rough heuristic, gains
under ~0.3 pp on CIFAR-10 fall within seed noise for a 60-epoch ResNet-34;
request ≥3 seeds when the decision matters.

**1f. Sanity baselines.** Did the change break any baseline it shouldn't
have? E.g. random-init logit distribution should still be ~0.1 per class on
init; an immediate train acc of 0.95 at epoch 0 means data leak.

### Step 2 — visualization-level analysis

**Visualizations are supporting evidence, not proof of causal mechanism.**
Grad-CAM heatmaps can "look right" without being the actual driver of the
prediction. Treat them as necessary-but-not-sufficient evidence: if they
contradict the hypothesis, the hypothesis is weakened; if they support it,
combine with quantitative evidence before declaring mechanism confirmed.

For each visualization type available:

**Grad-CAM** (`scripts/visualize_cam.py`):

- Does the heatmap localize on the object the class is named after, or on
  background / spurious cues (e.g. snow for "ship")?
- Are correctly-classified samples and incorrectly-classified samples
  showing different attention patterns?
- Did the post-change attention shift in the predicted direction (e.g. away
  from background toward object)?

**Feature t-SNE** (`scripts/visualize_tsne.py`):

- Did class clusters tighten? Spread?
- Are pairs the model historically confuses (cat/dog, ship/airplane in
  CIFAR-10) more or less overlapped?
- Are outliers (samples far from their cluster center) the ones the model
  misclassifies?

**Confusion matrix** (if generated):

- Where do off-diagonal mass concentrate? Are they the predicted hard pairs?
- Did any specific class become a "dumping ground" for predictions (large
  off-diagonal column)? That's a class-collapse symptom.

### Step 3 — regression case analysis

Aggregate metrics hide the structure of *which* samples got better and
*which* got worse. Run the four-quadrant analysis when sample-level
predictions exist (see `references/regression-analysis.md` for the full
recipe).

The key question: of the samples the new run got *wrong*, were any of them
*right* in the baseline? If yes, those are regressions — the change moved
backwards on that subset. The size of the regression bucket is often the
most decision-relevant number you can compute.

### Step 4 — write the report

For Quick and Standard analyses, follow `## Output structure` directly.
For Deep dive write-ups, expand the same structure into the full report
format defined in `references/report-template.md`.

## Verdict labels

Three axes, eight combinations. The label is the verdict you write in the
report's §6.

| Issue resolved? | Metric direction | Mechanism observed | Verdict |
|---|---|---|---|
| ✓ | up    | ✓ | **Success** — strongest possible signal |
| ✓ | flat  | ✓ | **Partial success** — mechanism works, gains too small |
| ✓ | down  | ✓ | **Partial success** — direction correct but cost; investigate trade-off |
| ✗ | up    | ✓ | **Confounded success** — gain is real but driven by something other than the targeted issue. *Do not* propagate the hypothesis without further investigation |
| ✗ | up    | ✗ | **Confounded gain** — likely shortcut / overfitting / lucky seed |
| ✗ | flat  | ✗ | **Failure** — hypothesis falsified |
| ✗ | down  | ✗ | **Failure** — hypothesis falsified |
| any | bug   | any | **Bug** — sanity baseline broke; halt and debug before more iteration |

A "bug" finding short-circuits the rest. Common triggers: random-init logits
too uniform, train acc plateauing at chance, optimizer LR exploding, NaN
loss in the first few hundred steps.

## Confidence

Append to every verdict:

- **High confidence** — multi-seed agreement; mechanism evidence consistent
  with metric direction; per-class story matches prediction.
- **Medium confidence** — single seed, but mechanism + metric agree;
  per-class story available and consistent.
- **Low confidence** — single seed; mechanism and metric agree but per-class
  story missing or noisy; OR strong metric without mechanism evidence.

Lower confidence by one tier if any "Bug"-adjacent symptom is present.

## Things that look like signal but usually aren't

- **Sub-seed-noise gains** on the headline metric. CIFAR-10 ResNet-34
  single-seed runs typically vary ±0.3 pp test_acc on rerun; treat smaller
  gains as noise unless backed by a clean per-class story.
- **Train accuracy improvements without test improvements.** That's the
  definition of overfitting; not interesting unless intentional (e.g.
  testing a regularizer).
- **Loss going down without accuracy going up.** Usually means the model
  became more confident on already-correct samples and more confident on
  already-wrong ones. Look at the calibration curve / ECE.
- **Big gains in the first few epochs.** Most "fast convergence" claims
  evaporate by epoch 30. Compare at the same training budget.

## Required reads (when this skill triggers)

- `references/failure-diagnostics.md` — common image-classification failure
  modes and what they look like in metrics + visualizations.
- `references/regression-analysis.md` — the four-quadrant recipe.
- `references/report-template.md` — the deep-dive format.
- The user's task-background skill (if one exists in `.claude/skills/`) —
  cross-check field-specific conventions against the methodology above.

## Boundary

Don't apply this skill when:

- The user explicitly asks for raw numbers without analysis.
- The "experiment" is a refactor or pipeline change that wasn't expected to
  affect metrics — there's no hypothesis to verify.
- The user is mid-debugging and just wants help finding a bug, not a
  verdict on results.
