# Regression case analysis — deep dive

When accuracy drops, don't panic. Don't scrap the change. Do this protocol carefully.

## Why this matters

The main SKILL says: "a drop in accuracy is not automatically a failure." This file is the how. You need to go from aggregate drop → concrete case-level evidence → informed verdict.

## Precondition — you need per-sample predictions

This analysis requires per-sample predictions from both baseline and new runs. **Do not approximate it from per-class numbers** — the point is to see individual cases that changed direction, and per-class aggregates destroy that signal.

If the user only has summary metrics, stop here and give them the dump command:

```python
# Run this at the end of eval for both baseline and new checkpoints
import torch

dump = {
    'y_true': y_true.cpu(),              # (N,) int64 ground truth class ids
    'y_pred': y_pred.cpu(),              # (N,) int64 predicted class ids
    'logits': logits.cpu(),              # (N, C) float, optional but useful
    'paths':  image_paths,               # list of N strings, needed to pull images for Step 3b
    'class_names': class_names,          # list of C strings
}
torch.save(dump, f'preds_{run_name}.pt')
```

Then they can resume with `preds_baseline.pt` and `preds_new.pt`.

In the meantime, offer a **per-class regression sketch** as the interim step (count how many classes improved vs regressed, and how much per-class U variance changed) — but explicitly note this is not a substitute for quadrant analysis, it's a stopgap.

## The quadrant framing

Partition the test set into four buckets:

|  | baseline correct | baseline wrong |
|---|---|---|
| **new correct** | ① stable correct | ② improvements |
| **new wrong** | ③ regressions | ④ stable wrong |

Counts: `|①| + |②| + |③| + |④| = N_test`

- Net accuracy change = `(|②| − |③|) / N_test`
- But this number doesn't tell you WHY.
- The story is in ② (what the change fixes) and ③ (what the change breaks).

## Building the quadrants

```python
import numpy as np
import torch

base = torch.load('preds_baseline.pt')
new  = torch.load('preds_new.pt')

y_true = base['y_true'].numpy()
y_base = base['y_pred'].numpy()
y_new  = new['y_pred'].numpy()

def quadrant(y_true, y_b, y_n):
    bc = (y_b == y_true)
    nc = (y_n == y_true)
    return (
         bc &  nc,    # q1 stable correct
        ~bc &  nc,    # q2 improvements
         bc & ~nc,    # q3 regressions ← focus here
        ~bc & ~nc,    # q4 stable wrong
    )

q1, q2, q3, q4 = quadrant(y_true, y_base, y_new)
print(f"stable correct: {q1.sum()}")
print(f"improvements:   {q2.sum()}")
print(f"regressions:    {q3.sum()}")
print(f"stable wrong:   {q4.sum()}")
net = q2.sum() - q3.sum()
print(f"net change: {net:+d} ({net / len(y_true) * 100:+.2f}%)")
```

## Sampling strategy

Sample 10–20 from ③ (or all of them if the pool is smaller than 20). Don't inspect all of ③ when it has hundreds — you'll drown.

**Stratified by class.** If regressions spread across many classes, 2–3 per class. If they cluster in a few, sample more from those.

```python
import pandas as pd

df = pd.DataFrame({
    'idx':        np.where(q3)[0],
    'true_class': y_true[q3],
    'base_pred':  y_base[q3],
    'new_pred':   y_new[q3],
})
print(df.groupby('true_class').size().sort_values(ascending=False).head(20))
```

**Stratified by confidence change.** If you have logits, cases where the new model is confidently wrong (and baseline was right) are most informative — something specific shifted the decision.

```python
# Assuming logits available
base_logits = base['logits'].numpy()
new_logits  = new['logits'].numpy()

new_conf_on_wrong = new_logits[q3].max(axis=1)
base_conf_on_true = base_logits[q3, y_true[q3]]

# High-confidence regressions
high_conf_reg = (new_conf_on_wrong > np.percentile(new_conf_on_wrong, 75)) & \
                (base_conf_on_true > np.percentile(base_conf_on_true, 50))
```

## Per-case analysis template

For each sampled regression case, fill this in:

```
Case #__  (sample_idx=____, path=____)

Ground truth:      <class name>
Baseline pred:     <class>    (correct / close miss / far miss)
New pred:          <class>    (close miss / far miss / apparently random)

Image (one sentence):  <what's in it, what's hard>

Baseline attention:   <where was it looking?>
New attention:        <where is it looking?>

Hypothesized cause:
  (a) intended tradeoff  — change shifted focus and this case paid the cost
  (b) unrelated side-effect — change broke something it shouldn't
  (c) noise / seed      — no clear mechanism
  (d) test-set edge     — baseline got lucky; the case is inherently ambiguous

Confidence in classification: high / medium / low
```

## Grouping into causes

After 10–20 cases (or whatever fewer cases you had), group:

| Cause | Count | Example cases |
|---|---|---|
| (a) intended tradeoff | __ | #3, #7, #11, ... |
| (b) unrelated side-effect | __ | #2, #5, ... |
| (c) noise | __ | #9, ... |
| (d) test-set edge | __ | #14, ... |

**Interpretation rules:**

- **(a) dominant (≥60% of inspected regressions):** change works as designed. Regressions are the expected cost. Next iteration should add back what the change removed (e.g., global context). **This is the Partial Success verdict — document as such, don't downgrade to Failure.**
- **(b) dominant:** change has an unintended side-effect. Isolate it next iteration.
- **(c) dominant:** variance. Run more seeds.
- **(d) dominant:** test-set quirks, not a problem with the change per se.
- **No dominant cause:** change is doing several things at once. Simplify and re-test.

## Don't skip improvements (②)

It's tempting to only dissect ③. But the improvements tell you whether the mechanism is firing as expected.

For each sampled improvement case:

- Was baseline failing here for the reason you hypothesized?
- Is new succeeding via the intended mechanism, or by luck?

If improvements are uniformly distributed across target-class AND non-target-class cases, the mechanism isn't the reason for the gain — something else is. **This matters**: a gain that's not explained by the intended mechanism is fragile and doesn't validate the hypothesis.

## Writing it up

Keep the regression analysis section concrete:

```
## Regression case analysis

Quadrants (CUB GZSL test set, seed 0):
  stable correct:  1,842 (61.4%)
  improvements:       89 ( 3.0%)
  regressions:       147 ( 4.9%)
  stable wrong:      923 (30.8%)
  net change: −58 samples (−1.9%)

Sampled 20 regression cases, stratified by class.
- 13 / 20 show attention shifted from background grass patch (which baseline
  used as a shortcut for ground-nesting birds) to the bird itself, but the
  bird features are not yet well-learned for these classes.
- 4 / 20 involve partial occlusion; the new model loses global context while
  baseline used scene context to recover.
- 3 / 20 show no clear pattern; likely noise.

Interpretation: 17/20 regressions are (a) intended-tradeoff cases. The change
is working as designed. Next iteration: either extend training so local
features mature, or re-introduce global context via a dual-stream design.
```

## Common mistakes

- **Sorting by per-class accuracy drop and looking only at top-3 classes.** Misses patterns distributed across many classes.
- **Looking at regressions without improvements.** You must compare against improvement cases. The story lives in both.
- **Claiming "it's a tradeoff" without evidence.** Tradeoffs are real but need per-case evidence, not hand-waving.
- **Not grouping by cause.** 15 scattered reasons is a list, not analysis. Group into 3–5 causes.
- **Over-interpreting small quadrant counts.** If |③| = 12 on a 3000-sample test set, inspecting 5 of them is closer to reading tea leaves than analysis. More seeds are more informative than more case-picking.
