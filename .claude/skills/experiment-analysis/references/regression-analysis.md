# Regression-case analysis — the four-quadrant recipe

Aggregate metrics hide *which* samples improved and *which* got worse. The
four-quadrant analysis shows you the structure: of the samples the new
run got wrong, were any of them right in the baseline? Of the samples the
new run got right, were any of them already right in the baseline (so the
"win" is double-counted)?

This recipe assumes single-label classification (CIFAR-10 demo style).
Adapt the prediction-correctness check for other tasks.

## Inputs needed

For each test sample, both runs' predictions:

```python
import torch
saved = {
    "y_true": y_test,            # (N,)  ground-truth labels
    "y_base": baseline_preds,    # (N,)  argmax predictions of baseline
    "y_new":  new_preds,         # (N,)  argmax predictions of new run
    "paths":  test_image_paths,  # (N,)  identifiers, for spot-checking
}
torch.save(saved, "preds.pt")
```

If you don't have these saved, dump them once with a small modification
to `test.py` (or with this snippet inside the eval loop):

```python
all_preds, all_paths = [], []
for x, y, paths in test_loader:           # paths optional
    logits = model(x.to(device))
    all_preds.append(logits.argmax(1).cpu())
    all_paths.extend(paths)
preds = torch.cat(all_preds)
torch.save({"y_true": ..., "y_pred": preds, "paths": all_paths},
           f"runs/<exp>/preds.pt")
```

## The four quadrants

For each test sample, compute:

```
correct_base = (y_base == y_true)
correct_new  = (y_new  == y_true)
```

This gives four buckets:

| Bucket | base | new | Meaning |
|---|---|---|---|
| **TT** | ✓ | ✓ | Both right — neutral; not informative for verdict |
| **TF** | ✓ | ✗ | **Regression** — the change broke previously-correct predictions |
| **FT** | ✗ | ✓ | **Improvement** — the change fixed previously-wrong predictions |
| **FF** | ✗ | ✗ | Both wrong — also not informative for verdict |

The headline accuracy delta is `|FT| - |TF|` divided by N. But that net
number erases the structure of the change.

## What to compute

Print the four counts:

```python
TT = ((y_base == y_true) & (y_new == y_true)).sum().item()
TF = ((y_base == y_true) & (y_new != y_true)).sum().item()
FT = ((y_base != y_true) & (y_new == y_true)).sum().item()
FF = ((y_base != y_true) & (y_new != y_true)).sum().item()
print(f"TT={TT} (both right) | TF={TF} (regression) | "
      f"FT={FT} (improvement) | FF={FF} (both wrong)")
print(f"Net Δ accuracy = ({FT} - {TF}) / {TT+TF+FT+FF} "
      f"= {(FT-TF)/(TT+TF+FT+FF):+.4f}")
```

## Reading the buckets

**Healthy improvement.** FT >> TF, with the FT samples concentrated in
classes the hypothesis predicted (e.g. for a "fix overfitting"
hypothesis, hard classes like cat/dog should dominate FT). TF should be
scattered, not concentrated in any one class.

**Confounded gain.** FT and TF both large (e.g. FT=300, TF=200, net +1
pp). The change is a *re-shuffle* — it's gaining on some samples by
losing others. Net positive but the mechanism may not be what you think;
the hypothesis likely needs sharpening.

**Trade-off.** FT large in predicted-improvement classes, TF large but
concentrated in a different class. This is often the right outcome for a
targeted fix — you traded easy-class accuracy for hard-class accuracy.
Whether it's a Success or Partial depends on whether that trade was the
intended outcome.

**Hidden regression.** FT and TF are similar in size (a few samples each)
but TF is concentrated in classes you care about. Net acc looks neutral
but the change has hurt the high-priority subset.

## Per-class breakdown of TF / FT

Beyond raw counts, ask which classes the regressions came from:

```python
import numpy as np
y_true_np = y_true.numpy()
tf_mask = (y_base == y_true) & (y_new != y_true)
ft_mask = (y_base != y_true) & (y_new == y_true)

print("Regressions per class:")
for c in range(num_classes):
    print(f"  class {c} ({CLASSES[c]:>10}): "
          f"TF={tf_mask[y_true_np == c].sum()}  "
          f"FT={ft_mask[y_true_np == c].sum()}")
```

The expected pattern from the hypothesis (Step 0) should match the
actual pattern of FT — otherwise the change worked, but for different
reasons than predicted.

## Spot-checking individual cases

Pick 4–8 samples from TF (the regressions) and run Grad-CAM on both the
baseline and new model. Look for:

- Did the attention shift to a different region? Where?
- Did the shifted region include a non-class-distinctive feature (e.g.
  shifted from cat's face to cat's tail when distinguishing cat vs dog)?
- Is the regressed sample's true class hard for the new model in
  general, or is this an isolated instance?

Pick another 4–8 from FT (the improvements). The ideal: the new
attention sits on the class-defining region in a way the baseline's
didn't.

## When this analysis matters most

- The headline accuracy delta is small (|Δ| < 1 pp) and you're trying to
  decide if it's signal or noise.
- The hypothesis predicted a *structured* improvement (specific classes
  / specific failure modes) — quadrant analysis tells you whether the
  prediction was right.
- You're considering propagating the change to downstream cells of the
  ablation matrix and want to be sure it isn't a re-shuffle.

## When to skip

- Headline gain is huge (>5 pp) AND clearly localized to expected
  classes.
- You're at the smoketest / sanity stage — no need for fine-grained
  attribution yet.
- You don't have sample-level predictions and the cost of producing them
  exceeds the value of the analysis.
