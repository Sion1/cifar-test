# Failure diagnostics cheat sheet

Quick lookup: given a failure mode (what's wrong), which visualization/metric confirms or denies it? Use this when analyzing results, to match what you see in the numbers/images to a named failure mode, and to know what additional evidence to collect.

Cross-reference: the three canonical GZSL failure modes are defined in the `gzsl` skill's §"Three canonical failure modes". This file is the operational extension — how to detect them from experiment outputs.

Remember: visualizations are supporting evidence, not causal proof. Use them in combination with quantitative metrics, not in place of them.

---

## Seen-class bias

**Symptom in numbers:**
- S >> U (e.g., S=70, U=15, H=24.6)
- Large gap between CZSL accuracy on unseen and GZSL U on unseen
- γ sweep curve heavily right-skewed; optimal H requires γ far from 0

**Confirmatory visualizations:**
- **Confusion matrix (unseen block):** rows = unseen classes, cols = all classes. Seen-class columns show vertical stripes — unseen samples being dumped into specific seen classes.
- **Prediction distribution histogram:** for each unseen test sample, which class was predicted? Histogram concentrated on seen-class ids = seen-bias.
- **γ sweep curve:** H vs γ. Bias-dominated curves peak at large γ; well-calibrated methods peak near γ=0.

**Refuting visualizations:**
- Unseen predictions distributed roughly uniformly across classes (not concentrated on seen)
- γ sweep peaks near 0

**When the change should reduce bias, expect:**
- S drops, U rises, H rises
- Optimal γ shifts toward 0
- Confusion matrix seen-column stripes diminish

---

## Hubness

**Symptom in numbers:**
- Per-class U has extreme variance (some classes at 0%, some at 60%+)
- Top-1 U much worse than top-5 U
- Unseen predictions concentrated on a few **unseen** classes (different from seen-bias — concentration is within the unseen set)

**Confirmatory visualizations:**
- **k-occurrence distribution:** for each class `c`, count N_k(c) = number of samples for which `c` is in the top-k nearest neighbors. Heavy right tail (a few classes with huge N_k) = hubness. Skewness > 5 is suggestive, > 10 is severe.
- **Per-class U sorted bar chart:** right tail crashes to zero for many classes while top ~20% hold most of the accuracy → hubness.
- **CSLS / normalized similarity comparison:** switching from raw cosine to CSLS (cross-domain local scaling) giving large U improvement → hubness.

**Refuting visualizations:**
- k-occurrence roughly uniform
- Per-class U roughly flat, no extreme zeros

**When the change should reduce hubness, expect:**
- k-occurrence skewness drops
- Per-class U variance drops
- Gap between top-1 and top-5 U narrows

---

## Projection domain shift

**Symptom in numbers:**
- CZSL accuracy on unseen also low (not just GZSL — rules out pure calibration)
- Unseen visual features project closer to wrong (seen) prototypes than to true semantic prototype
- Large gap between train accuracy on seen and test accuracy on unseen even in CZSL

**Confirmatory visualizations:**
- **Distance-to-prototype histograms:** for unseen test samples, plot two histograms — (a) distance from projected visual feature to TRUE semantic prototype, (b) distance to NEAREST seen prototype. If (b) < (a) for most samples → projection shift.
- **t-SNE of projected visual features:** color by ground-truth class. Unseen clusters collapsed together or sprinkled inside seen clusters → projection not separating them.
- **Train/test accuracy gap on seen:** if seen test accuracy also much lower than seen train, projection is noisy/overfit.

**Refuting visualizations:**
- Distance histograms: (a) < (b) for most samples
- t-SNE: unseen classes form distinct, compact clusters

**When the change should reduce projection shift, expect:**
- Distance to true prototype shrinks relative to distance to nearest seen
- t-SNE clusters for unseen become compact and separated
- CZSL unseen accuracy rises (not just GZSL U)

---

## Attention / spatial shortcut

Not one of the three canonical GZSL failure modes, but common in practice (and the motivation for HySyn-v3-style attention interventions).

**Symptom in numbers:**
- Per-class U unbalanced based on where the object sits in the image (low U on off-center classes)
- High sensitivity to random/off-center crops

**Confirmatory visualizations:**
- **Grad-CAM / attention maps:** high-weight pixels vs ground-truth bbox IoU. Average IoU < 0.2 is strong shortcut signal.
- **Per-class attention heatmap average:** averaged across a class's samples, attention always in image center regardless of actual object position → center-prior shortcut.

**Refuting visualizations:**
- Attention consistently overlaps object across varied bbox positions
- Per-class U not correlated with object position statistics

**When the change should fix attention, expect:**
- Grad-CAM center shifts toward bbox
- Per-class U on off-center classes rises
- Crop augmentation sensitivity drops

---

## When multiple failures co-exist

Most real experiments have 2+ modes interacting. Priority for analysis:

1. **Seen-bias** usually dominates GZSL — check first (γ sweep is cheap).
2. **Projection shift** next, especially on fine-grained datasets like CUB.
3. **Hubness** tends to be severe in semantic-embedding methods, milder in visual or latent.
4. **Shortcut learning** is dataset-specific — check when per-subset patterns look suspicious.

A single experiment typically addresses one mode. Mixed results often mean one mode improved while another worsened. Per-class + quadrant analysis pulls them apart.

---

## Quick decision tree

```
Is S >> U (gap > 10 points)?
├─ yes → seen-bias likely dominant; check γ sweep
└─ no  → probably not pure bias

Is U variance across unseen classes extreme (some 0%, some 60%+)?
├─ yes → check hubness (k-occurrence distribution)
└─ no  → probably not hubness

Is CZSL unseen accuracy also low?
├─ yes → projection shift (not just bias); check distance histograms
└─ no  → issue likely at the seen/unseen boundary, not in the projection

Is per-class U correlated with a spatial / visual property?
├─ yes → shortcut learning; check attention maps
└─ no  → probably not a shortcut issue
```

Use this tree to narrow where to spend time, not as a replacement for looking at the data directly.
