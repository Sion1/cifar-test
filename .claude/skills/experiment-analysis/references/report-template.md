# Experiment report template

## How to use

Copy the template below into your experiment directory as `REPORT.md`. Fill in each section. The filled example after the template shows what "good" looks like. Don't skip sections — write "not applicable" and one sentence why if a section doesn't fit. Skipped sections hide bugs.

The verdict label is one of five, defined in the main SKILL: Success / Partial success / Failure / Noise / Bug. The confidence tag is one of three: High / Medium / Low, also defined in the main SKILL.

---

## Template

```markdown
# Experiment report — <short descriptive title>

**Date:** YYYY-MM-DD
**Experiment dir:** experiments/YYYY-MM-DD-name/
**Baseline:** <method + commit hash / paper>
**Seeds:** [0, 1, 2]  or  "single seed — see §3.3"
**Verdict label:** Success / Partial success / Failure / Noise / Bug
**Confidence:** High / Medium / Low

## 1. Hypothesis

**Issue targeted (from prior analysis):**
<one paragraph — what did we diagnose, where is the prior analysis documented?>

**Code change made:**
<one paragraph — what was added/removed/modified, one sentence + commit hash>

**Predicted effect if the change works:**
- Quantitative: <e.g., "H rises 2+ points on CUB, driven by U rising; S may drop 1–2">
- Qualitative: <e.g., "Grad-CAM center shifts from background to object bbox on off-center cases">
- On which subset: <e.g., "improvement concentrated on off-center-object classes (list)">

**What would falsify:**
<If we see ____, the change does not work as claimed.>

## 2. Headline

**One-line verdict:** <e.g., "Partial success at Low confidence — H rose 1.0 on CUB and mechanism confirmed, but gain within single-seed noise band">

| Metric | Baseline | New   | Δ    |
|--------|----------|-------|------|
| S      | __.__    | __.__ | ±__  |
| U      | __.__    | __.__ | ±__  |
| H      | __.__    | __.__ | ±__  |

**Caveats:**
- <e.g., "Single seed; Δ comparable to historical variance">
- <e.g., "Off-center class list was derived from the same diagnostic run that motivated the change — held-out confirmation pending">

## 3. Accuracy analysis

### 3.1 Full table

<expanded table; possibly with γ-sweep best-H column, AUSUC, or multiple datasets>

### 3.2 Per-class patterns

<1–2 paragraphs: which classes improved, which regressed? Consistent with hypothesis?>
<Optional: small table of top-5 improved and top-5 regressed classes>

### 3.3 Variance & sanity

- Seeds: <report>
- Sanity baselines unchanged? <yes/no, with values>
- Calibration curve: <shifted / flattened / reshaped / unchanged>

## 4. Visualization analysis

Note: visualizations are supporting evidence, not causal proof. Combine with case analysis in §5.

### 4.1 <Visualization type 1, e.g., Grad-CAM>

<side-by-side figure: baseline vs new, on 4–6 representative images>
<1–2 paragraphs: does it match the predicted qualitative effect from §1?>

### 4.2 <Visualization type 2, e.g., t-SNE>

<same pattern: figure + interpretation tied to hypothesis>

### 4.3 <Visualization type 3, e.g., confusion matrix delta>

<same>

## 5. Case analysis

### 5.1 Quadrant counts

| | baseline correct | baseline wrong |
|---|---|---|
| **new correct** | __ (__%) | __ (__%) |
| **new wrong** | __ (__%) | __ (__%) |

Net change: __ samples (__%)

### 5.2 Improvement cases (sample)

| # | class | baseline | new | baseline attn | new attn | why did new succeed? |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

### 5.3 Regression cases (sample)

Same structure. Plus:

| Cause | Count | Note |
|---|---|---|
| (a) intended tradeoff | | |
| (b) unrelated side-effect | | |
| (c) noise / seed | | |
| (d) test-set edge | | |

## 6. Verdict

Three-part judgment:
- **Issue resolved?** yes / partially / no — evidence: <which section>
- **Accuracy direction:** up / flat / down — <Δ on H>
- **Expected mechanism observed?** yes / partially / no — evidence: <which section>

**Label:** <Success | Partial success | Failure | Noise | Bug>
**Confidence:** <High | Medium | Low>

**Reasoning:** <one paragraph applying the label definitions to the evidence above>

**What would upgrade or downgrade the verdict:**
<e.g., "Would upgrade to Success (Medium→High confidence) if seeds 1 and 2 show the same +1.0 H delta. Would downgrade to Noise if a second seed shows no movement on H.">

## 7. Next steps

- If Success: <next hypothesis>
- If Partial success: <which tradeoff to address, or how to consolidate gain>
- If Failure: <what to re-engineer, or whether to abandon the direction>
- If Noise: <more seeds, or a stronger version of the change>
- If Bug: <which bug-checklist items to verify>

## 8. Appendix

- Full per-class accuracy tables
- All visualization figures, not just the representative ones
- Config diff against baseline (git diff output)
- Training curves if relevant
```

---

## Filled example — what "good" looks like

```markdown
# Experiment report — Local attention module on CUB-GZSL

**Date:** 2026-04-18
**Experiment dir:** experiments/2026-04-18-local-attn-cub/
**Baseline:** f-CLSWGAN (Xian 2018), re-run at commit a3f2c91
**Seeds:** [0] — seeds 1, 2 running
**Verdict label:** Partial success
**Confidence:** Low

## 1. Hypothesis

**Issue targeted:**
Earlier analysis (experiments/2026-04-10-diagnostic-gradcam/) showed that on 18 off-center-object classes in CUB, Grad-CAM consistently pointed to the image center even when the bird was in a corner. Per-class U on those 18 classes was 6.2% vs 24.1% on the other unseen classes — strong evidence of center-prior shortcut.

**Code change:**
Added a local attention module between ResNet-101 layer-4 and the semantic projection head, supervised softly by bbox priors (commit 8d4e102). The module learns a spatial weighting conditioned on the semantic query.

**Predicted effect:**
- Quantitative: U on the 18 off-center classes rises >10 points; U on centered-object classes flat or drops ≤2 points (tradeoff). Overall H rises 1–3 points.
- Qualitative: Grad-CAM center shifts toward ground-truth bbox on the 18 classes.
- S may drop 1–2 points (less center-shortcut benefit for seen training).

**What would falsify:**
If Grad-CAM center doesn't shift on the 18 target classes, or U on those classes doesn't improve, the attention module isn't doing its intended job.

## 2. Headline

**One-line:** Partial success at Low confidence — mechanism confirmed on target classes (U on 18 off-center classes rose 6.2 → 18.7), overall H rose 1.0 but within single-seed noise band.

| Metric | Baseline | New | Δ |
|---|---|---|---|
| S | 54.8 | 52.1 | −2.7 |
| U | 43.2 | 46.9 | +3.7 |
| H | 48.3 | 49.3 | +1.0 |

**Caveats:**
- Single seed. Seeds 1 and 2 running.
- Off-center class list derived from the same diagnostic run that motivated the change — circular. Held-out validation pending.

## 3. Accuracy analysis

### 3.1 Full table

| | Baseline | New | Δ |
|---|---|---|---|
| S | 54.8 | 52.1 | −2.7 |
| U | 43.2 | 46.9 | +3.7 |
| H (γ=0) | 48.3 | 49.3 | +1.0 |
| H (best γ) | 51.7 | 53.4 | +1.7 |

Best γ moved from 0.4 (baseline) to 0.25 (new) — consistent with less seen-bias.

### 3.2 Per-class patterns

- 18 off-center target classes: U 6.2 → 18.7 (+12.5 avg; +22.1 on top-5 most off-center).
- Remaining 32 unseen: 44.6 → 44.3 (−0.3, within noise).
- Seen: drop concentrated on 8 pathologically centered classes — baseline was free-riding on center prior.

Matches the prediction closely.

### 3.3 Variance & sanity

- Single seed. Historical variance on this architecture on CUB H is ±0.5, so current Δ of +1.0 is marginal pending seed replication.
- CLIP zero-shot unchanged (37.2 → 37.3, decimal noise).
- γ sweep curve flattened, as predicted.

## 4. Visualization analysis

### 4.1 Grad-CAM

Figure 1: six off-center target-class test images, baseline vs new side-by-side.

On 6/6 shown cases (and 14/18 on the broader check), new attention's top-10% pixels overlap with ground-truth bbox (IoU > 0.3). Baseline was center-biased in all 6. **Qualitatively supports the mechanism hypothesis.**

On 2 of the 18 classes, attention overshoots into adjacent tree branches. Minor edge case; noted for next iteration. Visual evidence combined with case analysis in §5 — not treating maps as proof on their own.

### 4.2 t-SNE of unseen features

Figure 2: t-SNE of unseen visual features, colored by class.

Clusters in similar positions overall; 18 off-center classes' clusters appear slightly tighter by visual inspection. No compactness metric computed; adding to next iteration.

## 5. Case analysis

### 5.1 Quadrant counts (unseen test set)

| | baseline correct | baseline wrong |
|---|---|---|
| **new correct** | 412 | 167 |
| **new wrong** | 93 | 528 |

Net: +74 samples on unseen.

### 5.2 Improvement cases

Sampled 15 from ② (stratified by class). 12/15 are off-center target classes; for 10/12 the new attention overlaps the bird and baseline attention was centered. Mechanism explanation for improvements: consistent.

### 5.3 Regression cases

Sampled 20 from ③.

| Cause | Count | Note |
|---|---|---|
| (a) intended tradeoff — new attention locks on a nearby texture, bird features not yet matured | 14 | concentrated on tiny-bird cases |
| (a) intended tradeoff — partial occlusion; new loses global context | 4 | scene cue shortcut broken |
| (c) noise | 2 | no clear pattern |

18/20 regressions are (a) intended tradeoff. Change works as designed.

## 6. Verdict

- **Issue resolved?** Yes on the 18 targeted classes (§3.2, §4.1). Secondary overshoot on 2 classes (§4.1).
- **Accuracy direction:** up (+1.0 on H, +1.7 with calibration).
- **Expected mechanism observed?** Yes — attention shift in §4.1, regression causes in §5.3 consistent with the intended tradeoff.

**Label:** Partial success.
**Confidence:** Low.

**Reasoning:** Issue is resolved on the targeted classes. Mechanism fires as designed (visualizations + regression cause pattern). Accuracy is net positive but within single-seed noise — hence Partial, and confidence is Low pending replication.

**What would upgrade or downgrade:**
- Upgrade to Success (and Medium confidence) if seeds 1 and 2 hold the +1.0 H delta.
- Upgrade to High confidence if additionally the off-center-class U improvement replicates across seeds with small std.
- Downgrade to Noise if seed replication shows H Δ within ±0.5 and no consistent off-center-class improvement.

## 7. Next steps

1. Finish seeds 1 and 2 to confirm Δ isn't noise (addresses the Low confidence).
2. Tiny-bird cases regressed — add scale-aware branch in next iteration.
3. Test on AwA2 to check mechanism transfer.
4. Consider two-stream (local + global) to recover occlusion performance without losing attention shift.

## 8. Appendix

<per-class full table, config diff, full figure set>
```

## Why this example is good

- **Predictions written before results** (§1); §3.2 and §4.1 explicitly compare against them.
- **Headline not oversold:** +1.0 within noise, explicitly flagged. Confidence tagged as Low matches this reality.
- **Visualization analysis ties back to hypothesis** — not "attention looks nice", but "14/18 target classes show bbox overlap >0.3".
- **Regression analysis done even though net accuracy rose.** Author identifies the tiny-bird tradeoff as the next-iteration target.
- **Verdict is three-part + label + confidence**, not a single word.
- **Explicit upgrade/downgrade paths** make the next run's decision criterion clear in advance — prevents post-hoc reinterpretation.
- **Next steps are specific and hypothesis-driven**, not "try more things."

## Anti-example — don't do this

```markdown
# Experiment — added local attention

H went from 48.3 to 49.3. Nice. Let's ship it.
Attention maps look better (see fig).
TODO: more seeds.
```

Problems: no hypothesis, no prediction, no falsification, "looks better" without criteria, single-seed treated as conclusive, no regression analysis, no verdict structure, no label, no confidence, vague next step.
