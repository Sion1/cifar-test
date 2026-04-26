---
name: experiment-analysis
description: Structured, hypothesis-driven analysis of experiment results for ZSL/GZSL research. Use this skill whenever the user finishes a training run and asks to analyze, interpret, compare, or write up results — including when they mention accuracy changes, ablation results, Grad-CAM / attention maps, confusion matrices, regression cases, comparing to baseline, or deciding whether a code change "worked". Also trigger on requests to write an experiment report, debug why a change made things worse, or decide next steps after seeing numbers. Trigger proactively even when the user just says "results are in, what do you think?" — don't just dump metrics back, apply this protocol.
---

# Experiment analysis — for ZSL/GZSL research

This skill governs how to analyze experiment results after a code change. The goal is not to decide "did the number go up?" but **"did we fix the issue we were trying to fix, and what do the results tell us about next steps?"**

## Core principle — the non-negotiable one

**A drop in accuracy is not automatically a failure.** If the change was designed to fix a specific issue (e.g., "model ignores the target region", "hubness inflates top-5 predictions"), and the qualitative evidence shows the issue is resolved, the run is a partial or full success even with lower raw accuracy.

Why: accuracy is a compound metric entangling many things. A change that reallocates attention correctly but hasn't yet learned the new region well will look bad on accuracy but good on attention maps. Dropping that change because of accuracy alone throws away a partially-working correct direction.

**Conversely, accuracy going up is not automatically a success.** If the issue isn't resolved and accuracy went up for unrelated reasons (shortcut exploitation, overfitting to the seen set, benign seed variance), the run doesn't validate the hypothesis.

Judge by three axes, all together: **(issue resolved?) × (accuracy direction) × (expected mechanism observed?)** — not one alone. The `## Verdict labels` section below defines the combinations.

## Minimum inputs

For a strong verdict, gather as many of the following as possible:

- Dataset and seen/unseen split (e.g., "CUB, Xian 2017 proposed split")
- Baseline method and its version/commit if available; new method and its version/commit if available
- S / U / H (or per-class CZSL accuracy if conventional ZSL — see note below)
- Whether results are single-seed or multi-seed; if multi, the std
- Per-class accuracy breakdown if available
- Calibration (γ) sweep data if available
- Visualization evidence if the hypothesis is mechanistic (attention maps, t-SNE, confusion matrix)
- Per-sample predictions from both runs if quadrant regression analysis is needed

**ZSL vs GZSL note:** the protocol below assumes GZSL (the realistic setting with both seen and unseen at test time). For conventional ZSL, replace the S/U/H block with the metric appropriate to that setting (typically per-class top-1 on unseen only), but keep the same `hypothesis → evidence → verdict` logic everywhere else.

**Missing-data principle** (applies to every step): if required inputs are absent, do not stop by default. Give a provisional analysis, explicitly state which evidence is missing, and lower the confidence level of the verdict accordingly. Ask a follow-up question only when the missing information would materially change the verdict (e.g., you genuinely cannot tell success from noise without a second seed or per-class breakdown).

When a step requires data the user doesn't have, **give them the command to produce it** rather than approximating:

> I can give a provisional verdict from the aggregate metrics, but the regression case analysis needs sample-level predictions from both runs. Dump them with: `torch.save({'y_true': ..., 'y_base': ..., 'y_new': ..., 'paths': ...}, 'preds.pt')`. Once that's available I can run the full quadrant analysis.

## Analysis depth

Pick the depth based on the user's request and available evidence:

- **Quick review** — user asks "how does this look?" with just headline numbers. Minimum output (see `## Output structure` below).
- **Standard analysis** — user wants to understand the result properly. Output: hypothesis reconstruction, S/U/H table with per-class patterns, 2–3 key visualizations tied back to hypothesis, caveats, three-part verdict with confidence.
- **Deep dive** — user is writing up the experiment or making a decision that matters. Output: full protocol (Steps 0–4 below) + formal report from the template.

Default to **Standard** unless the user signals otherwise ("just a quick check" → Quick; "write up the report" → Deep dive). Follow the protocol in order; deviate only when explicitly asked for a shorter or narrower analysis.

## Output structure

Every analysis, regardless of depth, must at minimum contain these five blocks in this order:

1. **Reconstructed hypothesis** — what the change was supposed to fix, one or two sentences
2. **Headline verdict** — one sentence stating the label (from `## Verdict labels`) and direction
3. **Evidence for the verdict** — keyed to the three axes (issue resolved / accuracy / mechanism); length scales with analysis depth
4. **Missing evidence and confidence** — what's not available, resulting confidence level (from `## Confidence`)
5. **Next step** — one concrete action

Quick review can do this in ~150 words. Standard expands block 3 with tables and per-class patterns. Deep dive expands all blocks per the `report-template.md` structure.

For Quick and Standard analyses, follow `## Output structure` directly. For Deep dive write-ups, expand the same structure into the full report format defined in Step 4.

## The analysis protocol

### Step 0 — reconstruct the hypothesis before interpreting results

Before analyzing the numbers, pin down:

1. What issue the change was supposed to fix.
2. What evidence should appear if it works (quantitative + qualitative, with specific subsets or classes if possible).
3. What result would falsify the hypothesis.

If the user did not state these clearly, reconstruct them from motivation + code change, state your assumptions explicitly, and proceed with a provisional analysis (per the Missing-data principle above).

Examples of good predictions:

- "Added attention module targeting object region → Grad-CAM heatmap center should shift toward ground-truth bounding box; unseen per-class accuracy on off-center-object classes should improve more than on centered classes."
- "Added contrastive loss to reduce seen-bias → S should drop 1–3%, U should rise 3–6%, H should rise, γ sweep curve should flatten."
- "Replaced semantic embedding with latent embedding to kill hubness → k-occurrence distribution skewness should drop; per-class U variance should shrink."

Reject vague predictions and sharpen them:

- "Accuracy should go up." → ask: on which subset, driven by S or U, by how much
- "The model should be better." → unfalsifiable, reject
- "Unseen recognition should improve." → by how much, on which classes, measured how

### Step 1 — accuracy-level analysis

Don't stop at "H went from 32.1 to 34.5". Do all of these when data allows:

**1a. The three GZSL numbers.** Always report S, U, H together. For metrics conventions (per-class not per-sample, harmonic mean formula, calibrated stacking, AUSUC) follow the `gzsl` skill.

```
           Baseline       New         Δ
S          __.__          __.__       +/- __
U          __.__          __.__       +/- __
H          __.__          __.__       +/- __
```

**1b. Direction check.** Is the Δ consistent with the prediction from Step 0? "Accuracy went up" doesn't count as consistent if you predicted "S down, U up, H up" and saw "S up, U flat, H slightly up" — that's a different mechanism firing.

**1c. Calibration curve.** If γ sweep data exists: is H-vs-γ shifted, flattened, or reshaped? A flatter or more robust H-vs-γ curve can indicate reduced sensitivity to calibration, which is often consistent with lower seen-bias — but check that the curve flattened at a higher H level, not a lower one (flat-but-worse is not progress).

**1d. Per-class breakdown.** Average accuracy hides the signal in GZSL. Ask:
- Which classes improved? Which regressed?
- Is the set of improved classes the one the hypothesis predicted (e.g., classes where the issue was strongest)?
- Did per-class variance narrow or widen?

**1e. Seed variance.** Treat single-seed gains as suggestive rather than confirmatory unless the gain is clearly larger than the run-to-run variance historically observed for that setup. As a rough heuristic, gains around 1–2 points on H may still fall within noise for fine-grained benchmarks; request ≥3 seeds when the decision matters.

**1f. Sanity baselines.** Did the change break any baseline it shouldn't have? E.g., CLIP zero-shot floor should be unchanged if the CLIP encoder wasn't touched. Unexpected movement suggests an implementation bug — jump to the bug checklist.

### Step 2 — visualization-level analysis

**Visualizations are supporting evidence, not proof of causal mechanism.** Attention maps can "look right" without being the actual driver of the prediction. Treat them as necessary-but-not-sufficient evidence: if they contradict the hypothesis, the hypothesis is weakened; if they support it, combine with quantitative evidence before declaring mechanism confirmed.

For each visualization type available:

**2a. Attention / Grad-CAM / saliency.**
- Does the attention now land where Step 0 predicted?
- Compare the SAME images side-by-side (baseline vs new). When enough examples are available, try to inspect representative cases from each quadrant; prioritize ② improvements and ③ regressions. A balanced set looks like:
  - 3 cases from ② (baseline wrong, new right — improvement)
  - 3 cases from ③ (baseline right, new wrong — regression)
  - 2–3 cases from ① (both right — is the rightness qualitatively different?)
  - 2–3 cases from ④ (both wrong — failing the same way, or differently?)
- Under **Standard analysis**, inspecting ② and ③ alone is sufficient. The ①/④ columns are a Deep-dive extension.
- Quantify when possible: IoU between top-X% attention pixels and ground-truth bbox, averaged over a class.

**2b. Feature embedding plots (t-SNE / UMAP).**
- Are unseen class clusters more separated or more collapsed?
- Are unseen samples closer to their true prototypes or drifting?
- Compare distribution shape, not just centroids.

**2c. Confusion matrix.**
- For seen-bias specifically: did the unseen-block diagonal strengthen?
- Are there specific seen classes that unseen samples were being dumped into? Did that pattern change?

**2d. Prototype distance histograms.**
- Visual-to-semantic distance for unseen samples: tighten or widen?
- Ratio of "distance to nearest seen prototype" / "distance to true unseen prototype" — did it flip?

### Step 3 — case-level analysis via the quadrant framing

**Precondition:** this step requires per-sample predictions from both runs. If only aggregate metrics are available, skip this step, note it explicitly, and provide the dump command from the Minimum Inputs section.

**3a. Partition test samples into four quadrants:**

|  | baseline correct | baseline wrong |
|---|---|---|
| **new correct** | ① stable correct | ② improvements |
| **new wrong** | ③ regressions | ④ stable wrong |

Count each. Net change is `|②| − |③|`. **Quality** of the change is judged by:
- Do improvements in ② match the hypothesis? (e.g., if we fixed attention for off-center objects, are the improvements concentrated there?)
- Are regressions in ③ concentrated in a specific failure mode?

**3b. For each regression case**, sample 10–20 from ③ (or all of them if the pool is smaller):
- Pull up image + baseline attention + new attention + baseline prediction + new prediction
- Write one sentence per case: **why does the new model fail here, given that baseline got it right?**
- Group reasons. If 15/20 regressions share a cause, you have a clear side-effect to name.

**3c. The crucial question:**
- Are regressions **consistent with the intended change**? (e.g., "we made the model focus on local region, it now misses cases needing global context" — expected tradeoff, change works as designed)
- Or are regressions **unrelated**? (random-looking pattern, different failure modes — likely noise or bug)

See `references/regression-analysis.md` for the full four-cause taxonomy and code templates.

### Step 4 — write up the report

Use the structure in `references/report-template.md`. The report is a **story with evidence**:

1. Hypothesis (from Step 0, written before numbers)
2. Headline result (one-sentence verdict + critical numbers + caveats)
3. Accuracy analysis (Step 1)
4. Visualization analysis (Step 2)
5. Case analysis (Step 3, if per-sample predictions available)
6. Verdict (three-part judgment + verdict label + confidence)
7. Next steps
8. Appendix

## Verdict labels

After the three-part judgment, assign exactly one:

- **Success** — intended issue is resolved, expected mechanism is observed, and accuracy is neutral or favorable. No significant new failure modes.
- **Partial success** — intended issue is resolved or partially resolved; accuracy regresses due to a clear, explainable tradeoff (not random). Mechanism fires as designed. Next iteration should address the tradeoff.
- **Failure** — intended issue is not resolved AND expected mechanism is not observed in visualizations/case analysis. Accuracy direction is irrelevant to this verdict.
- **Noise** — changes in metrics and visualizations are too small, inconsistent, or seed-dependent to distinguish from variance. Need more seeds or larger intervention.
- **Bug** — results violate sanity expectations (e.g., CLIP floor moved, reproducibility broken, train/test leakage suspected). Go to the bug checklist before re-analyzing. Note: a Bug verdict is not a scientific conclusion about the hypothesis — it suspends interpretation until the implementation or evaluation issue is identified and fixed. Don't report "the method failed" based on results that came from a buggy run.

Use these labels consistently — they shape the next-step recommendation. Partial success is the most common under the Core Principle; don't downgrade it to Failure just because H dropped.

**When evidence is incomplete, prefer the most conservative label the data supports, and state explicitly what would upgrade or downgrade the verdict.** For example: "Tentatively Partial success at Low confidence; would upgrade to Success if seed replication holds the +1.0 H delta; would downgrade to Noise if a second seed shows no movement."

## Confidence

Every verdict must be tagged with confidence:

- **High** — multi-seed results, per-class breakdown present, and the mechanism evidence (visualization + case analysis) is consistent with the hypothesis.
- **Medium** — aggregate metrics are available, and at least one additional evidence source (per-class breakdown, visualization, or case analysis) supports the mechanism **without strong contradictory evidence elsewhere**. If one evidence source supports and another clearly contradicts, downgrade to Low until the conflict is resolved.
- **Low** — single-seed and/or aggregate-only; no mechanism evidence available to distinguish real effect from noise.

Confidence is independent from the verdict label. You can have a Low-confidence Success (promising but needs replication) or a High-confidence Failure (multiple seeds, clear absence of mechanism). Low-confidence verdicts should be paired with the specific evidence that would raise confidence.

## Bug suspicion checklist

When a result looks wrong in a way that doesn't match any coherent hypothesis, check these before concluding "the idea doesn't work":

1. **Sanity baseline moved unexpectedly** — e.g., CLIP zero-shot changed by more than decimal noise, or a re-run of the baseline itself gives different numbers from the prior report.
2. **Seen/unseen split mismatch** — check the class index lists match between runs; verify not using the old Lampert split by accident (see `gzsl` skill on splits).
3. **Calibration γ silently changed** — if the baseline and new runs use different γ defaults, H comparison is invalid.
4. **Eval preprocessing differs from training** — image resize, normalization stats, crop strategy. Especially dangerous when switching between frameworks.
5. **Prototype normalization or temperature changed unintentionally** — L2 norm on prototypes, softmax temperature. These quietly rescale similarities.
6. **Class index mapping mismatch** — if class names → indices mapping changed between runs, accuracy numbers are comparing apples to oranges.
7. **Train/eval mode issue** — dropout, BatchNorm eval mode, teacher-forcing left on. `model.eval()` missed somewhere.
8. **Seed or data sampling inconsistency** — different random seed than claimed, different validation split, shuffle state not reset.
9. **Logging / checkpoint mismatch** — logged metrics are from a different epoch than the saved checkpoint you evaluated.
10. **Numerical precision / fp16 issue** — model trained in fp16 but evaluated in fp32 (or vice versa) can shift softmax outputs non-trivially.

Before filing a "bug" verdict, run **at least one** of these checks explicitly. Most "unexplained regressions" in ZSL/GZSL turn out to be items 2, 4, or 7.

## Anti-patterns to flag

Push back when you catch the user (or yourself) doing these:

1. **"Accuracy went up, ship it."** — Check the mechanism is what you thought. Otherwise you're optimizing H by shortcut.
2. **"Accuracy went down, scrap it."** — Check visualizations and regression pattern first. You may be throwing away a correct direction.
3. **Looking at aggregate numbers only.** — Per-class, per-failure-mode, per-subset. Average is a liar in GZSL.
4. **Single-seed strong claims.** — One seed is suggestive, not confirmatory.
5. **Weak-baseline comparison.** — Tuned method vs default-hyperparameter baseline isn't honest. Tune both or tune neither.
6. **Cherry-picked visualizations.** — If you pick the 3 nicest attention maps for the report, it's marketing, not analysis. Include regression cases too.
7. **HARKing** (Hypothesizing After Results Known) — Predicted effect didn't show up, but you reinterpret to fit a new "actual" hypothesis. Write the new hypothesis cleanly and test it on a new run.
8. **Confusing visual plausibility with causal evidence** — attention map "looks right" is a hint, not proof. Combine with case-level and quantitative evidence.

## When to go deeper

- GZSL domain knowledge (failure modes, metrics, calibrated stacking, AUSUC) → `gzsl` skill. Cross-check analysis against that skill's conventions.
- Regression case deep-dive template with code → `references/regression-analysis.md`
- Report template with a complete worked example → `references/report-template.md`
- Failure mode ↔ visualization cheat sheet → `references/failure-diagnostics.md`

## Interaction style

- Rushing to verdict is worse than asking for more data. Saying "I need per-class breakdown and 3 regression images before verdict" is better than a confident guess on thin evidence. But don't stall: always give what you can from what you have, mark the confidence, and name the missing pieces.
- Use the quadrant framing (①②③④) when discussing case counts.
- Prefer specific numbers with sources over vague claims. "H rose 2.3 points on CUB, single seed" beats "H improved meaningfully."
- If the user is excited about a positive result, stay level-headed — check the mechanism.
- If the user is discouraged by a negative result, apply the Core Principle — point out what they might be missing.
- When uncertain between two verdicts (e.g., Partial success vs Noise), name both candidates, say what evidence would resolve it, and ask for that evidence.
- Match output weight to question weight. If the user asks something narrow ("summarize this ablation in two sentences", "is the delta meaningful?"), don't force the full protocol — preserve the five-block `Output structure` but compress each block to one line. Full protocol is for full questions.
