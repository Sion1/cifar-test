# AutoResearch Design Flowcharts (drawn by Claude)

Generated 2026-04-26 after a 4-round Claude ↔ Codex review chain that converged
on the design's actual code-level behavior (including known design gaps).

## Files

| File | Description |
|---|---|
| `01_main_loop.png` | Figure 1 — `loop.sh` main scheduler state machine (single tick) |
| `02_analyze_subflow.png` | Figure 2 — Step 2 analyze + async consensus dispatch |
| `03_consensus_chain.png` | Figure 3 — 5-cycle consensus chain (background ~13min) |
| `04_training_modules.png` | Figure 4 — Training data / module dependency flow (M1/M2/M3/M4) |
| `*.mmd` | Mermaid source for each figure (high-fidelity, re-renderable) |
| `render_pil.py` | Local PIL renderer used to produce the PNGs |

## How to re-render

PNGs were rendered locally with PIL (no external services, no system packages
required). To regenerate after editing:

```bash
python3 figs/autoresearch_design_by_claude/render_pil.py
```

For higher-quality output via Mermaid (requires browser deps installed):

```bash
mmdc -i 01_main_loop.mmd -o 01_main_loop.png -w 2400 -H 1800 --backgroundColor white
```

## Highlights worth noting

These four figures together form the *maintenance-grade* view of the system —
i.e., the level of detail required to safely modify `loop.sh`, the consensus
workflow, or `git_iter_commit.sh`.

Critical design gaps and safety valves marked in **WARN** boxes:

1. **`status=failed` is a dead-end state** — not auto-analyzed, not counted in
   STOP rule. Distinct from `verdict=Failure`.
2. **Consensus gate only checks file existence**, not `STATUS=...` content.
   `PARSE_FAIL` final.md will still release the gate.
3. **`FREE_GPU` is propose pre-check only** — not passed to `run_experiment.sh`,
   which selects GPU independently.
4. **`AUTORES_GIT_AUTOPUSH` defaults to 1** (push enabled), despite top-of-file
   comment saying "default no". Code/comment mismatch.
5. **Defensive Bug-mark guard** fires for both rc=0 (silent) and rc=124
   (timeout) failures of `claude -p`, preventing 80K-token-per-tick burn.
6. **`9>&-` in setsid spawn** is critical — closes inherited flock fd, else
   consensus blocks all subsequent ticks for ~13 min.
7. **git_iter_commit.sh restore step** — after `git checkout PREV_BRANCH`,
   files are restored to working tree from the iter branch, else async
   consensus cannot find primary md → infinite analyze loop.
8. **Module dependencies**: M2 and M4 require M1 ON; M3 is independent.

## Provenance

This is the result of:
- v1: Claude drafts initial diagrams from project source (no logs/results read)
- v2: Codex review identifies 12 issues including 7 safety-valve omissions
- v3: Claude verifies each issue against code with `grep`/`sed`, produces
  corrected version
- v4: Codex tightens 2 over-stated claims (file-existence vs STATUS check;
  `LATEST_ANALYZED` Bug-skip second-order bug)
- Final (these PNGs): all 4 rounds of corrections incorporated.

The full review chain is preserved in the conversation transcript. These
diagrams reflect Claude's understanding *after* incorporating all of Codex's
corrections.
