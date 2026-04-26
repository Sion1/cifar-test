#!/usr/bin/env python3
"""parse_consensus.py — aggregate the 5-cycle consensus reports for one iter.

Reads all logs/iteration_NNN.consensus.*.md files and writes the canonical
logs/iteration_NNN.consensus.final.md that the propose phase reads.

The propose phase MUST honor the STATUS field:
- STATUS=CONSENSUS         → safe to propose using NEXT_STEP
- STATUS=OVERRIDE_BY_MAIN  → propose using NEXT_STEP but flag dissent in PR
- STATUS=PARSE_FAIL        → propose phase BLOCKS (loop tick exits without
                             launching new iter; user intervention required)

Usage: python3 scripts/parse_consensus.py <ITER>
"""
from __future__ import annotations
import sys, re, pathlib, datetime, glob

if len(sys.argv) != 2:
    print("usage: parse_consensus.py <ITER>", file=sys.stderr)
    sys.exit(1)

iter_num = int(sys.argv[1])
pad = f"{iter_num:03d}"
final_out = pathlib.Path(f"logs/iteration_{pad}.consensus.final.md")
primary = pathlib.Path(f"logs/iteration_{pad}.md")
r5_main = pathlib.Path(f"logs/iteration_{pad}.consensus.main.r5.md")
r3_main = pathlib.Path(f"logs/iteration_{pad}.consensus.main.r3.md")

def safe_read(p: pathlib.Path) -> str:
    try:
        return p.read_text(errors="replace") if p.exists() else ""
    except OSError:
        return ""

# --- Source of truth: cycle 5 main agent's final.md if present -----------
r5_text = safe_read(r5_main)

def extract(text: str, pat: str, default=None):
    m = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else default

status = None
if r5_text:
    status = extract(r5_text, r"##\s*Status\s*\n+\**\s*\**(\w+)")
    if status:
        status = status.upper()

# --- Sanity-recompute from R2 evals (in case main's status is malformed) -
r2_files = sorted(glob.glob(f"logs/iteration_{pad}.consensus.*.r2.md"))
r2_verdicts = {}
for f in r2_files:
    p = pathlib.Path(f)
    m = re.match(rf"iteration_{pad}\.consensus\.([^.]+)\.r2\.md", p.name)
    if not m:
        continue
    agent = m.group(1)
    text = safe_read(p)
    v = extract(text, r"##\s*R2 verdict\s*\n+\**\s*\**(\w+)")
    if not v:
        # Fallback: look for bolded keyword anywhere
        m2 = re.search(r"\*\*(AGREE|DISAGREE)\*\*", text)
        v = m2.group(1).upper() if m2 else "MISSING"
    r2_verdicts[agent] = v.upper() if v else "MISSING"

# Recompute consensus from R2
all_agree = bool(r2_verdicts) and all(v == "AGREE" for v in r2_verdicts.values())
recomputed = "CONSENSUS" if all_agree else "OVERRIDE_BY_MAIN"

# If main's R5 is missing OR mismatches recomputed, prefer recomputed
# (main agent's status may be stale; tally is authoritative)
if not status or status not in ("CONSENSUS", "OVERRIDE_BY_MAIN"):
    if r2_verdicts:
        final_status = recomputed
        status_source = "recomputed_from_R2 (main R5 missing/malformed)"
    else:
        final_status = "PARSE_FAIL"
        status_source = "no R2 evals found"
elif status != recomputed:
    # Mismatch: trust recomputed (R2 tally is the source of truth per project policy)
    final_status = recomputed
    status_source = f"recomputed_from_R2 (main said {status})"
else:
    final_status = status
    status_source = "main R5 confirmed"

# --- Extract NEXT_STEP from main's R5 or primary §8 ----------------------
next_step = ""
if r5_text:
    # Look for section ## Final next-step
    m = re.search(r"##\s*Final next-step.*?\n+(.*?)(?=\n##|\Z)", r5_text, re.DOTALL | re.IGNORECASE)
    if m:
        next_step = m.group(1).strip()
if not next_step:
    # Fallback: primary §8
    p_text = safe_read(primary)
    m = re.search(r"##\s*8\.\s*Next hypothesis\s*\n+(.*?)(?=\n##|\Z)", p_text, re.DOTALL | re.IGNORECASE)
    if m:
        next_step = m.group(1).strip()
if not next_step:
    next_step = "(NEXT_STEP NOT EXTRACTED — propose phase should fall back to standard frontier-pick prompt)"

# --- Eval R1 critiques summary -------------------------------------------
r1_files = sorted(glob.glob(f"logs/iteration_{pad}.consensus.*.r1.md"))
r1_summaries = []
for f in r1_files:
    p = pathlib.Path(f)
    m = re.match(rf"iteration_{pad}\.consensus\.([^.]+)\.r1\.md", p.name)
    if not m:
        continue
    agent = m.group(1)
    text = safe_read(p)
    verdict_review = extract(text, r"##\s*Verdict review.*?\n+\**\s*\**(\w+)") or "?"
    next_review = extract(text, r"##\s*Next-step review.*?\n+\**\s*\**([\w-]+)") or "?"
    r1_summaries.append((agent, verdict_review.upper(), next_review.upper()))

# --- Build consensus.final.md --------------------------------------------
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"# Iteration {pad} — consensus.final",
    f"Date: {ts}  |  Source: {status_source}",
    "",
    f"STATUS={final_status}",
    "",
    "## R2 tally (drives STATUS)",
    "",
]
for agent, verdict in r2_verdicts.items():
    lines.append(f"- {agent}: **{verdict}**")
if not r2_verdicts:
    lines.append("- (no R2 reports found)")

lines += [
    "",
    "## R1 critique summary (for context)",
    "",
    "| Agent | Verdict review | Next-step review |",
    "|---|---|---|",
]
for agent, vr, nr in r1_summaries:
    lines.append(f"| {agent} | {vr} | {nr} |")
if not r1_summaries:
    lines.append("| (no R1 reports) | — | — |")

lines += [
    "",
    "## Final next-step (propose phase MUST use this)",
    "",
    next_step,
    "",
    "## Source files",
    f"- Primary analysis: [iteration_{pad}.md](iteration_{pad}.md)",
]
for r1 in r1_files:
    lines.append(f"- R1: [{pathlib.Path(r1).name}]({pathlib.Path(r1).name})")
if r3_main.exists():
    lines.append(f"- R3 main revise: [{r3_main.name}]({r3_main.name})")
for r2 in r2_files:
    lines.append(f"- R2: [{pathlib.Path(r2).name}]({pathlib.Path(r2).name})")
if r5_main.exists():
    lines.append(f"- R5 main final: [{r5_main.name}]({r5_main.name})")
lines.append("")

final_out.write_text("\n".join(lines) + "\n")

# Stdout summary (tee'd to driver.log)
print(f"CONSENSUS_PARSE iter={pad} status={final_status} "
      f"r2_tally={r2_verdicts} source={status_source}")
