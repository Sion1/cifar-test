#!/usr/bin/env bash
# git_iter_commit.sh — Auto-commit one iteration's analysis to a per-iter branch.
#
# Called by loop.sh's analyze step after claude -p completes its iteration_NNN.md.
# Workflow:
#   1. Switch to (or create) branch "autoresearch/iter-NNN"
#   2. Stage all changes (config + report + viz + CLAUDE.md edits + state.tsv row)
#   3. Build a structured commit message using metrics from final.pth + iteration_NNN.md
#   4. Commit (no merge; AUTO-PUSH by default since AUTORES_GIT_AUTOPUSH defaults to 1 — set to 0 to keep branch local)
#   5. Switch back to main without merging
#
# Usage:
#   bash scripts/git_iter_commit.sh <ITER_NUM>
#
# Env:
#   AUTORES_GIT_AUTOPUSH=1 — also push the branch to origin (default: 1 / push enabled)
#   AUTORES_GIT_REMOTE=origin — remote name to push to (default: origin)
#
# Exit codes:
#   0 = commit OK (or nothing to commit, which is also OK)
#   1 = bad usage / git error / iteration metadata missing
set -u
cd "$(dirname "$0")/.."

ITER="${1:-}"
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "[git_iter_commit] usage: $0 <ITER_NUM>" >&2
    exit 1
fi
ITER_PAD=$(printf '%03d' "$ITER")
BRANCH="autoresearch/iter-${ITER_PAD}"
LOG=logs/driver.log
log() { printf '[%s] [git-bot] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# Sanity: this must be a git repo
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    log "ERROR: not a git repository; skipping commit"
    exit 1
fi

# Sanity: iter must be analyzed (have a row + iteration_NNN.md)
REPORT="logs/iteration_${ITER_PAD}.md"
if [ ! -f "$REPORT" ]; then
    log "ERROR: $REPORT not found; iteration $ITER_PAD not yet analyzed"
    exit 1
fi

# Read state.tsv row to extract verdict + best_h
ROW=$(awk -F'\t' -v i="$ITER" '$1 == i {print; exit}' state/iterations.tsv)
if [ -z "$ROW" ]; then
    log "ERROR: no row for iter $ITER in state/iterations.tsv"
    exit 1
fi
EXP_NAME=$(echo "$ROW" | awk -F'\t' '{print $3}')
BEST_H=$(echo "$ROW" | awk -F'\t' '{print $9}')
VERDICT=$(echo "$ROW" | awk -F'\t' '{print $10}')

# Extract one-line hypothesis summary from §1 of the report
HYPOTHESIS_LINE=$(awk '/^## 1\. Hypothesis/{flag=1; next} /^## /{flag=0} flag && NF{print; exit}' "$REPORT" \
    | head -c 250 | tr '\n' ' ')
# Extract H/U/S from §4 results table (look for the row starting with "| H ")
H_ROW=$(grep -E '^\| H *\|' "$REPORT" | head -1)
U_ROW=$(grep -E '^\| U *\|' "$REPORT" | head -1)
S_ROW=$(grep -E '^\| S *\|' "$REPORT" | head -1)
# Extract config diff snippet from §3
CONFIG_DIFF=$(awk '/^## 3\. Changes made/{flag=1; next} /^## 4\./{flag=0} flag' "$REPORT" \
    | head -10)
# Decision line from §7
DECISION=$(awk '/^## 7\. Decision/{flag=1; next} /^## 8\./{flag=0} flag && NF{print; exit}' "$REPORT" \
    | head -c 200)

# PREV_BRANCH is HARDCODED to main, NOT inferred from HEAD.
#
# Why: HEAD drifts. If a prior iter's commit step pushed to autoresearch/iter-N
# and never returned HEAD to main (push fail, kill -9, etc.), the next iter's
# `git symbolic-ref --short HEAD` returns autoresearch/iter-N. Then the new
# branch `autoresearch/iter-(N+1)` is created off iter-N (not main), AND when
# we restore the working tree at the end, we restore to iter-N rather than to
# main, leaving HEAD permanently drifted. Once that happens, every subsequent
# iter compounds the drift, and `git push origin autoresearch/iter-(N+M)`
# fails with "src refspec ... does not match any" because the local ref was
# never created — the commit is orphaned.
#
# Bug history: 2026-04-26 16:40 in Hysyn-ZSL-v3-SUN-autoresearch — iter009
# committed as f61e74a but no autoresearch/iter-009 ref existed; manual
# `git update-ref` recovery + PR #9 needed.
PREV_BRANCH="main"

# Defensive realign: if HEAD drifted from a prior iter that didn't clean up,
# move it back to main BEFORE doing anything else. Working-tree restoration
# below relies on PREV_BRANCH being main.
_CURRENT_HEAD=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
if [ -n "$_CURRENT_HEAD" ] && [ "$_CURRENT_HEAD" != "main" ]; then
    log "WARNING: HEAD drifted to $_CURRENT_HEAD — realigning to main before commit"
    git checkout main 2>&1 | tail -1 | sed "s/^/  /"
fi

# Create or switch to the per-iter branch (off main, not off whatever was checked out)
git fetch origin "$PREV_BRANCH" 2>/dev/null || true
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
    log "switching to existing branch ${BRANCH}"
    git checkout "$BRANCH" 2>&1 | tail -1 | sed "s/^/  /"
else
    log "creating new branch ${BRANCH} off main"
    git checkout -b "$BRANCH" main 2>&1 | tail -1 | sed "s/^/  /"
fi

# Verify the checkout actually succeeded — if HEAD is detached or sitting on
# a different branch, an orphan commit will be made and the push will fail
# with "src refspec ... does not match any". That bug burned iter009 in
# Hysyn-ZSL-v3-SUN-autoresearch on 2026-04-26. Abort loudly instead of
# silently producing an orphan.
_HEAD_AFTER_CHECKOUT=$(git symbolic-ref --short HEAD 2>/dev/null || echo "<detached>")
if [ "$_HEAD_AFTER_CHECKOUT" != "$BRANCH" ]; then
    log "FATAL: checkout did not land on $BRANCH (HEAD=$_HEAD_AFTER_CHECKOUT) — aborting"
    log "       refusing to commit on the wrong ref; investigate before retrying"
    exit 5
fi

# Stage relevant files: config, report, viz, CLAUDE.md, state.tsv, scripts.
# Each pathspec on its own `git add` so a missing file (e.g., figs/iter_NNN/
# never created when smoketest skips viz, or per_class_delta_iterNNN.py never
# generated for a Bug-verdict iter) does NOT abort the whole add and silently
# leave nothing staged. Discovered 2026-04-25 via smoketest iter999/iter998.
for _path in \
    "configs/ablation/iter${ITER_PAD}_"*.yaml \
    "configs/ablation/iter${ITER_PAD}.yaml" \
    "logs/iteration_${ITER_PAD}.md" \
    "figs/iter_${ITER_PAD}/" \
    "scripts/per_class_delta_iter${ITER_PAD}.py" \
    "scripts/plot_gamma_sweep_iter${ITER_PAD}.py" \
    CLAUDE.md \
    state/iterations.tsv; do
    # `[ -e ]` short-circuits on absent paths; the glob can also expand to a
    # literal unmatched pattern (e.g., the *.yaml when no config exists), so
    # `[ -e ]` rejects that case too.
    [ -e "$_path" ] && git add "$_path" 2>/dev/null
done
# Also pick up any other src/ or train.py modifications (rare)
git add -u src/ train.py 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet; then
    log "nothing to commit for iter ${ITER_PAD}"
    git checkout "$PREV_BRANCH" 2>&1 | tail -1 | sed "s/^/  /"
    exit 0
fi

# Build commit message
TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
MSG_FILE=$(mktemp)
cat > "$MSG_FILE" <<EOF
iter${ITER_PAD}: ${VERDICT:-Unknown} — H=${BEST_H:-?}  (${EXP_NAME})

## Hypothesis
${HYPOTHESIS_LINE}

## Config / Code Changes
${CONFIG_DIFF}

## Results
${H_ROW}
${U_ROW}
${S_ROW}

## Verdict
**${VERDICT:-Unknown}** — best_h=${BEST_H:-?}

## Decision
${DECISION}

---
Auto-committed by autoresearch-bot at ${TIMESTAMP}.
Full report: logs/iteration_${ITER_PAD}.md
This branch awaits user review before merging to main.
EOF

git commit -F "$MSG_FILE" 2>&1 | tail -3 | sed "s/^/  /"
COMMIT_RC=$?
COMMIT_SHA=$(git rev-parse --short HEAD)
rm -f "$MSG_FILE"

if [ "$COMMIT_RC" -ne 0 ]; then
    log "ERROR: git commit failed (rc=$COMMIT_RC)"
    git checkout "$PREV_BRANCH" 2>/dev/null
    exit 1
fi

log "committed iter${ITER_PAD} as ${COMMIT_SHA} on branch ${BRANCH}"

# Auto-push + auto-PR creation (default ON — user reviews PRs in GitHub UI).
# To disable, export AUTORES_GIT_AUTOPUSH=0 before launching the loop.
if [ "${AUTORES_GIT_AUTOPUSH:-1}" = "1" ]; then
    REMOTE="${AUTORES_GIT_REMOTE:-origin}"
    if git remote get-url "$REMOTE" >/dev/null 2>&1; then
        log "pushing ${BRANCH} to ${REMOTE}"
        PUSH_OUT=$(git push -u "$REMOTE" "$BRANCH" 2>&1)
        echo "$PUSH_OUT" | tail -3 | sed "s/^/  /"
        if echo "$PUSH_OUT" | grep -qE "new branch|Everything up-to-date|->.*${BRANCH##*/}"; then
            log "push OK"
            # Auto-create PR if not already open. Use gh api directly (gh pr create
            # is unreliable in headless mode — see feedback_subagent_cli_quirks gh #1).
            GH_BIN=$(command -v gh || echo /root/.local/bin/gh)
            REPO_PATH=$(git remote get-url "$REMOTE" | sed -E 's#.*github.com[:/]([^/]+/[^/.]+)(\.git)?#\1#')
            EXISTING_PR=$($GH_BIN pr list --head "$BRANCH" --state all --json number --jq '.[0].number' 2>/dev/null)
            if [ -n "$EXISTING_PR" ]; then
                log "PR #${EXISTING_PR} already exists for ${BRANCH}; skipping create"
            else
                PR_TITLE="iter${ITER_PAD}: ${VERDICT:-Unknown} — H=${BEST_H:-?}"
                PR_URL=$($GH_BIN api -X POST "repos/${REPO_PATH}/pulls" \
                    -f title="$PR_TITLE" \
                    -f head="$BRANCH" \
                    -f base="main" \
                    -f body="Auto-generated PR by autoresearch-bot at $(date -u '+%Y-%m-%d %H:%M:%S UTC').

## Verdict
**${VERDICT:-Unknown}** — best_h=${BEST_H:-?}

## Hypothesis
${HYPOTHESIS_LINE}

## Decision (§7)
${DECISION}

Full report: \`logs/iteration_${ITER_PAD}.md\`
Consensus: \`logs/iteration_${ITER_PAD}.consensus.final.md\` (when ready)
" --jq .html_url 2>&1 | tail -1)
                if echo "$PR_URL" | grep -qE '^https://github.com/'; then
                    log "PR created: $PR_URL"
                else
                    log "PR creation FAILED: $PR_URL"
                fi
            fi
        else
            log "push FAILED — branch is committed locally; user can push manually"
        fi
    else
        log "no remote '$REMOTE' configured; skipping push"
    fi
fi

# Switch back to previous branch
git checkout "$PREV_BRANCH" 2>&1 | tail -1 | sed "s/^/  /"

# Restore iter outputs to PREV_BRANCH's working tree as dirty/untracked, so
# downstream consumers on this branch (consensus_iter.sh reads the .md;
# next loop tick's reaper reads state.tsv; propose reads CLAUDE.md) can find
# them. Without this, the checkout above wipes the working-tree changes that
# the analyze step just produced, and consensus immediately ERRORs with
# "primary report logs/iteration_NNN.md not found", and the next reaper tick
# sees the iter as still 'running' (or 'completed') and re-fires analyze in
# an infinite loop. These files stay dirty in PREV_BRANCH; user merges the
# per-iter branch when they review the PR. (See feedback_autoresearch_operations
# rule #1: branch ops mid-tick wipe state.tsv.)
git checkout "$BRANCH" -- \
    "logs/iteration_${ITER_PAD}.md" \
    state/iterations.tsv \
    CLAUDE.md \
    2>/dev/null || true
# Restore figs/iter_NNN/ separately — git checkout errors hard if the path
# does not exist on the branch (e.g., Bug-verdict iters skip viz). Without
# this restore, viz pngs/csvs vanish from main's working tree after commit
# (they remain on the per-iter branch), and the user sees empty figs/iter_NNN/
# when reviewing locally. Discovered 2026-04-26 via figs/iter_004..007 missing
# despite per-iter branches having attn.png / gamma_sweep.png / tsne.png.
if git ls-tree -r --name-only "$BRANCH" -- "figs/iter_${ITER_PAD}/" 2>/dev/null | grep -q .; then
    git checkout "$BRANCH" -- "figs/iter_${ITER_PAD}/" 2>/dev/null || true
fi
log "restored iter${ITER_PAD} outputs (.md + state.tsv + CLAUDE.md + figs/) to ${PREV_BRANCH} working tree"

exit 0
