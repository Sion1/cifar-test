#!/usr/bin/env bash
# Wrapper that loop.sh invokes to launch one experiment.
#
# Responsibilities (you should keep ALL of these when adapting for your
# domain — the framework's reaper / scheduler depends on them):
#   1. Pick a free GPU (skip ones below VRAM threshold)
#   2. OOM preflight (refuse to launch if a sibling process would push the
#      device over the per-process footprint cap)
#   3. Append a `running` row to state/iterations.tsv with PID + log path
#   4. nohup background launch of train.py
#   5. The *child* (train.py) writes the FINISH-line that the reaper reads
#
# Usage (from loop.sh's propose step):
#   bash run_experiment.sh <CONFIG_PATH> <ITER_NUM>
#
# Env knobs:
#   AUTORES_SKIP_GPUS                  ← comma-sep GPU indices to skip
#   AUTORES_MIN_FREE_MB        24000   ← only consider GPUs with this much free
#   AUTORES_PER_PROCESS_FOOTPRINT_MB   ← OOM preflight cap (in MB)
#   PYTHON                     python3 ← override interpreter
set -u

CONFIG="${1:-}"
ITER_NUM="${2:-0}"
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "usage: $0 <config.yaml> <iter_num>" >&2
    exit 2
fi

cd "$(dirname "$0")"
mkdir -p logs state runs

ITER_PAD=$(printf '%03d' "$ITER_NUM")
EXP_NAME=$(grep -E '^exp_name:' "$CONFIG" | head -1 | awk -F'"' '{print $2}')
EXP_NAME="${EXP_NAME:-$(basename "${CONFIG%.*}")}"
LOG="logs/exp_${ITER_PAD}_${EXP_NAME}.log"

# ------- GPU selection --------------------------------------------------
SKIP_GPUS="${AUTORES_SKIP_GPUS-}"
MIN_FREE_MB="${AUTORES_MIN_FREE_MB:-24000}"
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -rn \
    | awk -F', ' -v skip="$SKIP_GPUS" -v thresh="$MIN_FREE_MB" '
        BEGIN { n = split(skip, s, ","); for (i=1; i<=n; i++) if (s[i] != "") skipm[s[i]] = 1 }
        $2+0 >= thresh && !($1 in skipm) { print $1; exit }')
if [ -z "$GPU" ]; then
    # No GPU available — exit so loop tick doesn't write a "running" row that
    # immediately fails. Loop will retry next tick when a GPU frees up.
    echo "[run_experiment] no GPU with free mem >= ${MIN_FREE_MB} MB (skipped: ${SKIP_GPUS:-none})" >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# ------- spawn ----------------------------------------------------------
PYTHON="${PYTHON:-python3}"
echo "[$(date '+%F %T')] launching iter $ITER_PAD: $EXP_NAME on GPU $GPU (log: $LOG)" >> logs/driver.log
nohup bash -c "
    set -u
    echo '[$(date)] iter $ITER_PAD START exp=$EXP_NAME gpu=$GPU pid=\$\$'
    $PYTHON -u train.py --config '$CONFIG' --iter-num '$ITER_NUM'
    rc=\$?
    echo '[iter $ITER_PAD] FINISH '\$(date '+%Y-%m-%d %H:%M:%S')' rc='\$rc
" > "$LOG" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true

# ------- ledger update --------------------------------------------------
TS=$(date '+%Y-%m-%dT%H:%M:%S%z')
TSV=state/iterations.tsv
if [ ! -f "$TSV" ]; then
    printf 'iter\tstatus\texp_name\tconfig\tgpu\tpid\tstarted_at\tfinished_at\tbest_metric\tverdict\n' > "$TSV"
fi
printf '%d\trunning\t%s\t%s\t%s\t%s\t%s\t\t\t\n' \
    "$ITER_NUM" "$EXP_NAME" "$CONFIG" "$GPU" "$PID" "$TS" >> "$TSV"

echo "[run_experiment] iter $ITER_PAD launched pid=$PID gpu=$GPU log=$LOG"
exit 0
