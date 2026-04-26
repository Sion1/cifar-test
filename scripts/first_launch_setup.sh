#!/usr/bin/env bash
# Interactive onboarding for a fresh agentic-research-assistant checkout.
#
# The autoresearch loop runs many experiments unattended and a fresh
# environment usually has 4 small surprises waiting for the user (or for an
# autonomous agent that gets handed the repo): which Python env to use, where
# to put the dataset, which GPUs to skip, whether to enable wandb / git push.
# Doing this once interactively at the start saves a lot of guess-and-fix
# wandering later.
#
# All four integrations are STRONGLY RECOMMENDED. You may decline any of
# them; the loop runs correctness-equivalently without them.
#
# Writes:
#   state/.env               — sourced by loop.sh / run_experiment.sh
#   state/.onboarding_done   — sentinel that unblocks Step 0.5 in loop.sh

set -u
cd "$(dirname "$0")/.."

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
BOLD='\033[1m'
NC='\033[0m'

prompt_yn() {
    local q="$1" def="${2:-y}" hint reply
    if [ "$def" = "y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    while true; do
        printf "%b%s%b %s " "$BOLD" "$q" "$NC" "$hint"
        read -r reply
        reply="${reply:-$def}"
        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo])     return 1 ;;
        esac
    done
}

prompt_str() {
    local q="$1" def="${2:-}" reply
    if [ -n "$def" ]; then
        printf "%b%s%b [default: %s]: " "$BOLD" "$q" "$NC" "$def"
    else
        printf "%b%s%b: " "$BOLD" "$q" "$NC"
    fi
    read -r reply
    printf '%s' "${reply:-$def}"
}

if [ ! -t 0 ]; then
    echo "ERROR: this script must run interactively (a TTY is required)." >&2
    echo "       run it directly in your shell, not via pipe / nohup."   >&2
    exit 2
fi

mkdir -p state
ENV_FILE=state/.env
: > "$ENV_FILE"

echo
printf "%b== agentic-research-assistant first-launch setup ==%b\n" "$BOLD" "$NC"
echo
echo "Four small choices, all strongly recommended. You may decline any."
echo

# ---- 1. Python environment -------------------------------------------------
echo
printf "%b[1/4] Python environment%b\n" "$BOLD" "$NC"
PYTHON_CMD=$(prompt_str "Path or command for python with torch + torchvision installed" "$(command -v python3 || echo python3)")
if ! "$PYTHON_CMD" -c 'import torch, torchvision' >/dev/null 2>&1; then
    printf "%b'%s' cannot import torch + torchvision.%b\n" "$RED" "$PYTHON_CMD" "$NC"
    echo "  Either:"
    echo "    a) activate a conda env with torch first, then re-run this script;"
    echo "    b) install:  pip install torch torchvision pyyaml scikit-learn matplotlib wandb"
    echo "  Then re-run this script."
    exit 3
fi
TORCH_VER=$("$PYTHON_CMD" -c 'import torch; print(torch.__version__)')
CUDA_OK=$("$PYTHON_CMD" -c 'import torch; print(torch.cuda.is_available())')
printf "%bpython=%s torch=%s cuda=%s%b\n" "$GREEN" "$PYTHON_CMD" "$TORCH_VER" "$CUDA_OK" "$NC"
echo "export PYTHON=$PYTHON_CMD" >> "$ENV_FILE"

# ---- 2. Dataset root --------------------------------------------------------
echo
printf "%b[2/4] Dataset root%b\n" "$BOLD" "$NC"
echo "  Where should CIFAR-10 (and other datasets) live?"
echo "  Default ./data/ stays inside the repo (safe, ~170 MB for CIFAR-10)."
echo "  Use an absolute path if you want to share a dataset cache across runs."
DATA_ROOT=$(prompt_str "Dataset root directory" "./data")
mkdir -p "$DATA_ROOT" 2>/dev/null || {
    printf "%bcannot create %s — pick another path%b\n" "$RED" "$DATA_ROOT" "$NC"
    exit 4
}
printf "%bdataset root: %s%b\n" "$GREEN" "$DATA_ROOT" "$NC"
echo "export AUTORES_DATA_ROOT=$DATA_ROOT" >> "$ENV_FILE"

# ---- 3. GPU skip-list -------------------------------------------------------
echo
printf "%b[3/4] GPU selection%b\n" "$BOLD" "$NC"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /'
    echo
    echo "  GPUs with < AUTORES_MIN_FREE_MB (default 15 GB) free are auto-skipped"
    echo "  by run_experiment.sh. Add indices here only if you also want to reserve"
    echo "  full ones for other work (e.g. a shared GPU 0 used by your IDE)."
    SKIP=$(prompt_str "Comma-separated GPU indices to FORCE-skip (blank = none)" "")
    if [ -n "$SKIP" ]; then
        echo "export AUTORES_SKIP_GPUS=$SKIP" >> "$ENV_FILE"
        printf "%bskip list: %s%b\n" "$GREEN" "$SKIP" "$NC"
    else
        printf "%bno forced skip%b\n" "$GREEN" "$NC"
    fi
else
    printf "%bnvidia-smi not found — assuming CPU-only%b\n" "$YELLOW" "$NC"
fi

# ---- 4. wandb ---------------------------------------------------------------
echo
printf "%b[4a/4] Weights & Biases (wandb)%b\n" "$BOLD" "$NC"
echo "  Tracks per-epoch metrics + hyperparameters with a free academic tier."
echo "  Without it: only local runs/<exp>/history.json available."
WANDB_OK=0
if prompt_yn "Enable wandb tracking? (strongly recommended)" "y"; then
    if ! "$PYTHON_CMD" -c 'import wandb' >/dev/null 2>&1; then
        printf "%bwandb not in this Python env.%b Install: pip install wandb\n" "$RED" "$NC"
        echo "  Then re-run this script."
        exit 5
    fi
    if "$PYTHON_CMD" -m wandb status 2>/dev/null | grep -qi "logged in"; then
        printf "%balready logged in%b\n" "$GREEN" "$NC"
    else
        echo "Launching 'wandb login' — paste API key from https://wandb.ai/authorize:"
        if ! "$PYTHON_CMD" -m wandb login; then
            printf "%bwandb login failed.%b You can re-run this script after fixing it.\n" "$RED" "$NC"
            exit 6
        fi
    fi
    PROJECT=$(prompt_str "wandb project name" "agentic-research-cifar10-demo")
    echo "export WANDB_PROJECT=$PROJECT" >> "$ENV_FILE"
    WANDB_OK=1
else
    printf "%bwandb disabled.%b Re-run this script anytime to enable.\n" "$YELLOW" "$NC"
fi

# ---- 4b. git / GitHub -------------------------------------------------------
echo
printf "%b[4b/4] git + (optional) GitHub remote%b\n" "$BOLD" "$NC"
echo "  Each analyzed iteration becomes a branch autoresearch/iter-NNN with"
echo "  config diff, log, viz outputs and consensus verdict. Pushable to GitHub"
echo "  for browsable history + multi-host sync."
GIT_OK=0; PUSH_OK=0
if prompt_yn "Enable per-iter git commits? (strongly recommended)" "y"; then
    if ! command -v git >/dev/null 2>&1; then
        printf "%bgit not installed.%b Install it and re-run.\n" "$RED" "$NC"
        exit 7
    fi
    if [ ! -d .git ] && [ ! -f .git ]; then
        echo "No git repo here yet. Initializing..."
        git init -b main 2>/dev/null || git init
    fi
    git config user.name  >/dev/null 2>&1 || git config user.name  "autoresearch"
    git config user.email >/dev/null 2>&1 || git config user.email "autoresearch@localhost"
    GIT_OK=1
    printf "%bgit ready%b\n" "$GREEN" "$NC"

    echo
    if prompt_yn "Also push each iter branch to a GitHub remote? (recommended)" "y"; then
        EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || true)
        if [ -n "$EXISTING_REMOTE" ]; then
            printf "  remote 'origin' already set to: %s\n" "$EXISTING_REMOTE"
        else
            REMOTE_URL=$(prompt_str "GitHub remote URL (blank = skip)" "")
            if [ -n "$REMOTE_URL" ]; then
                git remote add origin "$REMOTE_URL" || true
                EXISTING_REMOTE="$REMOTE_URL"
            fi
        fi
        if [ -n "${EXISTING_REMOTE:-}" ]; then
            echo "export AUTORES_GIT_AUTOPUSH=1" >> "$ENV_FILE"
            PUSH_OK=1
            printf "%bauto-push enabled%b\n" "$GREEN" "$NC"
        else
            echo "export AUTORES_GIT_AUTOPUSH=0" >> "$ENV_FILE"
            printf "%bno remote — auto-push disabled%b\n" "$YELLOW" "$NC"
        fi
    else
        echo "export AUTORES_GIT_AUTOPUSH=0" >> "$ENV_FILE"
        printf "%bauto-push disabled%b\n" "$YELLOW" "$NC"
    fi
else
    printf "%bgit commits disabled.%b Iter outputs will only live in figs/ + logs/.\n" "$YELLOW" "$NC"
fi

# ---- sentinel ---------------------------------------------------------------
{
    echo "onboarding_done_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "python=$PYTHON_CMD"
    echo "torch=$TORCH_VER"
    echo "cuda_available=$CUDA_OK"
    echo "data_root=$DATA_ROOT"
    echo "wandb_enabled=$WANDB_OK"
    echo "git_enabled=$GIT_OK"
    echo "git_autopush_enabled=$PUSH_OK"
} > state/.onboarding_done

echo
printf "%bSetup complete.%b Choices written to state/.env, sentinel at state/.onboarding_done.\n" "$GREEN" "$NC"
echo
echo "Quick smoketest (1 epoch on the GPU you just configured):"
echo "  source state/.env"
echo "  bash run_experiment.sh configs/cifar10_resnet34.yaml --smoketest"
echo
echo "Full run (60 epochs):"
echo "  bash run_experiment.sh configs/cifar10_resnet34.yaml"
echo
