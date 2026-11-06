#!/usr/bin/env bash
# Run this ON THE VM, from the project folder (or let `vmctl.sh setup` run it for you).
#
#   bash scripts/bootstrap_vm.sh              # rebuild the environment on a fresh VM
#   bash scripts/bootstrap_vm.sh --freeze     # first time only: after verification passes,
#                                             # write exact versions to requirements.txt
#   bash scripts/bootstrap_vm.sh --recreate   # delete and rebuild the virtual environment
#   bash scripts/bootstrap_vm.sh --skip-verify
#
# Optional environment variables:
#   VENV_DIR     where the venv lives            (default: ~/venv-dlba)
#   TORCH_INDEX  extra pip index for a specific CUDA build of PyTorch,
#                e.g. https://download.pytorch.org/whl/cu124   (default: plain PyPI wheel)
#
# The script is idempotent: running it twice is safe.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venv-dlba}"
TORCH_INDEX="${TORCH_INDEX:-}"
FREEZE=0
RECREATE=0
VERIFY=1

for arg in "$@"; do
  case "$arg" in
    --freeze)      FREEZE=1 ;;
    --recreate)    RECREATE=1 ;;
    --skip-verify) VERIFY=0 ;;
    -h|--help)     sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)             echo "Unknown option: $arg (try --help)"; exit 2 ;;
  esac
done

mkdir -p "$PROJECT_DIR/logs"
LOG="$PROJECT_DIR/logs/bootstrap_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { printf '\n==== %s ====\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*"; }
die()  { printf 'ERROR: %s\n' "$*"; exit 1; }

cd "$PROJECT_DIR"
echo "Project dir : $PROJECT_DIR"
echo "Venv dir    : $VENV_DIR"
echo "Log file    : $LOG"

# ---------------------------------------------------------------- 1. machine info
step "1/8 Machine check"
echo "OS      : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "Python  : $(python3 --version 2>&1 || echo 'python3 not found')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvidia-smi | grep -i "CUDA Version" || true
else
  warn "nvidia-smi not found. This VM may not have a GPU. Pick the 'Ubuntu 24 GPU with Cuda' image."
fi
free -h | sed -n '1,2p'
df -h "$HOME" | tail -1

# ---------------------------------------------------------------- 2. system packages
step "2/8 System packages"
need=(tmux git rsync python3-venv python3-pip)
missing=()
for p in "${need[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing: ${missing[*]}"
  if command -v sudo >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -y || warn "apt-get update failed"
    sudo apt-get install -y "${missing[@]}" || warn "apt-get install failed"
  else
    warn "sudo is not available; cannot install: ${missing[*]}"
  fi
else
  echo "All system packages already installed."
fi
command -v tmux  >/dev/null 2>&1 || warn "tmux is still missing (sessions will not survive disconnects)"
command -v rsync >/dev/null 2>&1 || warn "rsync is missing on the VM (push/pull will fail)"
python3 - <<'PY' || die "python3 >= 3.10 with the venv module is required (install python3-venv)."
import sys, venv, ensurepip
assert sys.version_info >= (3, 10), sys.version
PY

# ---------------------------------------------------------------- 3. folders
step "3/8 Project folders"
bash "$PROJECT_DIR/scripts/make_dirs.sh"

# ---------------------------------------------------------------- 4. virtual environment
step "4/8 Virtual environment"
if [ "$RECREATE" -eq 1 ] && [ -d "$VENV_DIR" ]; then
  echo "Removing existing venv (--recreate)"
  rm -rf "$VENV_DIR"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
  echo "Created venv."
else
  echo "Venv already exists, reusing."
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip setuptools wheel
echo "Using $(which python) ($(python --version 2>&1))"

# ---------------------------------------------------------------- 5. install packages
step "5/8 Install packages"
REQ="$PROJECT_DIR/requirements.txt"
if [ -f "$REQ" ]; then
  echo "Installing FROZEN versions from requirements.txt"
else
  REQ="$PROJECT_DIR/requirements.in"
  echo "No requirements.txt yet; installing from requirements.in (run with --freeze afterwards)"
fi
PIP_EXTRA=()
if [ -n "$TORCH_INDEX" ]; then
  PIP_EXTRA+=(--extra-index-url "$TORCH_INDEX")
  echo "Extra pip index: $TORCH_INDEX"
fi
python -m pip install ${PIP_EXTRA[@]+"${PIP_EXTRA[@]}"} -r "$REQ" \
  || die "pip install failed. If it mentions a '+cuXXX' torch version, set TORCH_INDEX and rerun."
python -m pip check || warn "pip check reported dependency conflicts (see above)"

# ---------------------------------------------------------------- 6. quality of life
step "6/8 tmux config and shell shortcut"
if [ ! -f "$HOME/.tmux.conf" ]; then
  cat > "$HOME/.tmux.conf" <<'EOF'
set -g mouse on
set -g history-limit 50000
EOF
  echo "Wrote ~/.tmux.conf (mouse scrolling on, 50k lines of history)"
else
  echo "~/.tmux.conf already exists, leaving it alone."
fi
ALIAS_LINE="alias dlba='source $VENV_DIR/bin/activate && cd $PROJECT_DIR'"
if ! grep -qF "alias dlba=" "$HOME/.bashrc" 2>/dev/null; then
  printf '\n# dlba project shortcut\n%s\n' "$ALIAS_LINE" >> "$HOME/.bashrc"
  echo "Added shortcut: type 'dlba' to activate the venv and cd into the project."
else
  echo "Shortcut 'dlba' already in ~/.bashrc."
fi

# ---------------------------------------------------------------- 7. verify
step "7/8 Verify environment"
VERIFY_OK=1
if [ "$VERIFY" -eq 1 ]; then
  if python "$PROJECT_DIR/scripts/verify_env.py"; then
    echo "Verification PASSED."
  else
    VERIFY_OK=0
    warn "Verification FAILED. Fix the failures above before continuing."
  fi
else
  echo "Skipped (--skip-verify)."
fi

# ---------------------------------------------------------------- 8. freeze
step "8/8 Freeze"
{
  echo "Generated: $(date -Is)"
  echo "Host: $(hostname)"
  echo "Python: $(python --version 2>&1)"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi
} > "$PROJECT_DIR/logs/env_report.txt" 2>&1 || true

if [ "$FREEZE" -eq 1 ]; then
  if [ "$VERIFY_OK" -eq 1 ] && [ "$VERIFY" -eq 1 ]; then
    python -m pip freeze --exclude-editable > "$PROJECT_DIR/requirements.txt"
    echo "Wrote requirements.txt ($(wc -l < "$PROJECT_DIR/requirements.txt") pinned packages)."
    echo "Pull it to your laptop and commit it: ./scripts/vmctl.sh sync \"freeze requirements\""
  else
    warn "Not freezing, because verification did not pass."
  fi
else
  echo "Not freezing (use --freeze after the first successful verification)."
fi

printf '\nDone. Next: run `dlba` (or `source %s/bin/activate`) in a new shell.\n' "$VENV_DIR"
[ "$VERIFY_OK" -eq 1 ] || exit 1
