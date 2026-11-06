#!/usr/bin/env bash
# Laptop-side helper. Run it from your Ubuntu laptop, never on the VM.
#
#   ./scripts/vmctl.sh init [HOST] [USER]   connect a (new) VM: saves details, SSH key + config
#   ./scripts/vmctl.sh setup [--freeze]     push project to the VM and run bootstrap_vm.sh there
#   ./scripts/vmctl.sh shell                SSH in and attach to (or create) the tmux session "main"
#   ./scripts/vmctl.sh push                 laptop -> VM   (code, configs, checkpoints, results ...)
#   ./scripts/vmctl.sh pull                 VM -> laptop
#   ./scripts/vmctl.sh sync ["message"]     pull, then git commit + git push
#   ./scripts/vmctl.sh status               GPU, disk, and tmux sessions on the VM
#   ./scripts/vmctl.sh ssh-config           (re)write the "dlba-vm" entry in ~/.ssh/config
#
# Environment switches:
#   DRY=1                 show what rsync would copy without copying
#   TORCH_INDEX=URL       forwarded to the VM by `setup` (only if the default torch wheel fails)
#   PUSH_EXCLUDE_CACHE=1  do not push cache/ (cached hidden states can be recomputed)
#
# Rules rsync follows: never deletes anything, and never overwrites a file that is
# NEWER on the receiving side (--update). So pushing and pulling is safe to repeat.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.vmenv"
ALIAS="dlba-vm"
KEY="$HOME/.ssh/id_ed25519_vcl"
REMOTE_DIR="dlba-project"          # relative to the VM home directory
BLOCK_START="# >>> dlba-vm (managed by vmctl.sh) >>>"
BLOCK_END="# <<< dlba-vm <<<"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*"; }

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}"; }

load_env() {
  [ -f "$ENV_FILE" ] || die "No VM saved yet. Run: ./scripts/vmctl.sh init"
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  [ -n "${VM_HOST:-}" ] && [ -n "${VM_USER:-}" ] || die ".vmenv is incomplete. Run: ./scripts/vmctl.sh init"
}

write_ssh_config() {
  load_env
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  local cfg="$HOME/.ssh/config"
  touch "$cfg" && chmod 600 "$cfg"
  # remove an older managed block, if any
  awk -v s="$BLOCK_START" -v e="$BLOCK_END" '
    $0==s {skip=1; next}
    $0==e {skip=0; next}
    !skip {print}
  ' "$cfg" | cat -s > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
  chmod 600 "$cfg"
  cat >> "$cfg" <<EOF

$BLOCK_START
Host $ALIAS
    HostName $VM_HOST
    User $VM_USER
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
    ServerAliveCountMax 6
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 4h
$BLOCK_END
EOF
  info "SSH entry '$ALIAS' -> $VM_USER@$VM_HOST written to ~/.ssh/config"
}

cmd_init() {
  local host="${1:-}" user="${2:-}"
  [ -f "$ENV_FILE" ] && source "$ENV_FILE" || true
  if [ -z "$host" ]; then
    read -r -p "VM host/IP from the VCL reservation [${VM_HOST:-}]: " host
    host="${host:-${VM_HOST:-}}"
  fi
  if [ -z "$user" ]; then
    read -r -p "VM username [${VM_USER:-}]: " user
    user="${user:-${VM_USER:-}}"
  fi
  [ -n "$host" ] && [ -n "$user" ] || die "Host and username are both required."
  printf 'VM_HOST=%q\nVM_USER=%q\n' "$host" "$user" > "$ENV_FILE"

  # A new reservation reuses IPs but has a new host key: forget the old one.
  ssh-keygen -R "$host" >/dev/null 2>&1 || true
  # Close any stale shared connection to the previous VM.
  ssh -O exit "$ALIAS" >/dev/null 2>&1 || true

  if [ ! -f "$KEY" ]; then
    info "Creating SSH key $KEY (press Enter for an empty passphrase, or set one)"
    ssh-keygen -t ed25519 -f "$KEY" -C "dlba-vcl"
  fi
  write_ssh_config
  info "Installing your key on the VM (you will be asked for your password once) ..."
  ssh-copy-id -i "$KEY.pub" "$ALIAS" || info "ssh-copy-id failed; password logins will still work."
  info "Testing connection ..."
  ssh "$ALIAS" 'echo "Connected to $(hostname) as $(whoami)"'
  info "Done. VS Code: F1 -> 'Remote-SSH: Connect to Host...' -> $ALIAS"
}

EXCLUDES=(--exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc'
          --exclude '.ipynb_checkpoints/' --exclude '.vmenv' --exclude '.DS_Store')

rsync_opts() {
  local opts=(-avh --update --partial --info=progress2 "${EXCLUDES[@]}")
  [ "${DRY:-0}" = "1" ] && opts+=(-n)
  printf '%s\n' "${opts[@]}"
}

cmd_push() {
  load_env
  local opts=() line
  while IFS= read -r line; do opts+=("$line"); done < <(rsync_opts)
  [ "${PUSH_EXCLUDE_CACHE:-0}" = "1" ] && opts+=(--exclude 'cache/')
  info "Pushing $ROOT/  ->  $ALIAS:~/$REMOTE_DIR/"
  rsync "${opts[@]}" "$ROOT/" "$ALIAS:$REMOTE_DIR/"
}

cmd_pull() {
  load_env
  local opts=() line
  while IFS= read -r line; do opts+=("$line"); done < <(rsync_opts)
  info "Pulling $ALIAS:~/$REMOTE_DIR/  ->  $ROOT/"
  rsync "${opts[@]}" "$ALIAS:$REMOTE_DIR/" "$ROOT/"
}

cmd_setup() {
  load_env
  cmd_push
  info "Running bootstrap on the VM ..."
  # shellcheck disable=SC2029
  # TORCH_INDEX (if you set it on the laptop) is forwarded to the VM.
  ssh -t "$ALIAS" "cd ~/$REMOTE_DIR && TORCH_INDEX=$(printf '%q' "${TORCH_INDEX:-}") bash scripts/bootstrap_vm.sh $(printf '%q ' "$@")"
}

cmd_shell() {
  load_env
  ssh -t "$ALIAS" "tmux new-session -A -s main"
}

cmd_sync() {
  local msg="${1:-sync $(date '+%Y-%m-%d %H:%M')}"
  cmd_pull
  cd "$ROOT"
  [ -d .git ] || die "This folder is not a git repository yet (see INSTRUCTIONS.md, Part A)."
  git add -A
  if git diff --cached --quiet; then
    info "Git: nothing new to commit."
  else
    git commit -m "$msg"
  fi
  if git remote get-url origin >/dev/null 2>&1; then
    git push -u origin HEAD
  else
    info "Git: no 'origin' remote set, so nothing was pushed to GitHub."
  fi
}

cmd_status() {
  load_env
  ssh "$ALIAS" 'hostname; echo; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; echo; df -h ~ | tail -1; echo; echo "tmux sessions:"; tmux ls 2>&1 || true'
}

case "${1:-help}" in
  init)        shift; cmd_init "$@" ;;
  setup)       shift; cmd_setup "$@" ;;
  shell|ssh)   cmd_shell ;;
  push)        cmd_push ;;
  pull)        cmd_pull ;;
  sync)        shift; cmd_sync "$@" ;;
  status)      cmd_status ;;
  ssh-config)  write_ssh_config ;;
  help|-h|--help) usage ;;
  *)           usage; exit 2 ;;
esac
