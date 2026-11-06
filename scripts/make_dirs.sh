#!/usr/bin/env bash
# Creates the full project folder structure. Safe to run any number of times.
# Git does not track empty folders, so run this after a fresh clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dirs=(
  configs
  src
  scripts
  notebooks
  data/splits
  checkpoints/M0
  checkpoints/M0plus
  checkpoints/M25
  checkpoints/M50
  cache/hidden_states
  results/labels
  results/signals
  results/metrics
  logs
  figures
)

for d in "${dirs[@]}"; do
  mkdir -p "$ROOT/$d"
done
echo "Project folders ready under $ROOT"
