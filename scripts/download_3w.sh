#!/usr/bin/env bash
# Fetch the Petrobras 3W dataset (v2.0.0, ~1.7 GB of Parquet) without the repo's history or demos.
# Usage: scripts/download_3w.sh [target_dir]
set -euo pipefail

TARGET="${1:-data/raw/3w}"
REPO="https://github.com/petrobras/3W.git"
CLONE="$(mktemp -d)/3W"
trap 'rm -rf "$(dirname "$CLONE")"' EXIT

git clone --depth 1 --filter=blob:none --sparse "$REPO" "$CLONE"
git -C "$CLONE" sparse-checkout set --no-cone \
  '/dataset/[0-9]/*.parquet' /dataset/dataset.ini /dataset/LICENSE-CC-BY

# Keep only the checked-out files; the clone's object store would double the disk footprint.
mkdir -p "$TARGET"
rm -rf "$TARGET/dataset"
mv "$CLONE/dataset" "$TARGET/dataset"

echo "3W instances: $(find "$TARGET/dataset" -name '*.parquet' | wc -l | tr -d ' ')"
