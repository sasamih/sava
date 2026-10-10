#!/usr/bin/env bash
# Runs a command (default: an interactive shell) in the SAVA dev container,
# with this repository mounted at /workspace. Builds the image on first use.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SAVA_DEV_IMAGE:-sava-dev}"
MOUNT="$REPO"
if command -v cygpath >/dev/null 2>&1; then
  MOUNT="$(cygpath -w "$REPO")"   # Git Bash on Windows
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO"
fi

TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS=(-it); fi
if [ $# -eq 0 ]; then set -- bash; fi

MSYS_NO_PATHCONV=1 exec docker run --rm "${TTY_FLAGS[@]}" \
  -v "$MOUNT:/workspace" -w /workspace "$IMAGE" "$@"
