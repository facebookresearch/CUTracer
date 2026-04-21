#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Symlink CUTracer's in-repo SKILL.md files into the user's home so that
# Devmate (~/.llms/skills) and Claude Code (~/.claude/skills) load them
# globally — not only when CUTracer files are in scope.
#
# Run once per devserver. Idempotent (uses `ln -sfn`).
#
# Usage:
#   bash scripts/setup_local_skills.sh
#
# After running:
#   ~/.llms/skills/sync_install_cuda  → fbsource/.../CUTracer/.llms/skills/sync_install_cuda
#   ~/.claude/skills/sync-install-cuda → fbsource/.../CUTracer/.claude/skills/sync-install-cuda
#
# Edits to the canonical files in the repo take effect immediately —
# the symlink is followed to the live file each time the agent loads it.

set -euo pipefail

# Resolve the CUTracer repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVMATE_SRC="${REPO_DIR}/.llms/skills/sync_install_cuda"
CLAUDE_SRC="${REPO_DIR}/.claude/skills/sync-install-cuda"

DEVMATE_DST_DIR="${HOME}/.llms/skills"
CLAUDE_DST_DIR="${HOME}/.claude/skills"

DEVMATE_DST="${DEVMATE_DST_DIR}/sync_install_cuda"
CLAUDE_DST="${CLAUDE_DST_DIR}/sync-install-cuda"

echo "🔗 Setting up CUTracer agent skills"
echo "    repo:      ${REPO_DIR}"
echo "    home:      ${HOME}"

# Sanity: source dirs must exist before we make symlinks.
for src in "${DEVMATE_SRC}" "${CLAUDE_SRC}"; do
  if [ ! -d "${src}" ]; then
    echo "❌ Missing source skill directory: ${src}" >&2
    exit 1
  fi
done

mkdir -p "${DEVMATE_DST_DIR}" "${CLAUDE_DST_DIR}"

# `ln -sfn` semantics:
#   -s: symbolic
#   -f: force (replace existing)
#   -n: when target is a directory symlink, replace it instead of nesting inside
ln -sfn "${DEVMATE_SRC}" "${DEVMATE_DST}"
ln -sfn "${CLAUDE_SRC}"  "${CLAUDE_DST}"

echo "✅ Devmate skill: ${DEVMATE_DST} -> $(readlink "${DEVMATE_DST}")"
echo "✅ Claude  skill: ${CLAUDE_DST} -> $(readlink "${CLAUDE_DST}")"
echo
echo "Both skills are now globally active. To verify:"
echo "  ls -la ${DEVMATE_DST}"
echo "  ls -la ${CLAUDE_DST}"
echo
echo "To remove later:"
echo "  rm ${DEVMATE_DST} ${CLAUDE_DST}"
