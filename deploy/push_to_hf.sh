#!/usr/bin/env bash
# Push the committed main branch to a Hugging Face Space (Docker SDK).
# Usage (from the repo root): bash deploy/push_to_hf.sh <hf-username> <space-name>
# Git asks for your Hugging Face username and a write access token the first time; type them yourself.
set -euo pipefail
user="$1"; space="$2"
repo="$(git rev-parse --show-toplevel)"
header="$repo/deploy/hf_readme_header.md"
tmp="$(mktemp -d)"

git -C "$repo" worktree add -f --detach "$tmp" HEAD >/dev/null
cat "$header" "$tmp/README.md" > "$tmp/README.hf" && mv "$tmp/README.hf" "$tmp/README.md"
git -C "$tmp" add README.md
git -C "$tmp" commit -q -m "Hugging Face Space config"
git -C "$tmp" push -f "https://huggingface.co/spaces/$user/$space" HEAD:main
git -C "$repo" worktree remove --force "$tmp"
echo "Pushed. Watch the build at https://huggingface.co/spaces/$user/$space"
