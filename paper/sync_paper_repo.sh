#!/bin/bash
# Keep paper/ (code repository, branch qg2l-lyapunov-diag) and the Overleaf-linked paper repository
# (github.com/maxbeauchamp/4dvarnet-qg2l-paper) in sync, by plain file copies + ordinary commits.
#
# Roles (no conflicts possible):
#   - generated assets  (tables/, figures/, results/, scripts/, README.md, this script): code repo is the source
#   - text              (main.tex, refs.bib, anything else written on Overleaf):        paper repo is the source
#
#   bash paper/sync_paper_repo.sh push   # code -> paper repo (after a sweep); then "Pull changes" in Overleaf
#   bash paper/sync_paper_repo.sh pull   # paper repo -> code (after "Push to GitHub" in Overleaf); then commit here
#
# The paper repository is cloned next to the code repository (override with PAPER_DIR=...).
set -euo pipefail
CODE=$(git rev-parse --show-toplevel)
PAPER_URL=${PAPER_URL:-https://github.com/maxbeauchamp/4dvarnet-qg2l-paper.git}
PAPER_DIR=${PAPER_DIR:-$(dirname "$CODE")/4dvarnet-qg2l-paper}
GENERATED=(tables figures results scripts README.md sync_paper_repo.sh)

[ -d "$PAPER_DIR/.git" ] || git clone "$PAPER_URL" "$PAPER_DIR"
git -C "$PAPER_DIR" pull --ff-only

case "${1:-}" in
  push)
    for f in "${GENERATED[@]}"; do
        [ -e "$CODE/paper/$f" ] && rsync -a --delete "$CODE/paper/$f" "$PAPER_DIR/"
    done
    git -C "$PAPER_DIR" add -A "${GENERATED[@]}"
    if git -C "$PAPER_DIR" diff --cached --quiet; then
        echo "paper repo already up to date"
    else
        git -C "$PAPER_DIR" commit -q -m "Update generated assets from code $(git -C "$CODE" rev-parse --short HEAD)"
        git -C "$PAPER_DIR" push
        echo "pushed: now click 'Pull GitHub changes into Overleaf' in the Overleaf GitHub menu"
    fi
    ;;
  pull)
    excl=(--exclude=.git)
    for f in "${GENERATED[@]}"; do excl+=(--exclude="/$f"); done
    rsync -a "${excl[@]}" "$PAPER_DIR/" "$CODE/paper/"
    git -C "$CODE" status --short paper/
    echo "text copied into paper/: review, then commit and push the code branch"
    ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
