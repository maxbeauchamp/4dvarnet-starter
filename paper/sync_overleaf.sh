#!/bin/bash
# Synchronise paper/ (in the code repository, branch qg2l-lyapunov-diag) with an Overleaf project
# through Overleaf's Git integration (Overleaf menu > Git: copy the project's git URL).
#
#   OVERLEAF=https://git.overleaf.com/<project-id> bash paper/sync_overleaf.sh push   # repo -> Overleaf
#   OVERLEAF=https://git.overleaf.com/<project-id> bash paper/sync_overleaf.sh pull   # Overleaf -> repo
#
# 'pull' merges the edits made on Overleaf back into paper/ (then commit / push the code repo as usual).
set -euo pipefail
: "${OVERLEAF:?set OVERLEAF to the Overleaf project git URL}"
cd "$(git rev-parse --show-toplevel)"
case "${1:-push}" in
  push) git subtree push --prefix paper "$OVERLEAF" master ;;
  pull) git subtree pull --prefix paper "$OVERLEAF" master --squash -m "Merge Overleaf edits into paper/" ;;
  *) echo "usage: $0 push|pull"; exit 1 ;;
esac
