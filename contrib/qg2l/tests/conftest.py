import sys
from pathlib import Path

# make ``qg2l`` importable as a top-level package (numpy-only tests do not need
# hydra / the other contrib packages imported by contrib/__init__.py)
CONTRIB = Path(__file__).resolve().parents[2]
REPO = CONTRIB.parent
for p in (str(CONTRIB), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
