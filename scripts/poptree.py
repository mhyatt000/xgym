# use in parent directory that contains all the data folders. e.g., python xgym/scripts/poptree.py data/
from __future__ import annotations

from pathlib import Path
import shutil

files = Path().cwd().rglob("*.npz")

for f in files:
    out = f.parent.parent / f"{f.parent.name}_{f.name}"
    shutil.move(f, out)
