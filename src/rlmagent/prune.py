"""`rlm-agent session prune`: find and remove old sessions and their undo snapshots."""

from __future__ import annotations

import os
import re
import shutil
import stat
import sys
from pathlib import Path
from time import time

from rlmagent.harness.session.index import SessionCatalog
from rlmagent.undo import Shadow

_SAFE_ID = re.compile(r"[\w-]+")


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _remove(path: Path) -> None:
    if path.is_file():
        path.unlink()
        return

    def writable(function, name, _error):  # git marks its objects read-only on Windows
        os.chmod(name, stat.S_IWRITE)
        function(name)

    shutil.rmtree(path, onexc=writable)


def prune(base: Path, home: Path, days: int, *, apply: bool = False, out=sys.stdout) -> int:
    """List what is older than `days`; removes it only when `apply` is set."""
    catalog = SessionCatalog(base)
    cutoff = time() - days * 86400
    metas = catalog.list_all()
    old = [m for m in metas if m.updated_at < cutoff and _SAFE_ID.fullmatch(m.session_id)]
    kept = {Shadow(m.cwd, home).git_dir for m in metas if m not in old and m.cwd}
    targets = [f for m in old for f in base.glob(f"{m.session_id}.*")]
    shadows = {Shadow(m.cwd, home).git_dir for m in old if m.cwd} - kept
    targets += [s for s in shadows if s.exists()]
    total = sum(_size(t) for t in targets)
    print(f"{len(old)} of {len(metas)} sessions are older than {days} days; "
          f"{total / 1_000_000:.1f} MB in {len(targets)} files and snapshot folders.", file=out)
    if not apply:
        for target in targets[:20]:
            print(f"  {target}", file=out)
        if old:
            print("Nothing was removed. Add --yes to remove these.", file=out)
        return 0
    for target in targets:
        _remove(target)
    catalog.remove({m.session_id for m in old})
    print("Removed.", file=out)
    return 0
