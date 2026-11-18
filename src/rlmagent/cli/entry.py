"""The `rlm-agent` command. Kept small: it answers --version and starts the kernel
before the rest of the application is imported."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args in (["--version"], ["-V"]):
        from importlib.metadata import PackageNotFoundError, version

        try:
            print(f"rlm-agent {version('rlmness-cli')}")
        except PackageNotFoundError:
            print("rlm-agent dev")
        return 0
    try:
        from rlmagent import prewarm

        if prewarm.wanted(args):
            prewarm.start()
    except Exception:  # noqa: BLE001 - a failed head start only costs the time it would have saved
        pass
    from rlmagent.cli.main import main as run

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
