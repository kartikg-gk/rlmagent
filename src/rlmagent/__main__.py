"""`python -m rlmagent` runs the same command as `rlm-agent`.

Useful where a policy blocks the generated `rlm-agent.exe` launcher but allows Python.
"""

from rlmagent.cli.main import main

raise SystemExit(main())
