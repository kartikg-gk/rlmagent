#!/usr/bin/env bash
# Run each test file on its own; fail if any file fails.
status=0
for f in "${@:-tests/test_*.py}"; do
  for g in $f; do
    out=$(.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider "$g" 2>&1 | tail -1)
    echo "$g: $out"
    case "$out" in *failed*|*error*|*"no tests ran"*) status=1;; esac
  done
done
exit $status
