#!/usr/bin/env bash
# Run files separately and keep complete diagnostics when a file fails.
status=0
for f in "${@:-tests/test_*.py}"; do
  for g in $f; do
    out=$(.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider "$g" 2>&1)
    code=$?
    echo "$g: $(printf '%s\n' "$out" | tail -1)"
    if [ "$code" -ne 0 ]; then
      printf '%s\n' "$out"
      status=1
    fi
  done
done
exit "$status"
