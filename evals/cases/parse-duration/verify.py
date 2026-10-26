import sys
sys.path.insert(0, ".")
from duration import parse_duration

assert parse_duration("1h30m") == 5400
assert parse_duration("45s") == 45
assert parse_duration("2h5m10s") == 7510
for bad in ("", "1x", "h", "30m1h"):
    try:
        parse_duration(bad)
    except ValueError:
        continue
    raise AssertionError(f"{bad!r} should raise ValueError")
