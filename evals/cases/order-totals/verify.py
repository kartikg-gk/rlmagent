import json

with open("totals.json", encoding="utf-8") as f:
    totals = json.load(f)
assert totals == {"ana": 49.75, "ben": 8.75, "cleo": 12.0}, totals
assert list(totals) == sorted(totals)
