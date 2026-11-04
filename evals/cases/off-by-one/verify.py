import sys
sys.path.insert(0, ".")
from window import last_n

assert last_n([1, 2, 3, 4], 2) == [3, 4]
assert last_n([1, 2, 3], 3) == [1, 2, 3]
assert last_n(["a"], 1) == ["a"]
