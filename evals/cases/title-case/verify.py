import sys
sys.path.insert(0, ".")
from names import title_case

assert title_case("the  art of war") == "The Art of War"
assert title_case("of mice") == "Of Mice"
assert title_case("") == ""
