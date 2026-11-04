import sys
sys.path.insert(0, ".")
from summary import summary

assert summary() == {"total_value": 7.5, "low": ["gear", "nut"]}
