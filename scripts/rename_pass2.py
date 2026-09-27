"""Second rename pass: names the first pass could not tell apart from stream deltas."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = [
    (r"\bDELTA_", "RLM_AGENT_"),
    (r"\bDelta(?=[A-Z])", "RlmAgent"),
    (r"(?<![A-Za-z0-9_)\]])\.delta\b", ".rlm-agent"),
    (r"\bdelta_(home|only|ext_)", r"rlm_agent_\1"),
    (r'"delta\.', '"rlm-agent.'),
    (r'"""delta ', '"""rlm-agent '),
    (r'prog="delta', 'prog="rlm-agent'),
    (r"pip install delta\[", "pip install rlmagent["),
    (r"\bdelta(?= (config|provider|session|v\{)|>|`)", "rlm-agent"),
    (r"<delta home>", "<rlm-agent home>"),
    (r"<delta_home>", "<rlm_agent_home>"),
    (r"\bDelta\b(?!-)", "rlm-agent"),
]
for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "src").rglob("*.tcss")):
    if "__pycache__" in path.parts:
        continue
    text = path.read_text(encoding="utf-8")
    new = text
    for pattern, repl in RULES:
        new = re.sub(pattern, repl, new)
    if new != text:
        path.write_text(new, encoding="utf-8")
