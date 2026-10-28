"""Wire-safe tool-call identifiers."""

from __future__ import annotations

import re
from hashlib import sha256

# The narrowest alphabet any supported vendor accepts. Ids already inside it
# pass through untouched so same-vendor replay stays byte-identical.
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DIGEST_CHARS = 40


def wire_call_id(raw: str) -> str:
    """Map ``raw`` to a deterministic id every provider accepts."""
    if _SAFE_ID.fullmatch(raw):
        return raw
    return "tc_" + sha256(raw.encode("utf-8")).hexdigest()[:_DIGEST_CHARS]


__all__ = ["wire_call_id"]
