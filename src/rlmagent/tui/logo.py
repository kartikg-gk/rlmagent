"""The name at the top of the conversation, in small grey block letters.

It sits at the head of the transcript, so it scrolls away as the work does.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

#: Five pixel rows per letter; drawn three lines high with half blocks.
_GLYPHS = {
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "M": ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
    "-": ["    ", "    ", "████", "    ", "    "],
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "G": [" ████", "█    ", "█  ██", "█   █", " ███ "],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
}

WORD = "RLM-AGENT"
GREY = "#5a5a5a"


def logo_lines(word: str = WORD) -> list[str]:
    """The word in three lines: each line packs two pixel rows into half blocks."""
    rows = [" ".join(_GLYPHS[ch][row] for ch in word) for row in range(5)] + [""]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width) for r in rows]
    pick = {(True, True): "█", (True, False): "▀", (False, True): "▄", (False, False): " "}
    return ["".join(pick[(top[i] != " ", low[i] != " ")] for i in range(width)).rstrip()
            for top, low in zip(rows[0::2], rows[1::2])]


class Logo(Static):
    """The block-letter name, in grey."""

    def __init__(self) -> None:
        super().__init__(Text("\n".join(logo_lines()), style=GREY), classes="logo")


__all__ = ["Logo", "logo_lines"]
