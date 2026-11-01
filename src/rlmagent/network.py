"""Bounded host-side text fetching."""

from __future__ import annotations

from html.parser import HTMLParser


class _Readable(HTMLParser):
    """The text of a page, without its scripts, styles and tags."""

    BLOCKS = {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self.hidden += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def readable(html: str) -> str:
    page = _Readable()
    page.feed(html)
    lines = (" ".join(line.split()) for line in "".join(page.parts).splitlines())
    return "\n".join(line for line in lines if line)


async def fetch_text(url: str, max_chars: int = 50000, raw: bool = False) -> dict:
    from urllib.parse import urlsplit

    import httpx

    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("fetch_url requires an HTTP or HTTPS URL")
    limit = max(1, min(int(max_chars), 50000))
    data = bytearray()
    truncated = False
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True, max_redirects=5) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    remaining = limit * 4 + 1 - len(data)
                    data.extend(chunk[:remaining])
                    if len(data) >= limit * 4 + 1:
                        truncated = True
                        break
                text = data.decode(response.encoding or "utf-8", errors="replace")
                if not raw and "html" in response.headers.get("content-type", ""):
                    text = readable(text)
                return {
                    "url": str(response.url),
                    "text": text[:limit],
                    "truncated": truncated or len(text) > limit,
                }
    except httpx.HTTPError as error:
        return {"ok": False, "error": str(error)}
