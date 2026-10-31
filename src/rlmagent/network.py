"""Bounded host-side text fetching."""

from __future__ import annotations


async def fetch_text(url: str, max_chars: int = 50000) -> dict:
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
                return {
                    "url": str(response.url),
                    "text": text[:limit],
                    "truncated": truncated or len(text) > limit,
                }
    except httpx.HTTPError as error:
        return {"ok": False, "error": str(error)}
