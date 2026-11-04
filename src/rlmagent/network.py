"""Bounded host-side text fetching."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit


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


MAX_REDIRECTS = 5


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_global and not ip.is_multicast


async def _public_address(host: str) -> str:
    """An address for `host`, refusing a name where any address is not public."""
    try:
        found = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ValueError(f"cannot resolve {host}: {error}") from None
    addresses = [item[4][0] for item in found]
    if not addresses or not all(_is_public(a) for a in addresses):
        raise ValueError(f"{host} is a private or local address; fetch_url only reaches public "
                         "ones unless rlm-agent runs with --allow-private-fetch")
    return addresses[0]


async def _open(client, url: str, allow_private: bool):
    """One request. Public hosts are reached at the address just checked, so DNS cannot
    answer differently for the connection; the name still goes in Host and TLS."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("fetch_url requires an HTTP or HTTPS URL")
    if allow_private:
        return await client.send(client.build_request("GET", url), stream=True)
    address = await _public_address(parts.hostname)
    shown = f"[{address}]" if ":" in address else address
    netloc = f"{shown}:{parts.port}" if parts.port else shown
    request = client.build_request(
        "GET", urlunsplit(parts._replace(netloc=netloc)),
        headers={"Host": parts.netloc.rsplit("@", 1)[-1]},
        extensions={"sni_hostname": parts.hostname},
    )
    return await client.send(request, stream=True)


async def fetch_text(url: str, max_chars: int = 50000, raw: bool = False,
                     allow_private: bool = False) -> dict:
    import httpx

    limit = max(1, min(int(max_chars), 50000))
    data = bytearray()
    truncated = False
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            current = url
            for _ in range(MAX_REDIRECTS + 1):
                response = await _open(client, current, allow_private)
                if not response.is_redirect:
                    break
                current = urljoin(current, response.headers["location"])
                await response.aclose()
            else:
                await response.aclose()
                return {"ok": False, "error": f"more than {MAX_REDIRECTS} redirects"}
            try:
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
                    "url": current,
                    "text": text[:limit],
                    "truncated": truncated or len(text) > limit,
                }
            finally:
                await response.aclose()
    except httpx.HTTPError as error:
        return {"ok": False, "error": str(error)}
