"""Web search through whichever search API the user has a key for."""

from __future__ import annotations

import os

TIMEOUT = 20.0
MAX_RESULTS = 10

# Checked in this order; the first key found is used.
SERVICES = ("BRAVE_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY")


def configured() -> str | None:
    return next((name for name in SERVICES if os.environ.get(name)), None)


async def _brave(client, key, query, count):
    r = await client.get("https://api.search.brave.com/res/v1/web/search",
                         params={"q": query, "count": count},
                         headers={"X-Subscription-Token": key, "Accept": "application/json"})
    r.raise_for_status()
    return [{"title": i.get("title", ""), "url": i.get("url", ""),
             "snippet": i.get("description", "")} for i in r.json().get("web", {}).get("results", [])]


async def _tavily(client, key, query, count):
    r = await client.post("https://api.tavily.com/search",
                          json={"query": query, "max_results": count},
                          headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return [{"title": i.get("title", ""), "url": i.get("url", ""),
             "snippet": i.get("content", "")} for i in r.json().get("results", [])]


async def _serper(client, key, query, count):
    r = await client.post("https://google.serper.dev/search", json={"q": query, "num": count},
                          headers={"X-API-KEY": key})
    r.raise_for_status()
    return [{"title": i.get("title", ""), "url": i.get("link", ""),
             "snippet": i.get("snippet", "")} for i in r.json().get("organic", [])]


_CALLS = {"BRAVE_API_KEY": _brave, "TAVILY_API_KEY": _tavily, "SERPER_API_KEY": _serper}


def make_web_search(network_allowed, policy=None):
    """web_search for agents; network_allowed() says whether this run may go online."""

    async def web_search(query: str, count: int = 5) -> list[dict]:
        """Search the web: [{"title", "url", "snippet"}]. Read a result with fetch_url(url)."""
        import httpx

        if not network_allowed():
            raise ValueError("network access is off for this run")
        name = configured()
        if name is None:
            raise ValueError("no search key is set; set one of " + ", ".join(SERVICES))
        count = max(1, min(int(count), MAX_RESULTS))
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                rows = await _CALLS[name](client, os.environ[name], str(query), count)
            except httpx.HTTPStatusError as exc:
                # The key is in a header, never in the message.
                raise ValueError(f"search failed with HTTP {exc.response.status_code}") from None
        if policy is not None and policy.search_block:
            from rlmagent.policy import blocked_url

            rows = [row for row in rows if not blocked_url(str(row.get("url", "")), policy)]
        return [{k: str(v)[:500] for k, v in row.items()} for row in rows[:count]]

    return web_search


__all__ = ["SERVICES", "configured", "make_web_search"]
