"""search_files returns only matches, so its length is always a count."""

import pytest

from rlmagent.agents import workspace
from rlmagent.agents.approval import Approver
from rlmagent.agents.workspace import Workspace


@pytest.mark.asyncio
async def test_a_large_search_returns_every_match(tmp_path):
    (tmp_path / "server.log").write_text("".join(f"ERROR status=503 {i}\n" for i in range(3345)))
    hits = await Workspace(tmp_path, Approver("deny")).search_files("status=503", path="server.log")
    assert len(hits) == 3345 and all("note" not in h for h in hits) and not hits.truncated


@pytest.mark.asyncio
async def test_an_explicit_limit_stops_early_and_says_so(tmp_path):
    (tmp_path / "a.txt").write_text("hit\n" * 50)
    hits = await Workspace(tmp_path, Approver("deny")).search_files("hit", limit=10)
    assert len(hits) == 10 and hits.truncated and all("note" not in h for h in hits)


@pytest.mark.asyncio
async def test_past_the_ceiling_is_an_error_not_a_short_list(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "MAX_MATCHES", 20)
    (tmp_path / "a.txt").write_text("hit\n" * 50)
    with pytest.raises(ValueError, match="more than 20"):
        await Workspace(tmp_path, Approver("deny")).search_files("hit")


@pytest.mark.asyncio
async def test_small_search_is_exact(tmp_path):
    (tmp_path / "a.txt").write_text("hit\nmiss\nhit\n")
    hits = await Workspace(tmp_path, Approver("deny")).search_files("hit")
    assert len(hits) == 2
