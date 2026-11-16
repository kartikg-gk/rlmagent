"""A capped search says how many matches there really were, so it is not taken as a count."""

import pytest

from rlmagent.agents.approval import Approver
from rlmagent.agents.workspace import MAX_MATCHES, Workspace


@pytest.mark.asyncio
async def test_capped_search_of_one_file_gives_the_true_total(tmp_path):
    (tmp_path / "server.log").write_text("".join(f"ERROR status=503 {i}\n" for i in range(1234)))
    hits = await Workspace(tmp_path, Approver("deny")).search_files("status=503", path="server.log")
    assert len(hits) == MAX_MATCHES + 1
    assert f"of 1234 matching lines" in hits[-1]["note"]


@pytest.mark.asyncio
async def test_capped_search_of_a_folder_says_it_is_not_a_count(tmp_path):
    for n in range(3):
        (tmp_path / f"f{n}.txt").write_text("hit\n" * 300)
    hits = await Workspace(tmp_path, Approver("deny")).search_files("hit")
    assert "not a count" in hits[-1]["note"]


@pytest.mark.asyncio
async def test_uncapped_search_has_no_note(tmp_path):
    (tmp_path / "a.txt").write_text("hit\nmiss\nhit\n")
    hits = await Workspace(tmp_path, Approver("deny")).search_files("hit")
    assert len(hits) == 2 and all("note" not in h for h in hits)
