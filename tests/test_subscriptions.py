import asyncio
from pathlib import Path

import pytest

from rlmagent.jobs import Job, Jobs


async def test_path_subscription_keeps_ids_events_and_reads_after_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr("rlmagent.subscriptions.POLL_SECONDS", 0.005)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    folder = tmp_path / "jobs"
    jobs = Jobs(str(workspace), folder)
    store = jobs.subscriptions
    watch = await store.watch_start(path="notes.txt")
    identity = watch["id"]
    waiting = store.prepare_wait(identity, 1)
    sleeper = asyncio.create_task(store.wait_for_change(waiting, lambda: False))
    await asyncio.sleep(0)
    (workspace / "notes.txt").write_text("first", encoding="utf-8")
    assert await sleeper == {"event": "watch", "watch_id": identity}
    row, = await store.watch_events(identity)
    assert "change" not in row and row["status"] == "unread"
    payload = await store.watch_read(identity, row["id"])
    assert payload["change"] == {"paths": ["notes.txt"], "truncated": False}
    await jobs.close()

    reopened = Jobs(str(workspace), folder)
    try:
        store = reopened.subscriptions
        assert not await store.watch_events(identity)
        assert await store.watch_read(identity, row["id"]) == payload
        assert await store.watch_cancel(identity) == {"id": identity, "active": False}
        assert not await store.watch_list()
        with pytest.raises(ValueError, match="outside the workspace"):
            await store.watch_start(path="../private")
    finally:
        await reopened.close()


async def test_job_subscription_records_output_and_completion(tmp_path):
    jobs = Jobs(str(tmp_path), tmp_path / "jobs")
    log = Path(jobs.folder) / "1234abcd.log"
    log.parent.mkdir()
    log.touch()
    job = Job("1234abcd", "fixture", log, persistent=True)
    jobs.jobs[job.id] = job
    store = jobs.subscriptions
    try:
        watch = await store.watch_start(job=job.id)
        log.write_bytes(b"hello")
        first, = await store.watch_events(watch["id"])
        assert (await store.watch_read(watch["id"], first["id"]))["change"]["end"] == 5
        job.ended = job.started + 1
        job.exit = 0
        finished, = await store.watch_events(watch["id"])
        assert (await store.watch_read(watch["id"], finished["id"]))["change"]["done"]
        assert not (await store.watch_list())[0]["active"]
    finally:
        await jobs.close()


async def test_wait_only_announces_new_events_and_wakes_for_steering(tmp_path):
    store = Jobs(str(tmp_path), tmp_path / "jobs").subscriptions
    try:
        watch = await store.watch_start(path="tracked")
        (tmp_path / "tracked").write_text("first", encoding="utf-8")
        assert await store.watch_events(watch["id"])
        request = store.prepare_wait(watch["id"], 0.01)
        assert await store.wait_for_change(request, lambda: False) == {"event": "timeout"}
        assert await store.wait_for_change(request, lambda: True) == {"event": "steering"}
        await store.watch_cancel(watch["id"])
        assert (await store.wait_for_change(request, lambda: False))["event"] == "cancelled"
    finally:
        await store.close()
