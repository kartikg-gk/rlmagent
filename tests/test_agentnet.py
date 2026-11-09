import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from rlmagent import worker
from rlmagent.agentnet import AgentNet, inbox_note


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RLM_AGENT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


async def test_sessions_find_and_message_each_other(tmp_path):
    a, b = AgentNet("s-aaaa", str(tmp_path)), AgentNet("s-bbbb", str(tmp_path))
    a.register()
    b.register()
    names = {row["name"]: row for row in await a.list_agents()}
    assert names["s-aaaa"]["me"] and names["s-bbbb"]["kind"] == "session"
    assert await a.send_message("s-bbbb", "check the logs") == "sent to session s-bbbb"
    got = await b.read_messages()
    assert [(m["from"], m["text"]) for m in got] == [("s-aaaa", "check the logs")]
    assert await b.read_messages() == []
    with pytest.raises(ValueError, match="that is you"):
        await a.send_message("s-aaaa", "x")
    with pytest.raises(ValueError, match="no agent named 'ghost'"):
        await a.send_message("ghost", "x")
    b.leave()
    with pytest.raises(ValueError, match="no agent named"):
        await a.send_message("s-bbbb", "x")
    assert inbox_note(got) == "Messages from other agents arrived:\n[message from s-aaaa] check the logs"


async def test_a_message_to_a_worker_becomes_its_task(tmp_path):
    worker.folder("night").mkdir(parents=True)
    net = AgentNet("s-aaaa", str(tmp_path))
    assert await net.send_message("night", "rerun") == "sent to worker night"
    [message] = worker._take_inbox("night")
    assert message["text"] == "[message from s-aaaa] rerun" and message["source"] == "s-aaaa"
    rows = await net.list_agents()
    assert rows[0] == {"name": "night", "kind": "worker", "state": "not running",
                       "cwd": None, "me": False}


async def test_scheduling_and_heartbeats(tmp_path):
    net = AgentNet("s-aaaa", str(tmp_path))
    with pytest.raises(ValueError, match="not a worker"):
        await net.schedule("later", in_seconds=60)
    net.worker = "night"
    with pytest.raises(ValueError, match="exactly one of"):
        await net.schedule("later")
    with pytest.raises(ValueError, match="at least 60"):
        await net.schedule("later", in_seconds=5, every=10)
    note = await net.schedule("rerun checks", in_seconds=120, every=3600)
    assert note.startswith("job ") and "on night" in note and "every 3600s" in note
    [job] = worker.jobs_of("night")
    assert job["text"] == "rerun checks" and job["every"] == 3600
    assert job["due"] == pytest.approx(time.time() + 120, abs=5)
    seen = []
    net.on_progress = seen.append
    assert await net.heartbeat("halfway through") == "noted"
    assert seen == ["halfway through"]
    log = (worker.folder("night") / "log.jsonl").read_text()
    assert '"kind": "progress"' in log and "halfway through" in log


async def test_bridge_shows_and_injects_waiting_messages(tmp_path):
    from rlmagent.tui.adapter import AgentNote, SessionBridge

    injected = []

    class Session:
        agentnet = AgentNet("s-bbbb", str(tmp_path))

        def inject(self, text):
            injected.append(text)

        async def submit(self, text):
            return
            yield

    Session.agentnet.register()
    await AgentNet("s-aaaa", str(tmp_path)).send_message("s-bbbb", "hello")
    updates = [u async for u in SessionBridge(Session()).submit("go")]
    assert isinstance(updates[0], AgentNote) and "hello" in updates[0].text
    assert injected == [updates[0].text]


@pytest.fixture
def agents(tmp_path):
    sender = AgentNet("s-sender", str(tmp_path))
    recipient = AgentNet("s-target", str(tmp_path))
    recipient.register()
    return sender, recipient


async def test_session_receipt_survives_read_and_reopen(agents):
    sender, recipient = agents
    sent = await sender.send_message(recipient.me, "verify package", receipt=True)
    identity = sent["id"]
    assert sent == await sender.message_receipt(recipient.me, identity, "session")
    assert sent["status"] == "queued"

    received, = await recipient.read_messages()
    assert (received["id"], received["text"], received["from"]) == (
        identity, "verify package", sender.me)
    assert "receipt_owner" not in received
    recipient.leave()
    reopened = AgentNet(sender.me, sender.cwd)
    assert (await reopened.message_receipt(recipient.me, identity, "session"))["status"] == "read"
    assert not await AgentNet(recipient.me, recipient.cwd).read_messages()
    other = AgentNet("s-other", sender.cwd, worker=sender.me)
    assert (await other.message_receipt(recipient.me, identity, "session"))["status"] == "unknown"


async def test_concurrent_read_claims_a_session_message_once(agents):
    sender, recipient = agents
    assert await sender.send_message(recipient.me, "once") == "sent to session s-target"
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        received = await asyncio.gather(*(loop.run_in_executor(pool, recipient.take) for _ in range(2)))
    assert [row["text"] for batch in received for row in batch] == ["once"]


async def test_worker_receipt_tracks_queue_and_final_outcome(agents):
    sender, _ = agents
    worker.folder("review").mkdir(parents=True)
    sent = await sender.send_message("review", "test again", receipt=True)
    identity = sent["id"]
    assert sent == await sender.message_receipt("review", identity, "worker")
    assert worker._take_inbox("review")[0]["text"] == "[message from s-sender] test again"
    assert (await sender.message_receipt("review", identity, "worker"))["status"] == "in-flight"
    worker._finish_message("review", identity, "interrupted")
    assert (await sender.message_receipt("review", identity, "worker"))["status"] == "interrupted"
    assert "receipt_owner" not in worker.receipt_of("review", identity)
    stranger = AgentNet("s-other", sender.cwd)
    assert (await stranger.message_receipt("review", identity, "worker"))["status"] == "unknown"
    with pytest.raises(ValueError, match="invalid message ID"):
        await sender.message_receipt("review", "../meta", "worker")
