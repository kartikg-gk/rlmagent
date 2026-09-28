import asyncio
import time

from _scripted import code_turn, text_turn
from rlmness import Allowance
from test_agent_tree import _SlowProvider

from rlmagent_app.agents.tree import AgentTree
from rlmagent_model.scripted import ReplayProvider


def _tree(tmp_path, provider, **kw):
    return AgentTree(
        provider=provider, provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(max_depth=kw.pop("max_depth", 2), max_cost=10.0, max_calls=kw.pop("max_calls", 60),
                            max_live=kw.pop("max_live", 8)),
        sessions_dir=None,
        system_for=lambda node, tools: "s", first_message_for=lambda node: node.task,
        cell_timeout=30, **kw,
    )


async def test_spawn_returns_at_once_and_result_waits(tmp_path):
    tree = _tree(tmp_path, _SlowProvider([code_turn("FINAL(7)"), text_turn("ok")], delay=2))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("pass")
        started = time.monotonic()
        await k.run("h = await spawn('slow')")
        assert time.monotonic() - started < 1.5
        assert (await k.run("print(await h.status())")).output.strip() in ("starting", "running")
        out = await k.run("print(await h.result(), await h.status())")
        assert out.output.strip() == "7 done", out.error
    finally:
        await tree.close()


async def test_result_timeout_leaves_it_running(tmp_path):
    tree = _tree(tmp_path, _SlowProvider([code_turn("FINAL(1)"), text_turn("ok")], delay=3))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('slow')")
        out = await k.run("try:\n    await h.result(timeout=0.5)\nexcept TimeoutError:\n    print('timeout', await h.status())")
        assert out.output.strip() in ("timeout running", "timeout starting")
        assert (await k.run("print(await h.result())")).output.strip() == "1"
    finally:
        await tree.close()


async def test_failed_child_raises_from_result(tmp_path):
    tree = _tree(tmp_path, ReplayProvider([]), max_depth=1, max_calls=0)
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('x')")
        out = await k.run("await h.result()")
        assert out.error and "calls" in out.error
        assert (await k.run("print(await h.status())")).output.strip() == "failed"
    finally:
        await tree.close()


async def test_cancel_stops_a_child_and_its_own_children(tmp_path):
    tree = _tree(tmp_path, _SlowProvider([code_turn("g = await spawn('grandchild')\nFINAL(1)")] + [text_turn("x")] * 10, delay=1))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('child')")
        await asyncio.sleep(2.5)
        await k.run("await h.cancel()")
        assert (await k.run("print(await h.status())")).output.strip() == "cancelled"
        assert all(r.status in ("cancelled", "done", "failed") for r in tree.records.values())
        assert [n for n, kern in tree.kernels.items() if n != "root" and kern.started] == []
    finally:
        await tree.close()


async def test_keep_goes_idle_and_wakes_on_send(tmp_path):
    tree = _tree(tmp_path, ReplayProvider([
        code_turn("FINAL(1)"), text_turn("ok"),
        code_turn("FINAL(2)"), text_turn("ok"),
    ]))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('count', keep=True)")
        assert (await k.run("print(await h.result(), await h.status())")).output.strip() == "1 idle"
        await k.run("await h.send('next')")
        out = await k.run("print(await h.result(), await h.status())")
        assert out.output.strip() == "2 idle", out.error
    finally:
        await tree.close()


async def test_send_to_a_finished_child_is_refused(tmp_path):
    tree = _tree(tmp_path, ReplayProvider([code_turn("FINAL(1)"), text_turn("ok")]))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('once')\nawait h.result()")
        assert "finished" in (await k.run("await h.send('more')")).error
    finally:
        await tree.close()


async def test_tell_parent_and_messages(tmp_path):
    tree = _tree(tmp_path, ReplayProvider([
        code_turn("await tell_parent('half way')\nawait tell_parent('done soon')\nFINAL(0)"),
        text_turn("ok"),
    ]))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('work')\nawait h.result()")
        assert (await k.run("print(await h.messages())")).output.strip() == "['half way', 'done soon']"
        assert (await k.run("print(await h.messages())")).output.strip() == "[]"
    finally:
        await tree.close()


async def test_children_recovers_lost_handles(tmp_path):
    tree = _tree(tmp_path, ReplayProvider([code_turn("FINAL(5)"), text_turn("ok")]))
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('a')\ndel h")
        out = await k.run("hs = await children()\nprint(len(hs), hs[0].id, await hs[0].result())")
        assert out.output.strip() == "1 root.1 5", out.error
    finally:
        await tree.close()


async def test_max_agents_is_enforced_at_spawn(tmp_path):
    tree = _tree(tmp_path, _SlowProvider([text_turn("x")] * 10, delay=5), max_agents=1)
    await tree.start()
    try:
        k = tree.root_kernel()
        await k.run("h = await spawn('a')")
        assert "sub-agents" in (await k.run("await spawn('b')")).error
    finally:
        await tree.close()


async def test_close_cancels_background_children(tmp_path):
    tree = _tree(tmp_path, _SlowProvider([text_turn("x")] * 10, delay=5))
    await tree.start()
    k = tree.root_kernel()
    await k.run("h = await spawn('a')")
    await asyncio.sleep(1.5)
    await tree.close()
    assert all(not kern.started for kern in tree.kernels.values())


async def test_a_child_finishing_cancels_its_running_children(tmp_path):
    tree = _tree(tmp_path, _SlowProvider(
        [code_turn("g = await spawn('slow grandchild')\nFINAL(1)"), text_turn("ok")] + [text_turn("x")] * 10,
        delay=0.5,
    ))
    await tree.start()
    try:
        k = tree.root_kernel()
        assert (await k.run("print(await (await spawn('child')).result())")).output.strip() == "1"
        await asyncio.sleep(0.5)
        assert tree.records["root.1.2"].status == "cancelled"
    finally:
        await tree.close()



async def test_close_does_not_cut_short_a_finished_sub_agents_cleanup(tmp_path, monkeypatch):
    from rlmagent_app.kernel import KernelSession

    completed = []
    original = KernelSession.shutdown

    async def slow_shutdown(self):
        await asyncio.sleep(0.3)
        await original(self)
        completed.append(self)

    monkeypatch.setattr(KernelSession, "shutdown", slow_shutdown)
    tree = _tree(tmp_path, ReplayProvider([code_turn("FINAL(1)"), text_turn("ok")]))
    await tree.start()
    k = tree.root_kernel()
    await k.run("await rlm('x')")
    child_kernel = tree.kernels["root.1"]
    await tree.close()
    assert completed.count(child_kernel) == 2  # its own cleanup, then close()
