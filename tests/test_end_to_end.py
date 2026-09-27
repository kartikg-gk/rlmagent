import pytest
from rlmness import Allowance

from rlmagent_app.agents.prompts import child_first_message, child_system, root_system
from rlmagent_app.agents.tree import AgentTree
from rlmagent_app.conversation import CodingSession
from rlmagent_app.tools import build_tool_registry
from rlmagent_model.scripted import ReplayProvider
from _scripted import code_turn, text_turn



async def test_root_delegates_and_answers(tmp_path):
    (tmp_path / "data.txt").write_text("alpha\nbeta\ngamma\n")
    streams = [
        code_turn("lines = open('data.txt').read().split()\nn = await rlm('count items', lines)\nprint(n)"),
        code_turn("FINAL(len(CONTEXT))"),   # child
        text_turn("child done"),            # child
        text_turn("There are 3 lines."),    # root
    ]
    tree = AgentTree(
        provider=ReplayProvider(streams), provider_name="replay", model="m", cwd=str(tmp_path),
        allowance=Allowance(max_cost=10.0), sessions_dir=tmp_path / "sessions",
        system_for=lambda node, tools: child_system("base", node, True),
        first_message_for=child_first_message,
    )
    await tree.start()
    kernel = tree.root_kernel()
    session = await CodingSession.create(
        provider=tree.provider, provider_name="replay", model="m",
        system=root_system("base", True), tools=build_tool_registry(kernel),
        sessions_dir=tmp_path / "sessions", cwd=str(tmp_path),
    )
    session.agent_tree = tree
    try:
        async for _ in session.submit("How many lines in data.txt?"):
            pass
        tool_texts = [m.text for m in session.transcript if getattr(m, "role", "") == "toolResult"]
        assert any(t.strip() == "3" for t in tool_texts)
        assert session.transcript[-1].text == "There are 3 lines."
        saved = [f for f in (tmp_path / "sessions").glob("*.jsonl") if f.name != "index.jsonl"]
        assert len(saved) == 2   # root + child
    finally:
        await session.shutdown()
