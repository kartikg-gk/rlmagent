from rich.console import Console

from rlmagent.harness.contracts.stream import ToolRunStartEvent
from rlmagent.tui.composer import ModelBadge
from rlmagent.tui.messages import render_message


def _shown(renderable) -> str:
    console = Console(width=60, record=True, file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_messages_render_as_markdown_and_never_as_console_markup():
    shown = _shown(render_message("# Head\n\nlook [bold]here[/bold]\n\n```python\nx = 1\n```"))
    assert "Head" in shown and "[bold]here[/bold]" in shown and "x = 1" in shown
    assert "```" not in shown and "# Head" not in shown


def test_streamed_progress_stays_above_the_answer():
    from rlmagent.tui.messages import AssistantBlock

    block = AssistantBlock("```python\nx = 1\n```")
    block.finalize("the answer")
    assert "x = 1" in block.text and block.text.endswith("the answer")
    same = AssistantBlock("hello")
    same.finalize("hello")
    assert same.text == "hello"


def test_python_tool_calls_carry_their_code_to_the_screen():
    from rlmagent.tui.adapter import ToolStarted, summarize_call

    event = ToolRunStartEvent(tool_call_id="c", tool_name="python", args={"code": "x = 1"})
    started = ToolStarted(event.tool_call_id, event.tool_name,
                          summarize_call(event.tool_name, event.args), event.args["code"])
    assert started.code == "x = 1"


def test_the_input_row_says_when_auto_accept_is_on():
    badge = ModelBadge()
    badge.show("m", "high")
    assert badge.line == "m · high"
    badge.show("m", "high", True)
    assert badge.line == "auto-accept · m · high"


async def test_code_renders_inside_the_real_app():
    from textual.app import App

    from rlmagent.tui.messages import AssistantBlock, ToolCode

    class Host(App):
        def compose(self):
            yield AssistantBlock("before\n```python\nx = 1\n```")
            yield ToolCode("y = 2")

    async with Host().run_test() as pilot:
        await pilot.pause()
        block = pilot.app.query_one(AssistantBlock)
        block.append("\nafter")
        await pilot.pause()
        assert block.text.endswith("after") and block.region.height > 0
        assert pilot.app.query_one(ToolCode).region.height > 0


def test_approval_answers_to_y_and_n():
    from rlmagent.agents.approval import Request
    from rlmagent.tui.approval_screen import ApprovalScreen

    keys = {binding[0]: binding[1] for binding in ApprovalScreen.BINDINGS}
    assert keys["y"] == "once" and keys["n"] == "deny"
    assert ApprovalScreen(Request("run", "x")).request.target == "x"


async def test_the_question_screen_collects_several_picks():
    from textual.app import App
    from textual.widgets import Input, SelectionList

    from rlmagent.tui.question_screen import QuestionScreen

    answers = []

    class Host(App):
        def on_mount(self):
            self.push_screen(QuestionScreen("which?", ["a", "b"], True), answers.append)

    async with Host().run_test() as pilot:
        await pilot.pause()
        pilot.app.screen.query_one(SelectionList).select("b")
        pilot.app.screen.query_one(Input).focus()
        await pilot.press("enter")
        await pilot.pause()
    assert answers == [["b"]]


async def _driven_app(tmp_path, turns=()):
    from _scripted import text_turn  # noqa: F401
    from rlmness import Allowance

    from rlmagent.agents.approval import Approver
    from rlmagent.agents.tree import AgentTree
    from rlmagent.conversation import CodingSession
    from rlmagent.model.scripted import ReplayProvider
    from rlmagent.tui.adapter import SessionBridge
    from rlmagent.tui.app import RlmAgentApp

    tree = AgentTree(provider=ReplayProvider(list(turns)), provider_name="replay", model="m",
                     cwd=str(tmp_path), allowance=Allowance(max_cost=10.0), sessions_dir=None,
                     system_for=lambda n, t: "s", first_message_for=lambda n: n.task,
                     approver=Approver("ask"))
    await tree.start()
    session = await CodingSession.create(
        provider=tree.provider_for("root"), provider_name="replay", model="m", system="s",
        tools=[], cwd=str(tmp_path))
    session.agent_tree = tree
    return RlmAgentApp(SessionBridge(session)), tree


async def test_the_real_app_handles_approval_auto_accept_questions_and_esc(tmp_path):
    from rlmagent.agents.approval import Request
    from rlmagent.tui.approval_screen import ApprovalScreen
    from rlmagent.tui.composer import ModelBadge
    from rlmagent.tui.question_screen import QuestionScreen

    app, tree = await _driven_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        pending = app.run_worker(tree.approver.approve(Request("run", "make x")))
        await pilot.pause()
        assert isinstance(app.screen, ApprovalScreen)
        await pilot.press("y")
        await pending.wait()

        await pilot.press("shift+tab")
        await pilot.pause()
        assert tree.approver.policy == "auto"
        assert app.query_one(ModelBadge).line.startswith("auto-accept")
        await pilot.press("shift+tab")
        await pilot.pause()
        assert tree.approver.policy == "ask"

        asking = app.run_worker(tree.approver.question("which?", ["a", "b"]))
        await pilot.pause()
        assert isinstance(app.screen, QuestionScreen)
        await pilot.press("enter")
        assert await asking.wait() == "a"
        await pilot.pause()

        await pilot.press("x")
        await pilot.press("escape")
        await pilot.pause()
        assert app.is_running
        await pilot.press("backspace")
        await pilot.press("escape")
        await pilot.pause()
    assert app.return_code in (0, None)
    await tree.close()


async def test_the_model_page_offers_recent_models_and_filters_fetched_ones(tmp_path, monkeypatch):
    from textual.app import App
    from textual.widgets import Input, ListView

    from rlmagent.config.store import recent_models, remember_model
    from rlmagent.tui.model_screen import ModelScreen

    monkeypatch.setenv("RLM_AGENT_CONFIG_DIR", str(tmp_path))
    remember_model("anthropic", "old-model")
    remember_model("anthropic", "new-model")
    remember_model("anthropic", "old-model")
    assert [r["model"] for r in recent_models()] == ["old-model", "new-model"]

    class Host(App):
        def on_mount(self):
            self.push_screen(ModelScreen())

    async with Host().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        listing = screen.query_one("#model-list", ListView)
        assert listing.children[0].value == "recent:anthropic|old-model"
        screen._fetched = ["alpha-1", "beta-2"]
        screen._list_fetched("bet")
        await pilot.pause()
        assert [row.value for row in listing.children] == ["beta-2", "__custom__"]
        screen.query_one("#model-filter", Input)


def test_a_first_run_goes_to_setup(tmp_path, monkeypatch):
    import argparse

    from rlmagent.config.config import resolve_provider  # noqa: F401
    from rlmagent.model.settings import ConfigError
    from rlmagent.tui import app as tui

    def none(_name):
        raise ConfigError("No provider configured.")

    monkeypatch.setattr("rlmagent.config.config.resolve_provider", none)
    assert tui.needs_setup(argparse.Namespace(provider=None))
    assert not tui.needs_setup(argparse.Namespace(provider="cli"))
