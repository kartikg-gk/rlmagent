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
