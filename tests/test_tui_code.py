from rich.console import Console, Group
from rich.syntax import Syntax
from rich.text import Text

from rlmagent.harness.contracts.stream import ToolRunStartEvent
from rlmagent.tui.composer import ModelBadge
from rlmagent.tui.messages import render_message


def _shown(renderable) -> str:
    console = Console(width=60, record=True, file=None, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


def test_finished_python_blocks_are_coloured_and_text_stays_literal():
    rendered = render_message("look [bold]here[/bold]\n```python\nx = 1\n```\ndone")
    assert isinstance(rendered, Group)
    assert [type(part) for part in rendered.renderables] == [Text, Syntax, Text]
    shown = _shown(rendered)
    assert "[bold]here[/bold]" in shown and "x = 1" in shown and "```" not in shown


def test_a_block_still_streaming_stays_plain():
    rendered = render_message("```python\nx = ")
    assert isinstance(rendered, Text) and rendered.plain == "```python\nx = "


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
