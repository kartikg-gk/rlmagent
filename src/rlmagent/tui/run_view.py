"""The run panes around the conversation, laid out as the engine's own viewer.

A runs tree and the REPL namespace down the left; an info strip, then the
code and output of the selected step on the right. They read the same step
book the step cards do and only draw: nothing here starts or changes a run.
"""

from __future__ import annotations

from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Static, Tree

from rlmagent.tui.steps import DONE, FAILED, GENERATING, RUNNING, StepBook, short_id

MARKS = {RUNNING: "·", GENERATING: "·", DONE: "✓", FAILED: "✗"}
BOXES = {RUNNING: "[●]", GENERATING: "[●]", DONE: "[✓]", FAILED: "[✗]"}
COLOURS = {RUNNING: "#c9a4ff", GENERATING: "#c9a4ff", DONE: "#7a7a7a", FAILED: "#a05050"}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _clip(text: object, width: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


class RunsColumn(Vertical):
    """Left column: the tree of agents and steps, and the namespace below it."""

    def __init__(self) -> None:
        super().__init__(id="runs-column")

    def compose(self) -> ComposeResult:
        yield Tree("run", id="runs-tree")
        with VerticalScroll(id="namespace"):
            yield DataTable(id="variables", cursor_type="row")

    def on_mount(self) -> None:
        self.query_one("#runs-tree").border_title = "Runs"
        self.query_one("#namespace").border_title = "REPL Namespace"
        self.query_one("#variables", DataTable).add_columns("name", "type", "size", "preview")
        self.query_one("#runs-tree", Tree).root.expand()


class StepPanes(Horizontal):
    """Code and output of one step, side by side."""

    def __init__(self) -> None:
        super().__init__(id="step-panes")

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="code"):
            yield Static("[dim]—[/dim]", id="code-body")
        with VerticalScroll(id="output"):
            yield Static("[dim]—[/dim]", id="output-body")

    def on_mount(self) -> None:
        self.query_one("#code").border_title = "Code"
        self.query_one("#output").border_title = "Output"


class RunView:
    """Draws a step book into the runs column, the info strip and the step panes.

    Kept apart from the widgets so the app owns layout and this owns meaning:
    which step is selected, whether the view follows the newest one, and what
    each pane says about it.
    """

    def __init__(self, app) -> None:
        self.app = app
        self.selected: tuple[str, int | None] | None = None
        self.following = True
        self._agent_nodes: dict[str, object] = {}
        self._step_nodes: dict[tuple[str, int], object] = {}
        self._drawn = -1
        self.frame = 0

    def reset(self) -> None:
        self.app.query_one("#runs-tree", Tree).clear()
        self._agent_nodes.clear()
        self._step_nodes.clear()
        self.selected = None
        self.following = True
        self._drawn = -1

    def draw(self, book: StepBook, *, force: bool = False) -> None:
        if not force and book.version == self._drawn:
            return
        self._drawn = book.version
        if set(self._agent_nodes) - set(book.agents):  # the book was cleared for a new run
            self.reset()
        self._sync_tree(book)
        self._draw_info(book)
        self._draw_step(book)
        self._draw_namespace(book)

    # -- tree -----------------------------------------------------------------

    def _sync_tree(self, book: StepBook) -> None:
        """Add what is new and relabel what changed, keeping the cursor and folds."""
        widget = self.app.query_one("#runs-tree", Tree)
        for agent in book.agents.values():
            node = self._agent_nodes.get(agent.run_id)
            if node is None:
                if agent.parent is None or agent.parent not in self._agent_nodes:
                    node = widget.root if agent.parent is None and not self._agent_nodes else \
                        widget.root.add("", expand=True)
                else:
                    node = self._agent_nodes[agent.parent].add("", expand=True)
                self._agent_nodes[agent.run_id] = node
            node.set_label(self._agent_label(book, agent))
            node.data = (agent.run_id, None)
            for number in agent.steps:
                key = (agent.run_id, number)
                step_node = self._step_nodes.get(key)
                if step_node is None:
                    step_node = node.add_leaf("")
                    step_node.data = key
                    self._step_nodes[key] = step_node
                step_node.set_label(self._step_label(book.steps[key]))
        if self.following and book.order:
            self.selected = book.order[-1]

    def _agent_label(self, book: StepBook, agent) -> Text:
        mark = MARKS.get(agent.status, " ")
        name = book.child_names.get(agent.child_id or "", "") or (
            "root" if agent.parent is None else short_id(agent.run_id))
        return Text.assemble((f"{mark} ", COLOURS.get(agent.status, "#6a6a6a")),
                             (name, "#c9a4ff"), (f" (d{agent.depth})", "#6a6a6a"))

    def _step_label(self, step) -> Text:
        label = Text.assemble((BOXES.get(step.state, "[ ]"), COLOURS.get(step.state, "#6a6a6a")),
                              (f" Step {step.number}", "#c8c8c8"))
        if step.seconds is not None:
            label.append(f"  {step.seconds:.1f}s", "#6a6a6a")
        if step.state == FAILED:
            label.append("  error", "#a05050")
        return label

    # -- detail ---------------------------------------------------------------

    def _current(self, book: StepBook):
        if self.selected is None:
            return None, None
        run_id, number = self.selected
        agent = book.agents.get(run_id)
        step = book.steps.get((run_id, number)) if number is not None else None
        return agent, step

    def _draw_info(self, book: StepBook) -> None:
        panel = self.app.query_one("#info", Static)
        agent, step = self._current(book)
        totals = book.totals()
        if agent is None:
            panel.update("[dim]no step selected[/dim]")
            return
        name = "root" if agent.parent is None else short_id(agent.run_id)
        # Brackets escaped, or the state would be read as a markup tag and vanish.
        where = (f"step {step.number} \\[{step.state}]" if step is not None
                 else f"agent \\[{agent.status}]")
        follow = "follow" if self.following else "pinned"
        tokens = step.tokens if step is not None else totals["tokens"]
        cost = step.cost if step is not None else totals["cost"]
        seconds = f"  ·  {step.seconds:.1f}s" if step is not None and step.seconds is not None else ""
        panel.update(
            f"[#c9a4ff]{name}[/] (d{agent.depth})  ·  {where}  ·  {agent.model or '—'}  [dim]({follow})[/dim]\n"
            f"tokens {tokens:,}  ·  cost {'unknown' if cost is None else f'${cost:.4f}'}{seconds}\n"
            f"[dim]run: {totals['agents']} agents · {totals['steps']} steps · "
            f"{totals['tokens']:,} tokens · {totals['running']} running[/dim]"
        )

    def _draw_step(self, book: StepBook) -> None:
        _, step = self._current(book)
        code = self.app.query_one("#code-body", Static)
        output = self.app.query_one("#output-body", Static)
        if step is None:
            code.update("[dim]—[/dim]")
            output.update("[dim]—[/dim]")
            return
        if step.code:
            code.update(Syntax(step.code, "python", theme="monokai",
                               background_color="default", word_wrap=True))
        else:
            code.update("[dim]waiting for the model…[/dim]")
        # Output is shown as printed, not highlighted as if it were Python.
        output.update(Text(step.output) if step.output else "[dim]—[/dim]")

    def _draw_namespace(self, book: StepBook) -> None:
        table = self.app.query_one("#variables", DataTable)
        table.clear()
        _, step = self._current(book)
        rows = step.variables if step is not None else []
        if not rows:
            # A step that reported nothing shows its own agent's latest snapshot,
            # and failing that the newest one from any agent.
            mine = [k for k in reversed(book.order)
                    if self.selected is not None and k[0] == self.selected[0]]
            for key in [*mine, *reversed(book.order)]:
                if book.steps[key].variables:
                    rows = book.steps[key].variables
                    break
        for row in rows:
            size = row.get("size")
            table.add_row(str(row.get("name", "")), str(row.get("type", "")),
                          "—" if size is None else str(size), _clip(row.get("preview", ""), 60))

    # -- input ------------------------------------------------------------------

    def select(self, data) -> None:
        """A tree pick pins the view on that step until follow is turned back on."""
        if data is None:
            return
        self.selected = data
        self.following = False
        self._drawn = -1

    def toggle_follow(self) -> None:
        self.following = not self.following
        self._drawn = -1


def query_line(title: str, book: StepBook, *, busy: bool, frame: int, question: str,
               following: bool) -> str:
    """The QUERY strip: name, state, counts, and what was last asked."""
    totals = book.totals()
    if not question and not busy:
        return f"[b]{title}[/b]  [dim]waiting for a question[/dim]"
    if busy:
        mark, colour, word = SPINNER[frame % len(SPINNER)], "#c9a4ff", "running"
    elif any(a.status == FAILED for a in book.agents.values() if a.parent is None):
        mark, colour, word = "✗", "#a05050", "failed"
    else:
        mark, colour, word = "✓", "#7a7a7a", "done"
    follow = "follow" if following else "pinned"
    return (f"[b]{title}[/b]  [{colour}]{mark} {word}[/]  agents {totals['agents']}  "
            f"steps {totals['steps']}  [dim]({follow})[/dim]\n[dim]{_clip(question, 200)}[/dim]")


class DashboardScreen(Screen):
    """The run, inspected: query strip, runs tree, namespace, info, code and output.

    Opened over the conversation with F5 and closed with F5 or Esc. It reads
    the live step book on a clock, so it keeps up with a run still going.
    """

    BINDINGS = [
        Binding("f5", "close", "Back to chat"),
        Binding("escape", "close", "Back to chat"),
        Binding("ctrl+l", "follow", "Follow latest"),
    ]

    def __init__(self, book_source, question_source, busy_source) -> None:
        super().__init__(classes="dashboard-screen")
        self._book = book_source
        self._question = question_source
        self._busy = busy_source
        self.view = RunView(self)

    def compose(self) -> ComposeResult:
        yield Static(id="query")
        with Horizontal(id="dash-body"):
            yield RunsColumn()
            with Vertical(id="dash-right"):
                yield Static(id="info")
                yield StepPanes()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#query").border_title = "QUERY"
        self.query_one("#info").border_title = "Info"
        self._tick()
        self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        self.view.frame += 1
        book = self._book()
        try:
            self.query_one("#query", Static).update(query_line(
                "rlm-agent", book, busy=self._busy(), frame=self.view.frame,
                question=self._question(), following=self.view.following))
            self.view.draw(book)
        except NoMatches:
            return

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        event.stop()
        self.view.select(event.node.data)

    def action_follow(self) -> None:
        self.view.toggle_follow()

    def action_close(self) -> None:
        self.app.pop_screen()


def agents_line(book: StepBook) -> str:
    """Sub-agents at a glance, for the strip under the prompt."""
    children = [a for a in book.agents.values() if a.parent is not None]
    if not children:
        return ""
    running = sum(a.status == RUNNING for a in children)
    done = sum(a.status == DONE for a in children)
    failed = sum(a.status == FAILED for a in children)
    parts = [f"[#c9a4ff]● {running} running[/]", f"[#7a7a7a]✓ {done} done[/]"]
    if failed:
        parts.append(f"[#a05050]✗ {failed} failed[/]")
    return "sub-agents  " + "   ".join(parts) + "   [dim]F5 dashboard[/dim]"


__all__ = ["DashboardScreen", "RunView", "RunsColumn", "StepPanes", "agents_line", "query_line"]
