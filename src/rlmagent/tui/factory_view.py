"""Read-only native view of a detached factory run."""

from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Static

from rlmagent.factory_graph import run_status


class FactoryGraphApp(App[None]):
    CSS = """
    #run-state { height: auto; }
    DataTable { height: 1fr; }
    #stage-result { height: 8; overflow-y: auto; }
    """
    BINDINGS = [("q", "quit", "Close")]

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = directory
        self.snapshot = None
        self.details: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("", id="run-state", markup=False)
        yield DataTable(cursor_type="row")
        yield Static("", id="stage-result", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(DataTable).add_columns("Stage", "Dependencies", "Guard", "Status")
        self.refresh_run()
        self.set_interval(0.5, self.refresh_run)

    def refresh_run(self) -> None:
        heading = self.query_one("#run-state", Static)
        try:
            state = run_status(self.directory)
        except (OSError, ValueError, KeyError) as exc:
            self.snapshot = None
            heading.update(f"Could not read the run: {exc}")
            return
        if state == self.snapshot:
            return
        self.snapshot = state
        reason = state.get("error") or state.get("stop_reason")
        heading.update(f"Factory {state['id']} · {state['status']}"
                       + (f" · {reason}" if reason else ""))
        graph, records = state.get("graph", {}), state.get("stages", {})
        table = self.query_one(DataTable)
        selected = table.cursor_row
        table.clear()
        self.details.clear()
        for name in dict.fromkeys([*graph, *records]):
            node = graph.get(name, {})
            record = records.get(name, {})
            dependencies = ", ".join(node.get("after", ()))
            guard = node.get("when", "")
            if source := node.get("foreach"):
                guard += f" · foreach {source}"
            table.add_row(Text(name), Text(dependencies), Text(guard),
                          Text(record.get("status", "waiting")))
            self.details.append(self._output(record))
            for index, item in enumerate(record.get("items", ()), 1):
                table.add_row(Text(f"  ↳ item {index}"), "", "", Text(item["status"]))
                self.details.append(self._output(item))
        if self.details:
            selected = min(selected, len(self.details) - 1)
            table.move_cursor(row=selected)
        self._show_result(selected)

    @staticmethod
    def _output(record: dict) -> str:
        output = record.get("output")
        if isinstance(output, list):
            return "\n".join(f"{index}. {item}" for index, item in enumerate(output, 1))
        return str(output or "")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_result(event.cursor_row)

    def _show_result(self, row: int) -> None:
        self.query_one("#stage-result", Static).update(
            self.details[row] if 0 <= row < len(self.details) else "")
