import asyncio
from dataclasses import dataclass

from textual.app import App

from rlmagent.tui.adapter import SessionBridge, StepChanged, TextDelta, TurnFinished
from rlmagent.tui.steps import DONE, FAILED, RUNNING, StepBook, StepCard, footer_text, header_text
from rlmagent.tui.transcript import TranscriptView


@dataclass
class Usage:
    total_tokens: int
    cost: float | None


def _run(book, run_id="root", parent=None, depth=0):
    book.feed({"kind": "run_started", "run_id": run_id, "parent_run_id": parent,
               "depth": depth, "model": "m"})


def test_book_folds_a_step_from_start_to_finish():
    book = StepBook()
    _run(book)
    assert book.feed({"kind": "step_started", "run_id": "root", "step": 0}) is None
    step = book.feed({"kind": "step_started", "run_id": "root", "step": 1, "started": 10.0})
    assert header_text(step) == "RLM d0 · root · step 1 · generating"
    book.feed({"kind": "code_generated", "run_id": "root", "step": 1, "code": "print(1)"})
    assert step.state == RUNNING and step.code == "print(1)"
    book.feed({"kind": "output_received", "run_id": "root", "step": 1, "output": "1\n",
               "error": False})
    book.feed({"kind": "step_completed", "run_id": "root", "step": 1,
               "usage": Usage(7300, 0.0), "ended": 12.5})
    assert step.state == DONE
    assert footer_text(step) == "7.3k tokens · $0.0000 · 2.5s"


def test_child_steps_carry_depth_and_a_short_id():
    book = StepBook()
    _run(book)
    _run(book, "a1b2c3d4e5f6", "root", 1)
    step = book.feed({"kind": "step_started", "run_id": "a1b2c3d4e5f6", "step": 1})
    assert step.depth == 1 and step.short == "d4e5f6"
    book.feed({"kind": "step_completed", "run_id": "a1b2c3d4e5f6", "step": 1,
               "usage": {"prompt_tokens": 5, "completion_tokens": 5}, "error": True})
    assert step.state == FAILED and footer_text(step).startswith("10 tokens · cost unknown")
    book.feed({"kind": "run_completed", "run_id": "a1b2c3d4e5f6"})
    assert book.totals() == {"steps": 1, "tokens": 10, "cost": None, "agents": 2,
                             "running": 1, "depth": 1}


async def test_cards_show_code_and_fold_long_output():
    book = StepBook()
    _run(book)

    class Host(App):
        def compose(self):
            yield TranscriptView()

    async with Host().run_test() as pilot:
        view = pilot.app.query_one(TranscriptView)
        step = book.feed({"kind": "step_started", "run_id": "root", "step": 1})
        view.show_step(step)
        await pilot.pause()
        book.feed({"kind": "code_generated", "run_id": "root", "step": 1, "code": "x = 1"})
        book.feed({"kind": "output_received", "run_id": "root", "step": 1,
                   "output": "\n".join(str(i) for i in range(20))})
        card = view.show_step(step)
        await pilot.pause()
        assert len(view.cards) == 1 and isinstance(card, StepCard)
        assert card.has_class("step-live") and card.region.height > 0
        assert card._more is not None and card._more.title == "14 more lines"
        book.feed({"kind": "step_completed", "run_id": "root", "step": 1,
                   "usage": Usage(10, 0.001)})
        view.show_step(step)
        await pilot.pause()
        assert not card.has_class("step-live")
        view.new_run()
        again = view.show_step(book.feed({"kind": "step_started", "run_id": "root", "step": 1}))
        assert again is not card


class FakeBackend:
    def __init__(self):
        self.listeners = []
        self.cards = False


class FakeSession:
    def __init__(self, backend, events):
        self.backend = backend
        self._events = events

    async def submit(self, text):
        assert self.backend.cards
        for listener in self.backend.listeners:
            listener({"kind": "run_started", "run_id": "root", "depth": 0})
            listener({"kind": "step_started", "run_id": "root", "step": 1})
        await asyncio.sleep(0.05)
        for event in self._events:
            yield event


async def test_bridge_merges_step_events_with_the_reply():
    from rlmagent.harness.contracts.stream import MessageEndEvent
    from rlmagent.harness.contracts.transcript import ModelEntry, TextSegment

    backend = FakeBackend()
    end = MessageEndEvent(message=ModelEntry(model="m", content=[TextSegment(text="done")]))
    bridge = SessionBridge(FakeSession(backend, [end]))
    updates = [u async for u in bridge.submit("hi")]
    assert isinstance(updates[0], StepChanged) and updates[0].step.number == 1
    assert isinstance(updates[-1], TurnFinished) and updates[-1].text == "done"
    assert not any(isinstance(u, TextDelta) for u in updates)
    assert backend.listeners == [] and backend.cards is False
    assert bridge.budget_line() == "steps 1 1 running depth 0"
