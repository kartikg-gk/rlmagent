"""Build scripted model turns that call the python tool."""

from itertools import count

from rlmagent_harness.contracts.transcript import CallBlock, ModelEntry, TextSegment
from rlmagent_harness.provider.wire import CallCloseEvent, StreamCloseEvent

_ids = count(1)


def code_turn(code: str) -> list:
    call = CallBlock(name="python", id=f"call{next(_ids)}", arguments={"code": code})
    partial = ModelEntry(content=[call], stop_reason="toolUse")
    return [
        CallCloseEvent(content_index=0, tool_call=call, partial=partial),
        StreamCloseEvent(reason="toolUse", message=partial),
    ]


def text_turn(text: str) -> list:
    message = ModelEntry(content=[TextSegment(text=text)], stop_reason="stop")
    return [StreamCloseEvent(reason="stop", message=message)]
