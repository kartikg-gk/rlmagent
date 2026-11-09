"""A wall-clock deadline around one complete provider stream."""
import asyncio

from rlmagent.harness.contracts.transcript import ModelEntry
from rlmagent.harness.provider.wire import StreamFaultEvent


async def bounded_stream(stream, model, seconds=300.0):
    try:
        async with asyncio.timeout(seconds):
            async for event in stream:
                yield event
    except TimeoutError:
        yield StreamFaultEvent(reason="error", error=ModelEntry(model=model,
            stop_reason="error", error_message=f"Model request exceeded its {seconds:g}s deadline"))
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
