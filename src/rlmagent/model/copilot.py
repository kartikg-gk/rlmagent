"""Copilot subscription inference through the official runtime SDK."""

import asyncio
import tempfile
from contextlib import suppress
from uuid import uuid4

from rlmagent.harness.contracts.transcript import CallBlock, ModelEntry, TextSegment, UsageStats
from rlmagent.harness.provider.wire import CallCloseEvent, StreamCloseEvent, StreamFaultEvent
from rlmagent.model.external.textcalls import PROTOCOL, parse_reply, render


class CopilotProvider:
    def __init__(self, client_factory=None):
        self.client_factory = client_factory

    async def close(self):
        pass

    async def stream_response(self, *, model, system, messages, tools, signal=None, cache_key=None):
        client, session, task, temporary = None, None, None, None
        try:
            from copilot import CopilotClient
            from copilot.generated.rpc import PermissionDecisionReject

            coding = any(tool.name == "python" for tool in tools)
            prompt = system + ("\n\n" + PROTOCOL if coding else "") + "\n\n" + render(messages)
            with (
                temporary := tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            ) as directory:
                client = (self.client_factory or CopilotClient)(
                    working_directory=directory, use_logged_in_user=True
                )
                await client.start()

                def deny(*args, **kwargs):
                    return PermissionDecisionReject(feedback="Workspace actions belong to the host")

                session = await client.create_session(
                    model=model,
                    available_tools=[],
                    tools=[],
                    on_permission_request=deny,
                    working_directory=directory,
                    skip_custom_instructions=True,
                    enable_config_discovery=False,
                    enable_file_hooks=False,
                    enable_host_git_operations=False,
                    enable_skills=False,
                    enable_session_store=False,
                    mcp_servers={},
                    enable_session_telemetry=False,
                    custom_agents=[],
                    streaming=True,
                )
                usage = UsageStats()

                def event_seen(event):
                    kind = getattr(
                        getattr(event, "type", None), "value", getattr(event, "type", "")
                    )
                    if kind == "assistant.usage":
                        data = event.data
                        usage.input += max(
                            0,
                            (getattr(data, "input_tokens", 0) or 0)
                            - (getattr(data, "cache_read_tokens", 0) or 0),
                        )
                        usage.output += getattr(data, "output_tokens", 0) or 0
                        usage.cache_read += getattr(data, "cache_read_tokens", 0) or 0

                session.on(event_seen)
                task = asyncio.create_task(session.send_and_wait(prompt, timeout=120))
                while not task.done():
                    if signal is not None and signal.is_cancelled():
                        await session.abort()
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task
                        yield StreamFaultEvent(
                            reason="aborted", error=ModelEntry(stop_reason="aborted")
                        )
                        return
                    await asyncio.wait({task}, timeout=0.05)
                reply = task.result()
                if reply is None or not getattr(reply.data, "content", ""):
                    raise ValueError("Copilot returned no answer")
                text, code = (
                    parse_reply(reply.data.content) if coding else (reply.data.content, None)
                )
                content = [TextSegment(text=text)] if text else []
                reason = "stop"
                if code is not None:
                    call = CallBlock(id=uuid4().hex, name="python", arguments={"code": code})
                    content.append(call)
                    reason = "toolUse"
                usage.total_tokens = usage.input + usage.output + usage.cache_read
                message = ModelEntry(
                    content=content,
                    model=model,
                    provider="copilot",
                    api="copilot-sdk",
                    stop_reason=reason,
                    usage=usage,
                )
                if code is not None:
                    yield CallCloseEvent(
                        content_index=len(content) - 1, tool_call=call, partial=message
                    )
                yield StreamCloseEvent(reason=reason, message=message)
        except ImportError:
            yield StreamFaultEvent(
                reason="error",
                error=ModelEntry(
                    stop_reason="error",
                    error_message="Install rlmagent[copilot] and sign in with copilot login",
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            yield StreamFaultEvent(
                reason="error",
                error=ModelEntry(
                    stop_reason="error",
                    error_message=(f"Copilot request failed ({type(error).__name__}); "
                                   "check copilot login"),
                ),
            )
        finally:
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            if session is not None:
                with suppress(Exception):
                    await session.disconnect()
            if client is not None:
                with suppress(Exception):
                    await client.stop()
            if temporary is not None:
                temporary.cleanup()
