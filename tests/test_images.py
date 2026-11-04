import base64

from _scripted import text_turn

from rlmagent.conversation import CodingSession
from rlmagent.model.scripted import ReplayProvider

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


async def test_an_attached_image_goes_with_the_next_message_only(tmp_path):
    (tmp_path / "shot.png").write_bytes(PNG)
    (tmp_path / "notes.txt").write_text("x")
    provider = ReplayProvider([text_turn("I see it"), text_turn("ok")])
    session = await CodingSession.create(provider=provider, provider_name="replay", model="m",
                                         system="s", tools=[], cwd=str(tmp_path))
    assert (await session.handle_command("/image")).startswith("0 image(s)")
    assert await session.handle_command("/image missing.png") == f"No file at {tmp_path / 'missing.png'}."
    assert await session.handle_command("/image notes.txt") == "Attach a PNG, JPEG, GIF or WebP image."
    assert (await session.handle_command("/image shot.png")).startswith("Attached shot.png")
    async for _ in session.submit("what is in this picture?"):
        pass
    first = [m for m in session.transcript if m.role == "user"][0]
    assert [getattr(part, "type", "") for part in first.content] == ["text", "image"]
    async for _ in session.submit("thanks"):
        pass
    second = [m for m in session.transcript if m.role == "user"][1]
    assert isinstance(second.content, str)
    await session.shutdown()


async def test_the_default_engine_passes_images_to_the_root_model(tmp_path):
    from types import SimpleNamespace

    import pytest
    from rlmness import Allowance

    from rlmagent.agents.approval import Approver
    from rlmagent.backend import BackendProvider
    from rlmagent.harness.contracts.transcript import HumanEntry
    from rlmagent.kernel.wasm import available
    from rlmagent.tools.files import _read_image

    if not available():
        pytest.skip("sandbox unavailable")
    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    image = _read_image(path).content[0]
    provider = ReplayProvider([
        text_turn("```python\nvalue = await rlm(21)\nFINAL(value)\n```"),
        text_turn("```python\nFINAL(PROMPT * 2)\n```"),
        text_turn("```python\nFINAL(43)\n```"),
    ])
    tree = SimpleNamespace(cwd=str(tmp_path), approver=Approver("deny"), network=False,
                           cell_timeout=30, allowance=Allowance(max_calls=10, max_cost=2))
    backend = BackendProvider(provider, provider, tree)
    try:
        messages = [HumanEntry(content=[image])]
        for question, expected in [(messages, "42"), ([*messages, HumanEntry(content="next")], "43")]:
            events = [event async for event in backend.stream_response(
                model="m", system="", messages=question, tools=[SimpleNamespace(name="python")],
            )]
            assert events[-1].message.text == expected
        first, child, second = [call[2][0].content for call in provider.calls]
        assert first[1] == image
        assert isinstance(child, str)
        assert isinstance(second, str)
    finally:
        await backend.close()
