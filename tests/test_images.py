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


async def test_the_default_engine_says_images_need_the_session_engine(tmp_path):
    (tmp_path / "shot.png").write_bytes(PNG)
    session = await CodingSession.create(provider=ReplayProvider([]), provider_name="replay",
                                         model="m", system="s", tools=[], cwd=str(tmp_path))
    session.backend = object()
    assert "--engine session" in await session.handle_command("/image shot.png")
    session.backend = None
    await session.shutdown()
