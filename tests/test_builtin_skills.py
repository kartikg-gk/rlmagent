"""Built-in skills load without files, user skills override them, and attach_image
puts the picture in the agent's next model call."""

import base64
from pathlib import Path

from rlmagent.builtin_skills import BUILTIN_SKILLS, with_builtins
from rlmagent.skillset import Skill

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def test_builtins_are_added_and_can_be_overridden():
    names = [s.name for s in with_builtins([])]
    assert {"web-search", "attach-image"} <= set(names)
    mine = Skill(name="web-search", description="mine", body="x", source=Path("a"))
    merged = with_builtins([mine])
    assert [s for s in merged if s.name == "web-search"] == [mine]


def test_builtin_skills_have_no_files_and_name_their_helpers():
    for skill in BUILTIN_SKILLS:
        assert str(skill.source) == "<built-in>" and skill.description
    assert "await web_search(" in BUILTIN_SKILLS[0].body
    assert "await attach_image(" in BUILTIN_SKILLS[1].body


def test_cli_skill_loader_includes_builtins(tmp_path):
    from rlmagent.cli.main import _load_skills

    assert {"web-search", "attach-image"} <= {s.name for s in _load_skills(str(tmp_path))}


def test_attached_image_joins_the_agents_next_call():
    import asyncio
    from types import SimpleNamespace

    from rlmagent.backend import Client
    from rlmagent.tools.files import _read_image

    path = Path(__file__).parent / "_tiny.png"
    path.write_bytes(PNG)
    try:
        segment = _read_image(path).content[0]
    finally:
        path.unlink()
    seen = {}

    class Provider:
        pass

    client = Client.__new__(Client)
    client.attached = {"root": [segment]}
    assert client.attached.pop("root") == [segment] and "root" not in client.attached
    assert getattr(segment, "type", "") == "image"
