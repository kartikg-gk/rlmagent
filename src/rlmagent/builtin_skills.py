"""Skills that ship with rlm-agent, defined in code so they need no files on disk.

A skill with the same name in a user or project skill folder replaces the built-in one.
"""

from __future__ import annotations

from pathlib import Path

from rlmagent.skillset import Skill

_SOURCE = Path("<built-in>")

WEB_SEARCH = Skill(
    name="web-search",
    description=(
        "Look things up on the web: current facts, library docs, error messages, release "
        "notes, anything that may have changed since training. Use it before guessing about "
        "versions, APIs or recent events."
    ),
    body="""\
Search from the Python kernel and read what comes back before answering.

    hits = await web_search("httpx timeout configuration", count=5)
    for h in hits:
        print(h["title"], h["url"])
        print(h.get("snippet", "")[:300])

- Each hit is a dict with a title, a url and a short snippet. Treat it as a lead: open
  the page with `await fetch_url(url)` when the snippet is not enough.
- Search needs a key in the environment: BRAVE_API_KEY, TAVILY_API_KEY or SERPER_API_KEY.
  Without one, `web_search` says so; tell the user which variable to set instead of
  inventing an answer.
- Runs started with --no-network cannot search.
- Prefer a few precise queries over many vague ones, and say which sources an answer
  came from.
""",
    source=_SOURCE,
)

ATTACH_IMAGE = Skill(
    name="attach-image",
    description=(
        "Look at an image file yourself: screenshots, diagrams, charts, photos or scanned "
        "pages. Use it when you need to see what an image shows, not to edit its pixels."
    ),
    body="""\
Put an image in front of yourself from the Python kernel:

    print(await attach_image("docs/screenshot.png"))

- PNG, JPEG, GIF and WebP files inside the workspace work. The image arrives with your
  next step, so end the cell after attaching and describe what you see in the next one.
- It is shown once; write down anything you will need later.
- The model has to support images. If it does not, the request fails with an error from
  the provider; say so to the user rather than guessing at the contents.
- To measure, crop or convert an image, open it with a library such as Pillow instead.
  That works on the pixels but does not let you see the picture.
""",
    source=_SOURCE,
)

BUILTIN_SKILLS: tuple[Skill, ...] = (WEB_SEARCH, ATTACH_IMAGE)


def with_builtins(skills: list) -> list:
    """Add each built-in skill that no loaded skill already overrides by name."""
    names = {getattr(skill, "name", None) for skill in skills}
    return [*skills, *(skill for skill in BUILTIN_SKILLS if skill.name not in names)]
