"""Canonical filesystem layout and automatic resource discovery."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from rlmagent_harness.contracts.values import JValue

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

RLM_AGENT_HOME_ENV = "RLM_AGENT_HOME"
RLM_AGENT_SESSIONS_DIR_ENV = "RLM_AGENT_SESSIONS_DIR"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)


def _resolve_home() -> Path:
    """Resolve rlm-agent's home directory from ``$RLM_AGENT_HOME`` or ``~/.rlm-agent``."""
    env = os.environ.get(RLM_AGENT_HOME_ENV)
    return Path(env) if env else Path.home() / ".rlm-agent"


def _resolve_sessions(home: Path) -> Path:
    """Resolve the sessions directory from ``$RLM_AGENT_SESSIONS_DIR`` or ``<home>/sessions``."""
    env = os.environ.get(RLM_AGENT_SESSIONS_DIR_ENV)
    return Path(env) if env else home / "sessions"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Remove duplicate paths while preserving order."""
    seen: set[Path] = set()
    result: list[Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Path utility
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RlmAgentPaths:
    """Canonical filesystem locations for rlm-agent."""

    home: Path = field(default_factory=_resolve_home)
    project: Path = field(default_factory=Path.cwd)
    # False when the project folder is not trusted: search paths then leave
    # out every project-local location and only user resources load.
    project_enabled: bool = True

    # ── user home locations ───────────────────────────────────────────

    @property
    def sessions(self) -> Path:
        """Session transcript and index storage (``~/.rlm-agent/sessions``)."""
        return _resolve_sessions(self.home)

    @property
    def logs(self) -> Path:
        """Log file directory (``~/.rlm-agent/logs``)."""
        return self.home / "logs"

    @property
    def user_skills(self) -> Path:
        """User-level skill definitions (``~/.rlm-agent/skills``)."""
        return self.home / "skills"

    @property
    def user_prompts(self) -> Path:
        """User-level prompt templates (``~/.rlm-agent/prompts``)."""
        return self.home / "prompts"

    @property
    def user_themes(self) -> Path:
        """User-level theme files (``~/.rlm-agent/themes``)."""
        return self.home / "themes"

    # ── user .agents locations ────────────────────────────────────────

    @property
    def agents_home(self) -> Path:
        """Cross-tool agent resource directory (``~/.agents``)."""
        return self.home.parent / ".agents"

    @property
    def agents_skills(self) -> Path:
        """Cross-tool skill definitions (``~/.agents/skills``)."""
        return self.agents_home / "skills"

    @property
    def agents_prompts(self) -> Path:
        """Cross-tool prompt templates (``~/.agents/prompts``)."""
        return self.agents_home / "prompts"

    # ── project-local .rlm-agent locations ────────────────────────────────

    @property
    def project_skills(self) -> Path:
        """Project-level skill definitions (``<project>/.rlm-agent/skills``)."""
        return self.project / ".rlm-agent" / "skills"

    @property
    def project_prompts(self) -> Path:
        """Project-level prompt templates (``<project>/.rlm-agent/prompts``)."""
        return self.project / ".rlm-agent" / "prompts"

    @property
    def project_themes(self) -> Path:
        """Project-level theme files (``<project>/.rlm-agent/themes``)."""
        return self.project / ".rlm-agent" / "themes"

    # ── project-local .agents locations ───────────────────────────────

    @property
    def project_agents(self) -> Path:
        """Project-level cross-tool agent directory (``<project>/.agents``)."""
        return self.project / ".agents"

    @property
    def project_agents_skills(self) -> Path:
        """Project-level cross-tool skills (``<project>/.agents/skills``)."""
        return self.project_agents / "skills"

    @property
    def project_agents_prompts(self) -> Path:
        """Project-level cross-tool prompts (``<project>/.agents/prompts``)."""
        return self.project_agents / "prompts"


# ---------------------------------------------------------------------------
# Default instance factory
# ---------------------------------------------------------------------------


def default_paths(
    *,
    home: Path | None = None,
    project: Path | None = None,
) -> RlmAgentPaths:
    """Build a ``RlmAgentPaths`` with optional overrides for home and project."""
    from rlmagent_app.trust import project_inputs_allowed

    root = project or Path.cwd()
    return RlmAgentPaths(
        home=home or _resolve_home(),
        project=root,
        project_enabled=project_inputs_allowed(root),
    )


# ---------------------------------------------------------------------------
# Resource search paths — precedence-ordered directory lists
# ---------------------------------------------------------------------------


def _project_only(paths: RlmAgentPaths, *locations: Path) -> tuple[Path, ...]:
    """``locations`` when project inputs are enabled, otherwise nothing."""
    return locations if paths.project_enabled else ()


def skill_search_paths(paths: RlmAgentPaths) -> list[Path]:
    """Return skill directories in highest-precedence-first order."""
    return _dedupe_paths([
        *_project_only(paths, paths.project_agents_skills, paths.project_skills),
        paths.agents_skills,
        paths.user_skills,
    ])


def prompt_search_paths(paths: RlmAgentPaths) -> list[Path]:
    """Return prompt directories in highest-precedence-first order."""
    return _dedupe_paths([
        *_project_only(paths, paths.project_agents_prompts, paths.project_prompts),
        paths.agents_prompts,
        paths.user_prompts,
    ])


def theme_search_paths(paths: RlmAgentPaths) -> list[Path]:
    """Return theme directories in highest-precedence-first order."""
    return _dedupe_paths([
        *_project_only(paths, paths.project_themes),
        paths.user_themes,
    ])


def resource_search_paths(
    paths: RlmAgentPaths,
    resource_type: str,
    *,
    rlm_agent_only: bool = False,
) -> list[Path]:
    """Return search directories for an arbitrary resource type."""
    dirs: list[Path] = []
    if paths.project_enabled:
        if not rlm_agent_only:
            dirs.append(paths.project / ".agents" / resource_type)
        dirs.append(paths.project / ".rlm-agent" / resource_type)
    if not rlm_agent_only:
        dirs.append(paths.agents_home / resource_type)
    dirs.append(paths.home / resource_type)
    return _dedupe_paths(dirs)


# ---------------------------------------------------------------------------
# Resource discovery diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResourceDiagnostic:
    """Non-fatal diagnostic emitted during resource discovery."""

    resource_name: str
    winner: Path
    overridden: Path
    message: str


def collect_override_diagnostics(
    discovered: dict[str, Path],
    new_resources: dict[str, Path],
    source_dir: Path,
) -> list[ResourceDiagnostic]:
    """Produce diagnostics when resources in *new_resources* would be
    overridden by existing entries in *discovered*."""
    diagnostics: list[ResourceDiagnostic] = []
    for name, new_path in new_resources.items():
        if name in discovered:
            diagnostics.append(ResourceDiagnostic(
                resource_name=name,
                winner=discovered[name],
                overridden=new_path,
                message=(
                    f"Resource {name!r} from {source_dir} overridden by "
                    f"higher-precedence copy at {discovered[name]}"
                ),
            ))
    return diagnostics


# ---------------------------------------------------------------------------
# Markdown resource helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkdownResource:
    """A parsed markdown file with optional frontmatter metadata."""

    path: Path
    body: str
    metadata: dict[str, str]

    @property
    def name(self) -> str:
        """The ``name`` from frontmatter, or the file/directory stem."""
        return self.metadata.get("name", self.path.stem)

    @property
    def description(self) -> str:
        """The ``description`` from frontmatter, or the first non-heading paragraph."""
        explicit = self.metadata.get("description")
        if explicit:
            return explicit
        return derive_description(self.body)

    def metadata_json(self) -> dict[str, JValue]:
        """Return metadata as a JSON-compatible dictionary."""
        return dict(self.metadata)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split optional YAML-like frontmatter from the markdown body."""
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return {}, text

    raw = match.group(1)
    body = text[match.end():]
    meta: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        colon = stripped.find(":")
        if colon < 1:
            continue
        key = stripped[:colon].strip()
        value = stripped[colon + 1:].strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        meta[key] = value
    return meta, body


def derive_description(body: str) -> str:
    """Extract the first non-heading, non-blank paragraph as a description."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("---"):
            continue
        return stripped
    return ""


def parse_markdown(path: Path) -> MarkdownResource:
    """Read and parse a markdown file into a ``MarkdownResource``."""
    raw = path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw)
    return MarkdownResource(path=path, body=body, metadata=metadata)


__all__ = [
    "RLM_AGENT_HOME_ENV",
    "RLM_AGENT_SESSIONS_DIR_ENV",
    "RlmAgentPaths",
    "MarkdownResource",
    "ResourceDiagnostic",
    "collect_override_diagnostics",
    "default_paths",
    "derive_description",
    "parse_front_matter",
    "parse_markdown",
    "prompt_search_paths",
    "resource_search_paths",
    "skill_search_paths",
    "theme_search_paths",
]
