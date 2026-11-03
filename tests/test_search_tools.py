from rlmagent.tools import build_tool_registry
from rlmagent.tools.search import make_glob_tool, make_grep_tool


def _text(outcome):
    return "".join(part.text for part in outcome.content)


async def test_grep_finds_matching_lines_and_skips_hidden_folders(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\ndef main():\n")
    (tmp_path / "b.txt").write_text("DEF MAIN\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.py").write_text("def main\n")
    grep = make_grep_tool()
    out = _text(await grep.run("1", {"pattern": "def main", "path": str(tmp_path)}))
    assert out == f"{tmp_path / 'pkg' / 'a.py'}:2: def main():"
    out = _text(await grep.run("2", {"pattern": "def main", "path": str(tmp_path),
                                     "ignore_case": True, "glob": "*.txt"}))
    assert out.endswith("b.txt:1: DEF MAIN")
    out = _text(await grep.run("3", {"pattern": ".", "path": str(tmp_path), "limit": 1}))
    assert "stopped at 1 matches" in out
    assert "No matches" in _text(await grep.run("4", {"pattern": "zzz", "path": str(tmp_path)}))
    assert "missing" in _text(await grep.run("5", {"pattern": "(", "path": str(tmp_path)}))


async def test_glob_lists_files_by_pattern(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("")
    (tmp_path / "pkg" / "b.md").write_text("")
    glob = make_glob_tool()
    out = _text(await glob.run("1", {"pattern": "**/*.py", "path": str(tmp_path)}))
    assert out == str(tmp_path / "pkg" / "a.py")
    assert "No files" in _text(await glob.run("2", {"pattern": "*.rs", "path": str(tmp_path)}))


def test_search_tools_are_always_offered():
    names = [t.name for t in build_tool_registry()]
    assert {"Grep", "Glob"} <= set(names)
