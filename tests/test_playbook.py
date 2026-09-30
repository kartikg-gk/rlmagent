import json

from rlmagent_app.playbook.apply import apply_edits, reverse_edits, validate
from rlmagent_app.playbook.digest import fingerprint, render
from rlmagent_app.playbook.plan import build_request, parse_proposal
from rlmagent_app.playbook.store import Entry, Playbook, load, merged, save

SKILL = "def count_lines(path):\n    return len(open(path).read().splitlines())\n"


def _edit(**kw):
    base = {"action": "create", "kind": "memory", "id": "m1", "title": "t", "content": "c"}
    return {**base, **kw}


def test_store_round_trip_and_corrupt_file(tmp_path):
    path = tmp_path / "pb.json"
    pb = Playbook()
    pb.entries["m1"] = Entry(id="m1", kind="memory", title="t", content="c", scope="local")
    save(path, pb)
    assert load(path).entries["m1"].content == "c"
    path.write_text("{broken")
    assert load(path).entries == {}
    assert load(tmp_path / "missing.json").entries == {}


def test_validate_rules():
    assert validate(_edit()) is None
    assert "action" in validate(_edit(action="rename"))
    assert "kind" in validate(_edit(kind="prompt"))
    assert "id" in validate(_edit(action="delete", id=""))
    assert "content" in validate(_edit(content=" "))
    assert validate(_edit(kind="skill", content=SKILL, callable="count_lines")) is None
    assert "callable" in validate(_edit(kind="skill", content=SKILL))
    assert "defines" in validate(_edit(kind="skill", content=SKILL, callable="other"))
    assert "Python" in validate(_edit(kind="skill", content="def (", callable="x"))


def test_apply_creates_updates_deletes_and_records_history():
    pb = Playbook()
    event = apply_edits(pb, [_edit(), _edit(id="m2")], baseline={}, scope="local",
                        trigger="user", summary="s")
    assert set(pb.entries) == {"m1", "m2"} and all(c["applied"] for c in event["changes"])
    baseline = {k: e.version for k, e in pb.entries.items()}
    apply_edits(pb, [_edit(action="update", content="new"), _edit(action="delete", id="m2")],
                baseline=baseline, scope="local", trigger="user", summary="s")
    assert pb.entries["m1"].content == "new" and pb.entries["m1"].version == 2
    assert "m2" not in pb.entries
    assert len(pb.history) == 2


def test_bad_edits_are_skipped_not_fatal():
    pb = Playbook()
    event = apply_edits(pb, [_edit(kind="nope"), _edit(id="ok")], baseline={}, scope="local",
                        trigger="user", summary="s")
    assert [c["applied"] for c in event["changes"]] == [False, True]
    assert "ok" in pb.entries


def test_an_entry_changed_since_planning_is_skipped():
    pb = Playbook()
    apply_edits(pb, [_edit()], baseline={}, scope="local", trigger="user", summary="s")
    event = apply_edits(pb, [_edit(action="update", content="x")], baseline={"m1": 0},
                        scope="local", trigger="user", summary="s")
    assert not event["changes"][0]["applied"] and "changed" in event["changes"][0]["error"]


def test_undo_reverses_an_earlier_change():
    pb = Playbook()
    first = apply_edits(pb, [_edit(), _edit(id="m2")], baseline={}, scope="local", trigger="u", summary="s")
    second = apply_edits(pb, [_edit(action="update", content="new"), _edit(action="delete", id="m2")],
                         baseline={k: e.version for k, e in pb.entries.items()},
                         scope="local", trigger="u", summary="s")
    apply_edits(pb, reverse_edits(second), baseline={k: e.version for k, e in pb.entries.items()},
                scope="local", trigger="undo", summary="undo")
    assert pb.entries["m1"].content == "c" and pb.entries["m2"].content == "c"
    apply_edits(pb, reverse_edits(first), baseline={k: e.version for k, e in pb.entries.items()},
                scope="local", trigger="undo", summary="undo")
    assert pb.entries == {}


def test_merged_view_marks_clashes_and_shared_is_read_only_locally():
    shared, local = Playbook(), Playbook()
    shared.entries["x"] = Entry(id="x", kind="rule", title="t", content="g", scope="global")
    local.entries["x"] = Entry(id="x", kind="rule", title="t", content="l", scope="local")
    ids = {e.id for e in merged(local, shared)}
    assert ids == {"x", "local:x"}
    only_shared = Playbook()
    only_shared.entries["g"] = Entry(id="g", kind="rule", title="t", content="g", scope="global")
    event = apply_edits(Playbook(), [_edit(action="update", id="g", kind="rule")], baseline={},
                        scope="local", trigger="u", summary="s", read_only=only_shared)
    assert "shared" in event["changes"][0]["error"]


def test_parse_proposal_finds_the_json():
    text = 'Here you go:\n```json\n{"summary": "s", "rationale": "r", "expected": "e", "edits": []}\n```'
    assert parse_proposal(text)["summary"] == "s"
    try:
        parse_proposal("no json here")
    except ValueError as exc:
        assert "JSON" in str(exc)
    else:
        raise AssertionError


def test_request_carries_state_history_and_the_end_of_the_conversation():
    entries = [Entry(id="m1", kind="memory", title="Tests", content="run pytest -q", scope="local")]
    req = build_request(entries, [{"id": "e1", "summary": "added m1"}], "x" * 100_000 + "TAIL",
                        scope="local", instructions="focus on tests")
    assert "run pytest -q" in req and "added m1" in req and "focus on tests" in req
    assert req.count("x") < 70_000 and "TAIL" in req


def test_digest_is_short_ranked_and_fingerprinted():
    entries = [Entry(id=f"m{i}", kind="memory", title=f"note {i}", content=f"about topic{i} " * 50,
                     scope="local") for i in range(10)]
    text = render(entries, goal="topic7")
    assert text.count("\n- ") <= 5 and "m7" in text
    assert max(len(line) for line in text.splitlines() if line.startswith("- ")) <= 260
    fp = fingerprint(entries)
    entries[0].version += 1
    assert fingerprint(entries) != fp
    assert json.dumps(render([], goal="x"))  # empty playbook renders without error
