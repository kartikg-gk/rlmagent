import json

from _scripted import code_turn, text_turn
from test_improve import PROPOSAL, SKILL, _setup, _submit

ROLE = {"action": "create", "kind": "role", "id": "reviewer", "title": "Reviewer",
        "content": "Review the code you are given and list concrete bugs only."}
YES = json.dumps({"worth_it": True, "why": "user correction", "focus": "the test command"})
NO = json.dumps({"worth_it": False, "why": "nothing new", "focus": ""})


def _auto(improver, every=2, cooldown=0.0):
    improver.auto = True
    improver.every = every
    improver.cooldown = cooldown


async def test_auto_check_runs_every_n_turns_and_can_say_no(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        text_turn("a"), text_turn("b"), text_turn(NO),
    ])
    _auto(improver, every=2)
    try:
        await _submit(session, "one")
        assert len(provider.calls) == 1          # no check after one turn
        await _submit(session, "two")
        assert len(provider.calls) == 3          # check ran, said no, no planner
        assert improver.entries() == []
    finally:
        await session.shutdown()


async def test_auto_check_yes_runs_the_planner(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        text_turn("a"), text_turn(YES), text_turn(PROPOSAL),
    ])
    _auto(improver, every=1)
    try:
        await _submit(session, "tests use pytest -q")
        assert [e.id for e in improver.entries()] == ["tests"]
        assert improver.local.history[-1]["trigger"] == "auto:turns"
    finally:
        await session.shutdown()


async def test_cooldown_holds_back_the_next_check(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        text_turn("a"), text_turn(NO), text_turn("b"), text_turn("c"),
    ])
    _auto(improver, every=1, cooldown=3600)
    try:
        await _submit(session, "one")
        await _submit(session, "two")
        await _submit(session, "three")
        assert len(provider.calls) == 4          # one check only
    finally:
        await session.shutdown()


async def test_auto_off_never_checks(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn("a"), text_turn("b")])
    improver.auto, improver.every = False, 1
    try:
        await _submit(session, "one")
        await _submit(session, "two")
        assert len(provider.calls) == 2
    finally:
        await session.shutdown()


async def test_explicit_requests_skip_the_check(tmp_path):
    session, provider, improver = await _setup(tmp_path, [text_turn(PROPOSAL)])
    _auto(improver, every=1000)
    try:
        await session.handle_command("/improve keep it")
        assert len(provider.calls) == 1 and improver.entries()
    finally:
        await session.shutdown()


async def test_a_role_reaches_the_sub_agents_prompt(tmp_path):
    session, provider, improver = await _setup(tmp_path, [
        code_turn("print(await rlm('check this', 'x = 1', role='reviewer'))"),
        code_turn("FINAL('no bugs')"), text_turn("ok"), text_turn("done"),
    ])
    try:
        improver.apply([ROLE], trigger="test")
        await _submit(session, "go")
        child_system = provider.calls[1][1]
        assert "list concrete bugs only" in child_system
    finally:
        await session.shutdown()


async def test_an_unknown_role_is_an_error(tmp_path):
    session, provider, improver = await _setup(tmp_path, [])
    try:
        out = await session.agent_tree.root_kernel().run("await spawn('x', role='nobody')")
        assert "nobody" in out.error
    finally:
        await session.shutdown()


def test_cli_accepts_the_auto_improve_settings():
    from rlmagent_app.cli.main import _build_run_parser

    ns = _build_run_parser().parse_args(["--no-auto-improve", "--improve-every", "5",
                                         "--improve-cooldown", "60"])
    assert (ns.auto_improve, ns.improve_every, ns.improve_cooldown) == (False, 5, 60.0)
    ns = _build_run_parser().parse_args([])
    assert (ns.auto_improve, ns.improve_every, ns.improve_cooldown) == (True, 20, 900.0)
