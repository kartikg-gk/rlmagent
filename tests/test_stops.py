from types import SimpleNamespace

import pytest

from rlmagent.stops import classify, of_session


@pytest.mark.parametrize("refusal,text,error,reason", [
    (None, "the answer", None, "done"),
    (None, "", None, "no_answer"),
    ("60 calls reaches the limit of 60", "", None, "max_calls"),
    ("cost 0.5100 reaches the limit of 0.5", "", None, "max_cost"),
    ("9000 completion tokens reaches the limit of 9000", "", None, "max_tokens"),
    ("31.0s reaches the limit of 30s", "", None, "max_seconds"),
    (None, "", "agent c1 reached its limit of 500 completion tokens", "agent_tokens"),
    (None, "", "this run allows 1 compaction(s) and has used them", "compaction_failed"),
    (None, "", "provider down", "error"),
    (None, "", "cancelled", "cancelled"),
])
def test_stop_reasons(refusal, text, error, reason):
    assert classify(refusal, text, error) == reason


def test_stop_reason_of_a_session():
    turn = SimpleNamespace(role="assistant", text="", error_message=None, stop_reason="aborted")
    session = SimpleNamespace(transcript=[SimpleNamespace(role="user", text="go"), turn])
    assert of_session(session, None) == {"stop_reason": "cancelled", "stop_detail": ""}
