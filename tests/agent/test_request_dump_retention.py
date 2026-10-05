"""Request-dump retention keeps a bounded newest set and leaves other files."""

from __future__ import annotations

import os
import time

from agent.agent_runtime_helpers import prune_request_dumps


def _dump(directory, name: str, payload: str = "x") -> None:
    path = directory / name
    path.write_text(payload, encoding="utf-8")
    stamp = time.time() + len(list(directory.iterdir()))
    os.utime(path, (stamp, stamp))


def test_prune_keeps_newest_per_session_and_other_files(tmp_path):
    for index in range(25):
        _dump(tmp_path, f"request_dump_sess_{index:02d}.json", "body")
    _dump(tmp_path, "session_sess.json", "conversation")
    _dump(tmp_path, "request_dump_other_0.json", "other")
    prune_request_dumps(tmp_path, "sess", keep=20, max_bytes=10**9)
    kept = sorted(path.name for path in tmp_path.glob("request_dump_sess_*.json"))
    assert kept == [f"request_dump_sess_{index:02d}.json" for index in range(5, 25)]
    assert (tmp_path / "session_sess.json").read_text(encoding="utf-8") == "conversation"
    assert (tmp_path / "request_dump_other_0.json").is_file()


def test_prune_drops_an_oversized_newest_dump_to_meet_the_budget(tmp_path):
    _dump(tmp_path, "request_dump_sess_old.json", "o" * 50)
    _dump(tmp_path, "request_dump_sess_new.json", "n" * 80)
    _dump(tmp_path, "session_sess.json", "conversation")
    result = prune_request_dumps(tmp_path, "sess", keep=20, max_bytes=60)
    assert result["within_budget"] is True
    assert result["cleanup_failed"] is False
    assert result["bytes_remaining"] <= 60
    assert not (tmp_path / "request_dump_sess_old.json").exists()
    assert not (tmp_path / "request_dump_sess_new.json").exists()
    assert (tmp_path / "session_sess.json").read_text(encoding="utf-8") == "conversation"


def test_cleanup_failure_does_not_claim_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.agent_runtime_helpers._unlink_dump", lambda _path: False)
    _dump(tmp_path, "request_dump_sess_new.json", "n" * 80)
    result = prune_request_dumps(tmp_path, "sess", keep=20, max_bytes=60)
    assert result["cleanup_failed"] is True
    assert result["within_budget"] is False
    assert result["bytes_remaining"] == 80
    assert (tmp_path / "request_dump_sess_new.json").is_file()
