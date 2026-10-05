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


def test_prune_drops_oldest_bytes_but_keeps_the_newest_oversized_dump(tmp_path):
    _dump(tmp_path, "request_dump_sess_old.json", "o" * 50)
    _dump(tmp_path, "request_dump_sess_new.json", "n" * 80)
    prune_request_dumps(tmp_path, "sess", keep=20, max_bytes=60)
    assert not (tmp_path / "request_dump_sess_old.json").exists()
    assert (tmp_path / "request_dump_sess_new.json").read_text(encoding="utf-8") == "n" * 80
