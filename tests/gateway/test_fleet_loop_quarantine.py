"""Legacy DM quarantine selects by chat id and leaves other work alone."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gateway.fleet_loop_quarantine import QuarantineError, apply_quarantine, plan_quarantine

CHAT = "11111111-1111-4111-8111-111111111111"
OTHER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
PARENT = "ab" * 32
HUMAN = "424242"


def _home(tmp_path: Path) -> Path:
    root = tmp_path / "hermes-home"
    (root / "sessions").mkdir(parents=True)
    (root / "buzz").mkdir()
    conn = sqlite3.connect(root / "state.db")
    conn.execute(
        """CREATE TABLE delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            platform TEXT,
            chat_id TEXT,
            state TEXT,
            content TEXT,
            reply_to_message_id TEXT,
            updated_at REAL,
            last_error TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE gateway_routing (
            rowid INTEGER PRIMARY KEY,
            session_key TEXT,
            entry_json TEXT
        )"""
    )
    rows = [
        ("startup", CHAT, "pending", "gateway is online", None),
        ("error", CHAT, "failed", "HTTP 402", PARENT),
        ("response", CHAT, "attempting", "the worker finished", None),
        ("delivered", CHAT, "delivered", "already sent", None),
        ("human", OTHER, "pending", "HTTP 402 startup", None),
    ]
    conn.executemany(
        "INSERT INTO delivery_obligations VALUES (?, 'buzz', ?, ?, ?, ?, 0, NULL)",
        rows,
    )
    conn.execute(
        "INSERT INTO gateway_routing (session_key, entry_json) VALUES (?, ?)",
        (
            "agent:main:buzz:dm:" + CHAT,
            json.dumps(
                {
                    "resume_pending": True,
                    "resume_reason": "restart",
                    "origin": {"chat_id": CHAT, "platform": "buzz"},
                }
            ),
        ),
    )
    conn.execute(
        "INSERT INTO gateway_routing (session_key, entry_json) VALUES (?, ?)",
        (
            "agent:main:telegram:dm:" + HUMAN,
            json.dumps(
                {
                    "resume_pending": True,
                    "resume_reason": "restart",
                    "origin": {"chat_id": HUMAN, "platform": "telegram"},
                }
            ),
        ),
    )
    conn.commit()
    conn.close()
    (root / "sessions" / "sessions.json").write_text(
        json.dumps(
            {
                "_README": "mirror",
                "agent:main:buzz:dm:" + CHAT: {
                    "resume_pending": True,
                    "origin": {"chat_id": CHAT},
                },
                "agent:main:telegram:dm:" + HUMAN: {
                    "resume_pending": True,
                    "origin": {"chat_id": HUMAN},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "buzz" / "channel-cursors.json").write_text(
        json.dumps({"channels": {CHAT: {"last_ts": 10, "seen": ["aa" * 32]}, OTHER: {"last_ts": 3, "seen": []}}}),
        encoding="utf-8",
    )
    return root


def _journal(tmp_path: Path) -> Path:
    journal = tmp_path / "delegations"
    journal.mkdir()
    (journal / "dlg_pending.json").write_text(
        json.dumps(
            {
                "delegation_id": "dlg_" + "11" * 16,
                "state": "pending",
                "channel_id": CHAT,
                "origin_chat_id": HUMAN,
                "task": "keep this",
            }
        ),
        encoding="utf-8",
    )
    return journal


def test_dry_run_names_every_replayable_send_and_changes_nothing(tmp_path):
    home = _home(tmp_path)
    journal = _journal(tmp_path)
    before = (home / "state.db").read_bytes()
    report = plan_quarantine(home, chats=[CHAT], journal_dir=journal, loop_parent=PARENT)
    assert report["dry_run"] is True
    assert report["apply"] is False
    labels = {item["obligation_id"]: item["label"] for item in report["outbound"]}
    assert labels == {"startup": "startup", "error": "error", "response": "response"}
    assert report["outbound"][1]["replies_to_loop_parent"] or any(
        item["replies_to_loop_parent"] for item in report["outbound"]
    )
    assert report["preserved_outbound"] == 1
    assert report["inbound_sessions"][0]["chat_id"] == CHAT
    assert report["preserved_sessions"] == 1
    assert report["inbound_cursors"][0]["restart_replays_relay_history"] is True
    assert report["preserved_delegations"][0]["state"] == "pending"
    assert report["preserved_delegations"][0]["origin_chat_id"] == HUMAN
    assert (home / "state.db").read_bytes() == before
    assert not (home / "buzz" / "coordination-dispatch-hold.json").exists()


def test_apply_backs_up_and_leaves_other_work(tmp_path):
    home = _home(tmp_path)
    journal = _journal(tmp_path)
    journal_bytes = (journal / "dlg_pending.json").read_bytes()
    report = plan_quarantine(home, chats=[CHAT], journal_dir=journal, loop_parent=PARENT)
    backup = tmp_path / "backup"
    backup.mkdir()
    applied = apply_quarantine(home, report, backup_dir=backup / "copy")
    assert applied["apply"] is True
    assert (backup / "copy" / "state.db").is_file()
    assert (journal / "dlg_pending.json").read_bytes() == journal_bytes
    conn = sqlite3.connect(home / "state.db")
    states = dict(conn.execute("SELECT obligation_id, state FROM delivery_obligations"))
    conn.close()
    assert states["startup"] == "quarantined"
    assert states["error"] == "quarantined"
    assert states["response"] == "quarantined"
    assert states["delivered"] == "delivered"
    assert states["human"] == "pending"
    routing = json.loads(
        sqlite3.connect(home / "state.db")
        .execute("SELECT entry_json FROM gateway_routing WHERE session_key LIKE '%telegram%'")
        .fetchone()[0]
    )
    assert routing["resume_pending"] is True
    held = json.loads((home / "buzz" / "coordination-dispatch-hold.json").read_text(encoding="utf-8"))
    assert held["chats"] == [CHAT]
    mirror = json.loads((home / "sessions" / "sessions.json").read_text(encoding="utf-8"))
    assert mirror["agent:main:telegram:dm:" + HUMAN]["resume_pending"] is True
    assert mirror["agent:main:buzz:dm:" + CHAT]["resume_pending"] is False


def test_live_home_and_nested_backup_are_refused(tmp_path):
    home = _home(tmp_path)
    report = plan_quarantine(home, chats=[CHAT])
    with pytest.raises(QuarantineError):
        apply_quarantine(home, report, backup_dir=home / "backup")
    with pytest.raises(QuarantineError):
        plan_quarantine(Path("/home/hermes"), chats=[CHAT])
