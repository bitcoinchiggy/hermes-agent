"""BOM-prefixed coordination files still parse, and bad bytes still fail closed.

Windows tooling prefixes files with a UTF-8 BOM. These tests write that
prefix and call the production readers. A BOM does not relax validation:
malformed JSON or YAML adds no destination, hold, journal row, or
quarantine record.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

CHANNEL = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEGACY = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
OTHER = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
ASSIGNER = "a" * 64
DELEGATION = "dlg_" + "ab" * 16


def _bom(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))


def test_bom_configuration_stays_ready_until_the_yaml_is_invalid(tmp_path, monkeypatch):
    from gateway.buzz_recovery_quarantine import coordination_chat_id
    from gateway.fleet_coordination import load_coordination_settings

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("gateway.fleet_coordination._home", lambda: home)
    monkeypatch.setattr("gateway.delivery_ledger._db_path", lambda: home / "state.db")
    _bom(
        home / "config.yaml",
        "fleet:\n"
        f"  coordination_channel_id: {CHANNEL}\n"
        f"  control_coordination_chat_id: {LEGACY}\n"
        f"  control_assigner_pubkey: {ASSIGNER}\n",
    )
    ready = load_coordination_settings()
    assert ready.status == "ready"
    assert ready.channel_id == CHANNEL
    assert ready.legacy_dm_id == LEGACY
    assert ready.assigner_pubkey == ASSIGNER
    assert coordination_chat_id() == LEGACY

    known = json.loads((home / "fleet-coordination-known.json").read_text(encoding="utf-8-sig"))
    _bom(home / "fleet-coordination-known.json", json.dumps(known))
    _bom(home / "config.yaml", "fleet: [\n")
    broken = load_coordination_settings()
    assert broken.status == "unreadable"
    assert broken.known_chats() == frozenset({CHANNEL, LEGACY})
    assert broken.assigner_pubkey is None
    assert coordination_chat_id() is None

    _bom(home / "fleet-coordination-known.json", "{")
    forgotten = load_coordination_settings()
    assert forgotten.status == "unreadable"
    assert forgotten.known_chats() == frozenset()

    _bom(
        home / "config.yaml",
        "fleet:\n  control_coordination_chat_id: 12\n",
    )
    assert coordination_chat_id() is None


def test_bom_hold_and_journal_identify_only_valid_records(tmp_path, monkeypatch):
    from gateway.fleet_coordination import held_chat_ids, journal_chat_ids

    home = tmp_path / "home"
    monkeypatch.setattr("gateway.fleet_coordination._home", lambda: home)
    _bom(
        home / "buzz" / "coordination-dispatch-hold.json",
        json.dumps({"chats": [LEGACY, "bad\nid"], "reason": "legacy-coordination-replay"}),
    )
    assert held_chat_ids() == frozenset({LEGACY})
    _bom(home / "buzz" / "coordination-dispatch-hold.json", "{")
    assert held_chat_ids() == frozenset()
    _bom(
        home / "buzz" / "coordination-dispatch-hold.json",
        json.dumps({"chats": LEGACY}),
    )
    assert held_chat_ids() == frozenset()

    journal = tmp_path / "delegations"
    _bom(
        journal / f"{DELEGATION}.json",
        json.dumps(
            {
                "delegation_id": DELEGATION,
                "state": "pending",
                "channel_id": CHANNEL,
                "origin_chat_id": OTHER,
                "task": "keep this",
            }
        ),
    )
    _bom(journal / "broken.json", "{")
    assert journal_chat_ids(journal) == frozenset({CHANNEL})


def test_bom_quarantine_records_keep_validation(tmp_path, monkeypatch):
    from gateway.buzz_recovery_quarantine import load_quarantine
    from gateway.fleet_loop_quarantine import apply_quarantine, plan_quarantine

    home = tmp_path / "home"
    monkeypatch.setattr("gateway.delivery_ledger._db_path", lambda: home / "state.db")
    folder = home / "buzz-recovery-quarantine"
    good = "ob-bom"
    rejected = "ob-published"
    broken = "ob-broken"

    def record_path(obligation_id: str) -> Path:
        digest = hashlib.sha256(obligation_id.encode("utf-8")).hexdigest()
        return folder / f"{digest}.json"

    _bom(
        record_path(good),
        json.dumps(
            {
                "obligation_id": good,
                "platform": "buzz",
                "chat_id": LEGACY,
                "thread_id": None,
                "content": "worker result",
                "reply_to_message_id": None,
                "published": False,
            }
        ),
    )
    _bom(
        record_path(rejected),
        json.dumps(
            {
                "obligation_id": rejected,
                "platform": "buzz",
                "chat_id": LEGACY,
                "content": "already out",
                "reply_to_message_id": None,
                "published": True,
            }
        ),
    )
    _bom(record_path(broken), "{")
    loaded = load_quarantine(good)
    assert loaded is not None
    assert loaded["chat_id"] == LEGACY
    assert loaded["content"] == "worker result"
    assert loaded["published"] is False
    assert load_quarantine(rejected) is None
    assert load_quarantine(broken) is None

    (home / "sessions").mkdir(parents=True)
    (home / "buzz").mkdir()
    _bom(
        home / "sessions" / "sessions.json",
        json.dumps(
            {
                "_README": "mirror",
                "agent:main:buzz:dm:" + LEGACY: {
                    "resume_pending": True,
                    "origin": {"chat_id": LEGACY},
                },
                "agent:main:buzz:dm:" + OTHER: {
                    "resume_pending": True,
                    "origin": {"chat_id": OTHER},
                },
            }
        ),
    )
    _bom(
        home / "buzz" / "channel-cursors.json",
        json.dumps({"channels": {LEGACY: {"last_ts": 10, "seen": ["aa" * 32]}, OTHER: {"last_ts": 3, "seen": []}}}),
    )
    journal = tmp_path / "delegations"
    _bom(
        journal / f"{DELEGATION}.json",
        json.dumps(
            {
                "delegation_id": DELEGATION,
                "state": "pending",
                "channel_id": CHANNEL,
                "origin_chat_id": OTHER,
            }
        ),
    )
    _bom(journal / "broken.json", "{")
    report = plan_quarantine(home, chats=[LEGACY], journal_dir=journal)
    assert [item["chat_id"] for item in report["inbound_sessions"]] == [LEGACY]
    assert report["preserved_sessions"] == 1
    assert report["inbound_cursors"][0]["chat_id"] == LEGACY
    assert report["inbound_cursors"][0]["restart_replays_relay_history"] is True
    assert [item["delegation_id"] for item in report["preserved_delegations"]] == [DELEGATION]
    backup = tmp_path / "backup"
    backup.mkdir()
    apply_quarantine(home, report, backup_dir=backup / "copy")
    mirror = json.loads((home / "sessions" / "sessions.json").read_text(encoding="utf-8-sig"))
    assert mirror["agent:main:buzz:dm:" + LEGACY]["resume_pending"] is False
    assert mirror["agent:main:buzz:dm:" + LEGACY]["resume_reason"] == "coordination-quarantine"
    assert mirror["agent:main:buzz:dm:" + OTHER]["resume_pending"] is True
    assert (journal / f"{DELEGATION}.json").read_bytes().startswith(b"\xef\xbb\xbf")


def test_bom_unavailable_hold_is_retried_and_garbage_is_not(tmp_path, monkeypatch):
    from gateway.fleet_delegation import _retry_unavailable_holds

    journal = tmp_path / "journal"
    holds = journal / "unavailable-holds"
    monkeypatch.setenv("FLEET_DELEGATION_JOURNAL_DIR", str(journal))
    event_id = "ab" * 32
    payload = {
        "inbound_event_id": event_id,
        "channel_id": CHANNEL,
        "inbound_text": "worker result",
    }
    _bom(holds / f"{event_id}.json", json.dumps(payload))
    _bom(holds / "zzzz-broken.json", "{")
    seen: list[dict] = []

    def intake(body: dict) -> dict:
        seen.append(body)
        return {"status": "unavailable"}

    monkeypatch.setattr("gateway.fleet_delegation._run_intake", intake)
    asyncio.run(_retry_unavailable_holds(object()))
    assert seen == [payload]
    assert (holds / f"{event_id}.json").is_file()
    assert (holds / "zzzz-broken.json").is_file()
