"""Fleet coordination channel: one addressed worker, everyone else silent.

Buzz delivers a private-channel message to every member. These tests bind
the Buzz adapter's production disposition, not a copy of the predicate.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

import pytest
from unittest.mock import AsyncMock

from gateway.fleet_coordination import (
    CoordinationSettings,
    coordination_disposition,
    evaluation_body,
    should_post_evaluation,
    worker_should_execute,
)
from gateway.fleet_delegation import _post_thread_evaluation
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter

CHANNEL = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
LEGACY = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
OTHER = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
ASSIGNER = "a" * 64
WORKER_A = "b" * 64
WORKER_B = "c" * 64
DELEGATION = "dlg_" + "ab" * 16
ASSIGNMENT = EVENT = "d" * 64
RESULT = "e" * 64


def _settings() -> CoordinationSettings:
    return CoordinationSettings(
        channel_id=CHANNEL,
        legacy_dm_id=LEGACY,
        assigner_pubkey=ASSIGNER,
    )


def _event(
    content: str,
    *,
    author: str = ASSIGNER,
    mentioned: list[str] | None = None,
    kind: int = 9,
    reply_to: str | None = None,
    event_id: str = "1" * 64,
) -> dict:
    tags: list[list[str]] = [["h", CHANNEL]]
    for pubkey in mentioned or []:
        tags.append(["p", pubkey])
    if reply_to:
        tags.append(["e", reply_to, "", "reply"])
    return {
        "id": event_id,
        "pubkey": author,
        "content": content,
        "created_at": 1000,
        "kind": kind,
        "tags": tags,
    }


def _assignment(mentioned: list[str] | None = None, **kwargs) -> dict:
    return _event(
        f"[fleet-delegation {DELEGATION}]\nInspect the queue",
        mentioned=mentioned if mentioned is not None else [WORKER_A],
        **kwargs,
    )


def _adapter(pubkey: str):
    from gateway.config import PlatformConfig

    adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay"}))
    adapter._self_pubkey = pubkey
    adapter._dispatch_message = AsyncMock()
    adapter._resolve_user_name = AsyncMock(return_value="worker")
    return adapter


def _state(chat_type: str = "group") -> dict:
    return {"seen": OrderedDict(), "last_ts": 0, "chat_type": chat_type, "event_meta": OrderedDict()}


def _use(monkeypatch, settings: CoordinationSettings | None = None) -> CoordinationSettings:
    loaded = settings if settings is not None else _settings()
    monkeypatch.setattr(
        "gateway.fleet_coordination.load_coordination_settings",
        lambda path=None: loaded,
    )
    return loaded


def test_only_the_addressed_worker_executes_in_a_shared_channel():
    settings = _settings()
    assignment = _assignment()
    assert worker_should_execute(assignment, WORKER_A, settings.assigner_pubkey) is True
    assert worker_should_execute(assignment, WORKER_B, settings.assigner_pubkey) is False
    assert (
        coordination_disposition(
            channel_id=CHANNEL,
            event=assignment,
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=settings,
        )
        == "assignment"
    )
    assert (
        coordination_disposition(
            channel_id=CHANNEL,
            event=assignment,
            self_pubkey=WORKER_B,
            control_integration=False,
            settings=settings,
        )
        == "suppress"
    )


def test_membership_does_not_authorize_a_forged_or_shared_sender():
    settings = _settings()
    forged = _assignment(author=WORKER_B)
    shared = _assignment(mentioned=[WORKER_A, WORKER_B])
    assert worker_should_execute(forged, WORKER_A, settings.assigner_pubkey) is False
    assert worker_should_execute(shared, WORKER_A, settings.assigner_pubkey) is False
    assert worker_should_execute(_assignment(), WORKER_A, None) is False


@pytest.mark.parametrize(
    "event",
    [
        _event("gateway is online"),
        _event("ack"),
        _event("HTTP 402"),
        _event("!shutdown", mentioned=[WORKER_A]),
        _event("[fleet-evaluation " + DELEGATION + "]\nlooks right", reply_to=ASSIGNMENT, mentioned=[WORKER_A]),
        _event("joined", kind=40099, mentioned=[WORKER_A]),
        _event("👍", kind=7, mentioned=[WORKER_A]),
        _assignment(reply_to=ASSIGNMENT),
    ],
)
def test_notices_errors_and_evaluations_are_not_assignments(event):
    assert worker_should_execute(event, WORKER_A, ASSIGNER) is False
    assert (
        coordination_disposition(
            channel_id=CHANNEL,
            event=event,
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=_settings(),
        )
        == "suppress"
    )


def test_replay_of_the_same_notice_stays_silent():
    notice = _event("HTTP 402", event_id=RESULT)
    for _ in range(2):
        assert (
            coordination_disposition(
                channel_id=CHANNEL,
                event=notice,
                self_pubkey=WORKER_A,
                control_integration=False,
                settings=_settings(),
            )
            == "suppress"
        )


def test_other_chats_stay_on_the_normal_path():
    assert (
        coordination_disposition(
            channel_id=OTHER,
            event=_event("hello", mentioned=[WORKER_A]),
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=_settings(),
        )
        == "normal"
    )


def test_evaluation_copy_is_not_an_assignment_and_drops_secrets():
    body = evaluation_body(DELEGATION, "The queue is empty.")
    assert body is not None and body.startswith("[fleet-evaluation ")
    assert worker_should_execute(
        _event(body, reply_to=RESULT, mentioned=[WORKER_A]),
        WORKER_A,
        ASSIGNER,
    ) is False
    redacted = evaluation_body(DELEGATION, "nsec1secretdonotpost")
    assert redacted is not None and "nsec1" not in redacted
    assert "The queue is empty." not in (redacted or "")


def test_human_report_copy_targets_the_channel_thread_only(monkeypatch):
    _use(monkeypatch)
    plan = {
        "_evaluation_ready": True,
        "delegation_id": DELEGATION,
        "inbound_event_id": RESULT,
        "_evaluation_text": "The worker answered the stored task.",
    }
    assert should_post_evaluation(CHANNEL, plan, _settings()) is True
    assert should_post_evaluation(LEGACY, plan, _settings()) is False
    assert should_post_evaluation(CHANNEL, {**plan, "_evaluation_ready": False}, _settings()) is False

    sent: dict = {}

    class Adapter:
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            sent["chat_id"] = chat_id
            sent["content"] = content
            sent["reply_to"] = reply_to

    asyncio.run(_post_thread_evaluation(Adapter(), CHANNEL, plan))
    assert sent["chat_id"] == CHANNEL
    assert sent["reply_to"] == RESULT
    assert sent["content"].startswith("[fleet-evaluation ")
    assert "The worker answered the stored task." in sent["content"]
    sent.clear()
    asyncio.run(_post_thread_evaluation(Adapter(), LEGACY, plan))
    assert sent == {}


def test_adapter_dispatches_only_the_addressed_assignment(monkeypatch):
    _use(monkeypatch)
    monkeypatch.setattr("gateway.fleet_delegation.integration_enabled", lambda: False)
    addressed = _adapter(WORKER_A)
    other = _adapter(WORKER_B)
    assignment = _assignment(event_id=ASSIGNMENT)
    asyncio.run(addressed._handle_event(CHANNEL, _state(), assignment))
    asyncio.run(other._handle_event(CHANNEL, _state(), assignment))
    addressed._dispatch_message.assert_awaited()
    other._dispatch_message.assert_not_awaited()
    text = addressed._dispatch_message.await_args.kwargs["text"]
    assert text.startswith(f"[fleet-delegation {DELEGATION}]")
    assert addressed._dispatch_message.await_args.kwargs["message_id"] == ASSIGNMENT


def test_adapter_suppresses_chatter_errors_and_replay(monkeypatch):
    _use(monkeypatch)
    monkeypatch.setattr("gateway.fleet_delegation.integration_enabled", lambda: False)
    adapter = _adapter(WORKER_A)
    state = _state("dm")
    for event_id, content in (("2" * 64, "home channel notice"), ("3" * 64, "HTTP 402"), ("4" * 64, "ack")):
        asyncio.run(adapter._handle_event(LEGACY, state, _event(content, event_id=event_id)))
        asyncio.run(adapter._handle_event(LEGACY, state, _event(content, event_id=event_id)))
    adapter._dispatch_message.assert_not_awaited()


def test_adapter_control_does_not_answer_unrelated_channel_traffic(monkeypatch):
    _use(monkeypatch)
    monkeypatch.setattr("gateway.fleet_delegation.integration_enabled", lambda: True)
    handoff = AsyncMock(return_value=False)
    monkeypatch.setattr("gateway.fleet_delegation.maybe_handoff_worker_reply", handoff)
    adapter = _adapter(ASSIGNER)
    asyncio.run(adapter._handle_event(CHANNEL, _state(), _event("HTTP 402", author=WORKER_A)))
    handoff.assert_not_awaited()
    adapter._dispatch_message.assert_not_awaited()
    reply = _event("worker result", author=WORKER_A, reply_to=ASSIGNMENT, event_id=RESULT)
    asyncio.run(adapter._handle_event(CHANNEL, _state(), reply))
    handoff.assert_awaited()
    adapter._dispatch_message.assert_not_awaited()
    assert handoff.await_args.kwargs["reply_to_message_id"] == ASSIGNMENT
    assert handoff.await_args.kwargs["sender_public_key_hex"] == WORKER_A


def test_unconfigured_channel_still_dispatches_an_addressed_message(monkeypatch):
    _use(monkeypatch, CoordinationSettings())
    adapter = _adapter(WORKER_A)
    asyncio.run(
        adapter._handle_event(
            OTHER,
            _state(),
            _event("hello", author=ASSIGNER, mentioned=[WORKER_A], event_id="5" * 64),
        )
    )
    adapter._dispatch_message.assert_awaited()


def test_degraded_config_uses_chat_identity_and_not_message_text():
    settings = CoordinationSettings(status="unreadable", remembered=(LEGACY,))
    human_text = _event(
        f"[fleet-delegation {DELEGATION}]\nthis is a human conversation",
        mentioned=[WORKER_A],
    )
    assert (
        coordination_disposition(
            channel_id=OTHER,
            event=human_text,
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=settings,
        )
        == "normal"
    )
    assert (
        coordination_disposition(
            channel_id=LEGACY,
            event=_assignment(),
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=settings,
        )
        == "suppress"
    )
    assert (
        coordination_disposition(
            channel_id=LEGACY,
            event=_event("HTTP 402"),
            self_pubkey=WORKER_A,
            control_integration=False,
            settings=settings,
        )
        == "suppress"
    )


def test_degraded_control_correlates_a_reply_and_leaves_other_chats(monkeypatch):
    _use(monkeypatch, CoordinationSettings(status="malformed", remembered=(CHANNEL,)))
    monkeypatch.setattr("gateway.fleet_delegation.integration_enabled", lambda: True)
    handoff = AsyncMock(return_value=False)
    monkeypatch.setattr("gateway.fleet_delegation.maybe_handoff_worker_reply", handoff)
    adapter = _adapter(ASSIGNER)
    asyncio.run(adapter._handle_event(CHANNEL, _state(), _event("HTTP 402", author=WORKER_A, event_id="6" * 64)))
    handoff.assert_not_awaited()
    adapter._dispatch_message.assert_not_awaited()
    reply = _event("worker result", author=WORKER_A, reply_to=ASSIGNMENT, event_id=RESULT)
    asyncio.run(adapter._handle_event(CHANNEL, _state(), reply))
    handoff.assert_awaited()
    adapter._dispatch_message.assert_not_awaited()
    asyncio.run(
        adapter._handle_event(
            OTHER,
            _state("dm"),
            _event(
                f"[fleet-delegation {DELEGATION}]\nhello",
                author="f" * 64,
                mentioned=[WORKER_A],
                event_id="7" * 64,
            ),
        )
    )
    adapter._dispatch_message.assert_awaited()


def test_unreadable_config_remembers_the_last_ready_destination(tmp_path, monkeypatch):
    from gateway.fleet_coordination import load_coordination_settings

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("gateway.fleet_coordination._home", lambda: home)
    (home / "config.yaml").write_text(
        "fleet:\n"
        f"  coordination_channel_id: {CHANNEL}\n"
        f"  control_assigner_pubkey: {ASSIGNER}\n",
        encoding="utf-8",
    )
    ready = load_coordination_settings()
    assert ready.status == "ready"
    assert ready.channel_id == CHANNEL
    (home / "config.yaml").write_text("fleet: [\n", encoding="utf-8")
    broken = load_coordination_settings()
    assert broken.status == "unreadable"
    assert CHANNEL in broken.known_chats()
    (home / "config.yaml").write_text(
        "fleet:\n"
        "  coordination_channel_id: not-a-uuid\n"
        f"  control_coordination_chat_id: {LEGACY}\n",
        encoding="utf-8",
    )
    malformed = load_coordination_settings()
    assert malformed.status == "malformed"
    assert CHANNEL in malformed.known_chats()
    assert LEGACY in malformed.known_chats()
    assert malformed.assigner_pubkey is None


def test_journal_channel_is_known_without_reading_the_task(tmp_path, monkeypatch):
    from gateway.fleet_coordination import journal_chat_ids

    journal = tmp_path / "delegations"
    journal.mkdir()
    (journal / f"{DELEGATION}.json").write_text(
        '{"delegation_id": "%s", "channel_id": "%s", "task": "HTTP 402", "state": "accepted"}'
        % (DELEGATION, LEGACY),
        encoding="utf-8",
    )
    assert journal_chat_ids(journal) == frozenset({LEGACY})
    monkeypatch.setattr("gateway.fleet_coordination.journal_chat_ids", lambda directory=None: frozenset({LEGACY}))
    settings = CoordinationSettings(status="unreadable")
    assert (
        coordination_disposition(
            channel_id=LEGACY,
            event=_event("ack"),
            self_pubkey=ASSIGNER,
            control_integration=True,
            settings=settings,
        )
        == "control"
    )
    assert (
        coordination_disposition(
            channel_id=OTHER,
            event=_event("ack"),
            self_pubkey=ASSIGNER,
            control_integration=True,
            settings=settings,
        )
        == "normal"
    )


def test_hold_is_reread_and_blocks_recovered_sends(monkeypatch, tmp_path):
    """Restart reads the hold file again. Human-chat obligations still send."""
    import json

    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.fleet_coordination import held_chat_ids

    home = tmp_path / "home"
    (home / "buzz").mkdir(parents=True)
    db = home / "state.db"
    monkeypatch.setattr("gateway.delivery_ledger._db_path", lambda: db)
    monkeypatch.setattr("gateway.fleet_coordination._home", lambda: home)
    hold = {"chats": [LEGACY], "reason": "legacy-coordination-replay"}
    (home / "buzz" / "coordination-dispatch-hold.json").write_text(json.dumps(hold), encoding="utf-8")
    assert held_chat_ids() == frozenset({LEGACY})
    assert held_chat_ids() == frozenset({LEGACY})

    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            state TEXT,
            last_error TEXT,
            updated_at REAL
        )"""
    )
    rows = [
        ("startup", "gateway is online", None, False),
        ("error", "HTTP 402", "ab" * 32, False),
        ("response", "the worker finished", None, False),
        ("recovered", "kept result", "cd" * 32, True),
        ("human", "report to the operator", None, True),
    ]
    for obligation_id, _content, _parent, _marker in rows:
        conn.execute(
            "INSERT INTO delivery_obligations VALUES (?, 'pending', NULL, 0)",
            (obligation_id,),
        )
    conn.commit()
    conn.close()

    sent: list[str] = []

    class Adapter:
        async def send(self, *, chat_id, content, reply_to=None, metadata=None):
            sent.append(str(chat_id))
            return SendResult(success=True, message_id="sent")

    runner = object.__new__(GatewayRunner)
    runner._obligation_adapter = AsyncMock(return_value=Adapter())
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()
    claimed = []
    for obligation_id, content, parent, marker in rows:
        chat = OTHER if obligation_id == "human" else LEGACY
        claimed.append(
            {
                "adopted": False,
                "obligation_id": obligation_id,
                "platform": "buzz",
                "chat_id": chat,
                "thread_id": None,
                "reply_to_message_id": parent,
                "content": content,
                "needs_marker": marker,
                "attempts": 1,
            }
        )
    count = asyncio.run(GatewayRunner._redeliver_claimed_obligations(runner, claimed))
    assert count == 1
    assert sent == [OTHER]
    stored = {
        row[0]: row[1:]
        for row in sqlite3.connect(db).execute(
            "SELECT obligation_id, state, last_error FROM delivery_obligations"
        )
    }
    assert stored["human"][0] == "delivered"
    for obligation_id in ("startup", "error", "response", "recovered"):
        assert stored[obligation_id][0] == "quarantined"
        assert stored[obligation_id][1] == "quarantined: coordination dispatch hold; not published"
    count_again = asyncio.run(GatewayRunner._redeliver_claimed_obligations(runner, claimed))
    assert count_again == 1
    assert sent == [OTHER, OTHER]
