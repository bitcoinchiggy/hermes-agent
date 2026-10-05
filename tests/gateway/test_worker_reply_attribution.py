"""Successive top-level DM turns keep their own reply parents.

A worker DM is one session. A second delegation that arrives while the first
turn is running is a queued follow-up: the running turn's body is sent, then
the follow-up runs with that session's history still in place. The outer final
send is bracketed against the event that opened the chain. Before this, that
send reused the opening event's reply anchor, so the follow-up's body was
published as a reply to the opening message.

Restart recovery is a different send. The obligation stores the reply parent
of the send that failed, separately from any thread root, and startup and
runtime redelivery pass that parent back. A parentless Buzz row is quarantined
only when this Hermes home's config.yaml names that chat as
``fleet.control_coordination_chat_id``. Control's human reports, other Buzz
DMs and channels, parented worker results, and other platforms are published.
A missing setting does not identify the coordination DM, so those rows are
published too. Recovery does not borrow a later delegation's event id.
The follow-up is still delivered. History is still passed through.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.delivery_ledger import RECOVERED_MARKER, recovered_reply_metadata
from gateway.platforms.base import SendResult, _reply_anchor_for_event
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

NEW_MARKER = "FLEET-ROUNDTRIP-20261002-03"
OLD_MARKER = "FLEET-PHASE1-ROUNDTRIP-20261002-01"
NEW_ID = "eb72a494a81767dfe8031b06ab0cc9d3bab6826c0ca269e6c58b09e5cfc74d9f"
OLD_ID = "c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00c0ffee00"
SESSION_KEY = "agent:main:buzz:dm:worker-dm"
HUMAN_CHAT = "111001"
OTHER_DM = "other-dm"
BASELINE_CHANNEL = "8e063ae2-a34d-5051-aada-5e431a81b737"


def _buzz_source() -> SessionSource:
    return SessionSource(
        platform=Platform("buzz"), chat_id="worker-dm", chat_type="dm", user_id="operator",
    )


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text)
    runner._persist_prompt_pins = AsyncMock()
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._intake_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    return GatewayRunner, runner


def _pinned_inputs(source):
    return MagicMock(side_effect=lambda *_args, **_kwargs: (None, source))


def _turn(source, *, event_message_id, inbound_message_id, history):
    return SimpleNamespace(
        source=source, session_id="sid", session_key=SESSION_KEY, run_generation=1,
        _interrupt_depth=0, history=history, _status_thread_metadata=None,
        context_prompt=None, result_holder=[None], mute_notification_reply=False,
        stream_consumer_holder=[None], persist_user_display_kind=None, reply_expected=None,
        event_message_id=event_message_id, inbound_message_id=inbound_message_id,
    )


async def _send_final(event, text, monkeypatch):
    """The adapter final-send seam, with the ledger switched off."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import PlatformConfig

    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda config=None: False)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
    adapter.gateway_runner = SimpleNamespace(_schedule_flood_redelivery=MagicMock())
    await adapter._send_final_text(event, SESSION_KEY, text, {"notify": True}, False, 0, lambda _r: None)
    return adapter._send_with_retry.await_args.kwargs["reply_to"]


@pytest.mark.asyncio
async def test_a_queued_older_task_keeps_its_parent_and_the_session_history(monkeypatch):
    """The reported order: the new delegation finishes first, then a queued older task.

    The 27-character new marker is the opening turn. The 34-character older marker
    is the follow-up. Both used to be replies to the new event. The follow-up now
    replies to its own event, and the model still sees the new turn in history.
    """
    GatewayRunner, runner = _runner()
    source = _buzz_source()
    runner._pinned_channel_inputs = _pinned_inputs(source)
    history = [
        {"role": "user", "content": "reply with exactly " + OLD_MARKER},
        {"role": "assistant", "content": NEW_MARKER},
    ]
    seen = {}

    async def _model(**kwargs):
        seen["history"] = kwargs["history"]
        seen["message"] = kwargs["message"]
        seen["event_message_id"] = kwargs["event_message_id"]
        return {"final_response": OLD_MARKER, "messages": list(kwargs["history"]) + [
            {"role": "user", "content": kwargs["message"]},
            {"role": "assistant", "content": OLD_MARKER},
        ]}

    runner._run_agent = _model
    pending = MessageEvent(
        text="reply with exactly " + OLD_MARKER, source=source, message_id=OLD_ID,
    )
    turn = _turn(source, event_message_id=NEW_ID, inbound_message_id=NEW_ID, history=history)
    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn, adapter=None, pending=pending.text, pending_event=pending,
        response={"final_response": NEW_MARKER, "messages": history},
        result={"interrupted": False, "final_response": NEW_MARKER, "messages": history},
        stream_task=None,
    )

    first = runner._deliver_queued_first_response.await_args
    assert first.kwargs["event_message_id"] == NEW_ID
    assert first.args[0] == NEW_MARKER
    assert seen["event_message_id"] == OLD_ID
    assert seen["message"] == pending.text
    assert any(NEW_MARKER in (row.get("content") or "") for row in seen["history"])
    assert merged["final_response"] == OLD_MARKER
    assert merged["queued_terminal_reply_anchor"] == OLD_ID

    opening = MessageEvent(text="reply with exactly " + NEW_MARKER, source=source, message_id=NEW_ID)
    runner._apply_queued_terminal_delivery(opening, merged)
    assert _reply_anchor_for_event(opening) == OLD_ID
    assert await _send_final(opening, merged["final_response"], monkeypatch) == OLD_ID


@pytest.mark.asyncio
async def test_pending_recovery_does_not_inherit_the_new_delegation(monkeypatch):
    """An internal restart recovery queued behind the new delegation has no message id.

    Its body is still sent. It is not attached to the new delegation, and the new
    turn remains in the history the recovery model sees.
    """
    GatewayRunner, runner = _runner()
    source = _buzz_source()
    runner._pinned_channel_inputs = _pinned_inputs(source)
    history = [{"role": "assistant", "content": NEW_MARKER}]
    seen = {}

    async def _model(**kwargs):
        seen["history"] = kwargs["history"]
        seen["message"] = kwargs["message"]
        return {"final_response": OLD_MARKER, "messages": kwargs["history"]}

    runner._run_agent = _model
    pending = MessageEvent(text="", source=source, internal=True)
    turn = _turn(source, event_message_id=NEW_ID, inbound_message_id=NEW_ID, history=history)
    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn, adapter=None, pending="", pending_event=pending,
        response={"final_response": NEW_MARKER, "messages": history},
        result={"interrupted": False, "final_response": NEW_MARKER, "messages": history},
        stream_task=None,
    )

    assert runner._deliver_queued_first_response.await_args.kwargs["event_message_id"] == NEW_ID
    assert any(NEW_MARKER in (row.get("content") or "") for row in seen["history"])
    assert merged["final_response"] == OLD_MARKER
    assert merged["queued_terminal_reply_anchor"] is None

    opening = MessageEvent(text="reply with exactly " + NEW_MARKER, source=source, message_id=NEW_ID)
    runner._apply_queued_terminal_delivery(opening, merged)
    assert _reply_anchor_for_event(opening) is None
    assert await _send_final(opening, OLD_MARKER, monkeypatch) is None


@pytest.mark.asyncio
async def test_a_new_delegation_queued_behind_recovery_keeps_its_own_parent(monkeypatch):
    """The other race: recovery is the running turn, and the new delegation is queued.

    Recovery's early send has no reply parent. The new marker then replies to the
    new event. The older marker stays in history.
    """
    GatewayRunner, runner = _runner()
    source = _buzz_source()
    runner._pinned_channel_inputs = _pinned_inputs(source)
    history = [{"role": "user", "content": "reply with exactly " + OLD_MARKER}]
    seen = {}

    async def _model(**kwargs):
        seen["history"] = kwargs["history"]
        return {"final_response": NEW_MARKER, "messages": kwargs["history"]}

    runner._run_agent = _model
    pending = MessageEvent(
        text="reply with exactly " + NEW_MARKER, source=source, message_id=NEW_ID,
    )
    turn = _turn(source, event_message_id=None, inbound_message_id=None, history=history)
    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn, adapter=None, pending=pending.text, pending_event=pending,
        response={"final_response": OLD_MARKER, "messages": history},
        result={"interrupted": False, "final_response": OLD_MARKER, "messages": history},
        stream_task=None,
    )

    first = runner._deliver_queued_first_response.await_args
    assert first.kwargs["event_message_id"] is None
    assert first.args[0] == OLD_MARKER
    assert any(OLD_MARKER in (row.get("content") or "") for row in seen["history"])
    assert merged["final_response"] == NEW_MARKER
    assert merged["queued_terminal_reply_anchor"] == NEW_ID

    opening = MessageEvent(text="", source=source, internal=True)
    runner._apply_queued_terminal_delivery(opening, merged)
    assert _reply_anchor_for_event(opening) == NEW_ID
    assert await _send_final(opening, NEW_MARKER, monkeypatch) == NEW_ID


@pytest.mark.asyncio
async def test_the_handler_applies_the_terminal_parent_before_the_outer_send(monkeypatch):
    """The production handler, not a copy of it, moves the outer send's parent."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "gateway.run_heartbeat_acceptance.heartbeat_owner_is_current", lambda *_a, **_k: True,
    )
    source = _buzz_source()
    runner = object.__new__(GatewayRunner)
    event = MessageEvent(text="reply with exactly " + NEW_MARKER, source=source, message_id=NEW_ID)
    prepared = GatewayRunner._PreparedTurn(
        history=[{"role": "assistant", "content": OLD_MARKER}],
        context_prompt="", message_text=event.text, persist_user_message=event.text,
        persist_user_timestamp=None, persist_user_display_kind=None,
        persistence_session_id="sid", persistence_owner="owner",
    )
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_resolve_session = AsyncMock(return_value=(source, SimpleNamespace(session_id="sid"), SESSION_KEY))
    runner._hmwa_prepare_turn = AsyncMock(return_value=(prepared, []))
    runner._pinned_channel_inputs = _pinned_inputs(source)
    runner._persist_prompt_pins = AsyncMock()
    runner._run_agent = AsyncMock(return_value={
        "final_response": OLD_MARKER,
        "queued_terminal_inbound_id": OLD_ID,
        "queued_terminal_reply_anchor": OLD_ID,
    })
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._is_session_run_current = MagicMock(return_value=False)
    runner._hmwa_discard_stale_result = MagicMock()
    runner._clear_session_env = MagicMock()

    await GatewayRunner._handle_message_with_agent(runner, event, source, SESSION_KEY, 1)

    assert runner._run_agent.await_args.kwargs["event_message_id"] == NEW_ID
    assert runner._run_agent.await_args.kwargs["history"][0]["content"] == OLD_MARKER
    assert event.ledger_message_id == OLD_ID
    assert _reply_anchor_for_event(event) == OLD_ID
    assert await _send_final(event, OLD_MARKER, monkeypatch) == OLD_ID


@pytest.mark.asyncio
async def test_recovered_delivery_keeps_its_body_and_does_not_borrow_a_later_event(monkeypatch, tmp_path):
    """A stored parent is replayed. Only the configured coordination DM is held.

    The 22:27:16 send had no stored parent. A later obligation's event id is
    not copied onto it. A Buzz row that recorded its own parent is redelivered
    to that parent. Control's human chat, another DM, a baseline channel, and
    a telegram row with no parent are still sent. The row text is not a route.
    """
    from gateway.run import GatewayRunner

    home = _isolate_ledger(monkeypatch, tmp_path)
    _write_coordination_chat(
        home._db_path().parent,
        "worker-dm",
        extra=(
            "buzz:\n"
            "  allowed_users:\n"
            "    - " + ("ab" * 32) + "\n"
            "  baseline_channels:\n"
            "    - " + BASELINE_CHANNEL + "\n"
        ),
    )

    assert recovered_reply_metadata(None) is None
    assert recovered_reply_metadata("topic-root") == {"thread_id": "topic-root"}
    assert recovered_reply_metadata(None, OLD_ID) == {"reply_to_message_id": OLD_ID}
    assert recovered_reply_metadata("topic-root", OLD_ID) == {
        "thread_id": "topic-root",
        "reply_to_message_id": OLD_ID,
    }
    assert NEW_ID not in str(recovered_reply_metadata(None))
    assert NEW_ID not in str(recovered_reply_metadata("topic-root", OLD_ID))

    sent = []

    class _Adapter:
        async def send(self, *, chat_id, content, reply_to=None, metadata=None):
            sent.append({
                "chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata,
            })
            return SendResult(success=True, message_id="recovered")

    runner = object.__new__(GatewayRunner)
    runner._obligation_adapter = AsyncMock(return_value=_Adapter())
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()
    monkeypatch.setattr("gateway.delivery_ledger.mark_delivered", MagicMock())
    monkeypatch.setattr("gateway.delivery_ledger.mark_failed", MagicMock())
    count = await GatewayRunner._redeliver_claimed_obligations(runner, [
        {
            "adopted": False,
            "obligation_id": "ob-legacy",
            "platform": "buzz",
            "chat_id": "worker-dm",
            "thread_id": None,
            "content": OLD_MARKER,
            "needs_marker": True,
            "attempts": 1,
        },
        {
            "adopted": False,
            "obligation_id": "ob-parent",
            "platform": "buzz",
            "chat_id": "worker-dm",
            "thread_id": "topic-root",
            "reply_to_message_id": OLD_ID,
            "content": NEW_MARKER,
            "needs_marker": True,
            "attempts": 1,
        },
        {
            "adopted": False,
            "obligation_id": "ob-telegram",
            "platform": "telegram",
            "chat_id": "4242",
            "thread_id": None,
            "content": "telegram-plain",
            "needs_marker": True,
            "attempts": 1,
        },
        {
            "adopted": False,
            "obligation_id": "ob-human",
            "platform": "buzz",
            "chat_id": HUMAN_CHAT,
            "thread_id": None,
            "content": "CONTROL-HUMAN-REPORT",
            "needs_marker": True,
            "attempts": 1,
        },
        {
            "adopted": False,
            "obligation_id": "ob-other-dm",
            "platform": "buzz",
            "chat_id": OTHER_DM,
            "thread_id": None,
            "content": "worker-dm is mentioned only in this body",
            "needs_marker": True,
            "attempts": 1,
        },
        {
            "adopted": False,
            "obligation_id": "ob-channel",
            "platform": "buzz",
            "chat_id": BASELINE_CHANNEL,
            "thread_id": None,
            "content": "channel-plain",
            "needs_marker": True,
            "attempts": 1,
        },
    ])

    assert count == 5
    by_body = {item["content"].split("\n\n")[-1]: item for item in sent}
    assert OLD_MARKER not in by_body
    parented = by_body[NEW_MARKER]
    telegram = by_body["telegram-plain"]
    assert telegram["reply_to"] is None
    assert telegram["metadata"] is None
    assert telegram["chat_id"] == "4242"
    human = by_body["CONTROL-HUMAN-REPORT"]
    assert human["chat_id"] == HUMAN_CHAT
    assert human["reply_to"] is None
    other = by_body["worker-dm is mentioned only in this body"]
    assert other["chat_id"] == OTHER_DM
    assert other["reply_to"] is None
    channel = by_body["channel-plain"]
    assert channel["chat_id"] == BASELINE_CHANNEL
    assert channel["reply_to"] is None
    from gateway.buzz_recovery_quarantine import load_quarantine

    quarantined = load_quarantine("ob-legacy")
    assert quarantined is not None
    assert quarantined["content"] == OLD_MARKER
    assert quarantined["published"] is False
    assert quarantined["reply_to_message_id"] is None
    assert quarantined["platform"] == "buzz"
    assert quarantined["chat_id"] == "worker-dm"
    assert "origin" not in quarantined
    assert load_quarantine("ob-parent") is None
    assert load_quarantine("ob-telegram") is None
    assert load_quarantine("ob-human") is None
    assert load_quarantine("ob-other-dm") is None
    assert load_quarantine("ob-channel") is None
    assert all(OLD_MARKER not in item["content"] for item in sent)
    assert parented["reply_to"] == OLD_ID
    assert parented["metadata"] == {"thread_id": "topic-root", "reply_to_message_id": OLD_ID}
    assert parented["metadata"]["thread_id"] != parented["reply_to"]
    assert NEW_ID not in str(parented["metadata"])
    assert NEW_MARKER in parented["content"]


@pytest.mark.asyncio
async def test_unanchored_buzz_publishes_when_the_coordination_chat_is_not_configured(
    monkeypatch, tmp_path,
):
    """No trusted coordination id means normal Buzz recovery, including the worker DM."""
    from gateway.run import GatewayRunner

    _isolate_ledger(monkeypatch, tmp_path)
    sent = []

    class _Adapter:
        async def send(self, *, chat_id, content, reply_to=None, metadata=None):
            sent.append({"chat_id": chat_id, "content": content, "reply_to": reply_to})
            return SendResult(success=True, message_id="recovered")

    runner = object.__new__(GatewayRunner)
    runner._obligation_adapter = AsyncMock(return_value=_Adapter())
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()
    monkeypatch.setattr("gateway.delivery_ledger.mark_delivered", MagicMock())
    monkeypatch.setattr("gateway.delivery_ledger.mark_failed", MagicMock())
    count = await GatewayRunner._redeliver_claimed_obligations(runner, [
        {
            "adopted": False,
            "obligation_id": "ob-legacy",
            "platform": "buzz",
            "chat_id": "worker-dm",
            "thread_id": None,
            "content": "LEGACY-NO-PARENT",
            "needs_marker": False,
            "attempts": 1,
        },
    ])
    assert count == 1
    assert sent[0]["chat_id"] == "worker-dm"
    assert sent[0]["content"] == "LEGACY-NO-PARENT"
    assert sent[0]["reply_to"] is None
    from gateway.buzz_recovery_quarantine import load_quarantine

    assert load_quarantine("ob-legacy") is None
    assert not (tmp_path / "hermes-home" / "buzz-recovery-quarantine").exists()


@pytest.mark.asyncio
async def test_buzz_send_uses_the_per_turn_reply_parent():
    """Buzz publishes ``--reply-to`` from the anchor the gateway just chose.

    A thread root still wins when one is stored, which is the live threaded
    send. A top-level recovery has no thread root, so the stored reply parent
    is the CLI anchor. A legacy recovery has neither.
    """
    from plugins.platforms.buzz.adapter import BuzzAdapter

    adapter = object.__new__(BuzzAdapter)
    adapter._reply_to_mode = "first"
    adapter._thread_roots = {}
    adapter._self_pubkey = "abc"
    adapter._channel_state = {}
    adapter._mention_pubkeys_for = AsyncMock(return_value=[])
    adapter._run_message_send = AsyncMock(return_value=(0, "", ""))
    adapter._send_result = lambda *_a, **_k: SendResult(success=True, message_id="sent")

    await adapter.send("worker-dm", NEW_MARKER, reply_to=NEW_ID, metadata={"notify": True})
    await adapter.send("worker-dm", OLD_MARKER, reply_to=OLD_ID, metadata={"notify": True})
    await adapter.send("worker-dm", RECOVERED_MARKER + OLD_MARKER, reply_to=None, metadata=None)
    await adapter.send(
        "worker-dm", RECOVERED_MARKER + OLD_MARKER, reply_to=OLD_ID,
        metadata={"reply_to_message_id": OLD_ID},
    )
    await adapter.send(
        "worker-dm", RECOVERED_MARKER + NEW_MARKER, reply_to=OLD_ID,
        metadata={"thread_id": "topic-root", "reply_to_message_id": OLD_ID},
    )

    calls = [call.args[0] for call in adapter._run_message_send.await_args_list]
    assert calls[0][calls[0].index("--reply-to") + 1] == NEW_ID
    assert calls[1][calls[1].index("--reply-to") + 1] == OLD_ID
    assert "--reply-to" not in calls[2]
    assert calls[3][calls[3].index("--reply-to") + 1] == OLD_ID
    assert calls[4][calls[4].index("--reply-to") + 1] == "topic-root"
    assert NEW_ID not in calls[3]
    assert NEW_ID not in calls[4]
    contents = [call.args[1] for call in adapter._run_message_send.await_args_list]
    assert contents[0] == NEW_MARKER
    assert contents[1] == OLD_MARKER
    assert contents[2] == RECOVERED_MARKER + OLD_MARKER


def _write_coordination_chat(home, chat_id: str, *, extra: str = "") -> None:
    (home / "config.yaml").write_text(
        "fleet:\n  control_coordination_chat_id: " + chat_id + "\n" + extra,
        encoding="utf-8",
    )


def _isolate_ledger(monkeypatch, tmp_path):
    from gateway import delivery_ledger as dl

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    return dl


def _failing_transport():
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="permanent refusal"))
    adapter.gateway_runner = SimpleNamespace()
    return adapter


def _redelivery_runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform("buzz"): adapter}
    runner._profile_adapters = {}
    store = SimpleNamespace()
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(_store=store, clear_resume_pending=AsyncMock())
    return runner


class _RecordingSend:
    def __init__(self):
        self.sent = []

    async def send(self, *, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({
            "chat_id": chat_id, "content": content, "reply_to": reply_to, "metadata": metadata,
        })
        return SendResult(success=True, message_id=f"recovered-{len(self.sent)}")


@pytest.mark.asyncio
async def test_runtime_redelivery_replays_the_stored_parent_and_the_thread_root(monkeypatch, tmp_path):
    """A failed live send is retried in-process with the parent it recorded."""
    dl = _isolate_ledger(monkeypatch, tmp_path)
    _write_coordination_chat(dl._db_path().parent, "worker-dm")
    transport = _failing_transport()
    top_level = _buzz_source()
    threaded = SessionSource(
        platform=Platform("buzz"), chat_id="worker-dm", chat_type="dm", user_id="operator",
        thread_id="topic-root",
    )
    await transport.send_final_ledgered(
        MessageEvent(text="reply with exactly " + OLD_MARKER, source=top_level, message_id=OLD_ID),
        SESSION_KEY, OLD_MARKER, {"notify": True}, reply_to=OLD_ID,
    )
    await transport.send_final_ledgered(
        MessageEvent(text="topic", source=threaded, message_id="topic-trigger"),
        SESSION_KEY + ":topic", "threaded-body", {"notify": True}, reply_to=NEW_ID,
    )
    await transport.send_final_ledgered(
        MessageEvent(text="legacy", source=top_level, message_id="legacy-id"),
        SESSION_KEY, "LEGACY-NO-PARENT", {"notify": True}, reply_to=None,
    )
    from gateway.delivery_ledger import compute_obligation_id, mark_failed, record_obligation

    for session, ref, chat, content in (
        (f"agent:main:buzz:dm:{HUMAN_CHAT}", "human-report", HUMAN_CHAT, "CONTROL-HUMAN-REPORT"),
        (f"agent:main:buzz:dm:{OTHER_DM}", "other-dm", OTHER_DM, "worker-dm is mentioned only in this body"),
        (f"agent:main:buzz:channel:{BASELINE_CHANNEL}", "channel", BASELINE_CHANNEL, "channel-plain"),
    ):
        obligation_id = compute_obligation_id(session, ref, content)
        record_obligation(
            obligation_id=obligation_id, session_key=session, platform="buzz",
            chat_id=chat, thread_id=None, content=content, reply_to_message_id=None,
        )
        mark_failed(obligation_id, "permanent refusal")
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET updated_at = updated_at - 31")
        stored = {
            row[0]: row[1]
            for row in conn.execute("SELECT content, reply_to_message_id FROM delivery_obligations")
        }
    assert stored[OLD_MARKER] == OLD_ID
    assert stored["threaded-body"] == NEW_ID
    assert stored["LEGACY-NO-PARENT"] is None

    recording = _RecordingSend()
    runner = _redelivery_runner(recording)
    from gateway.run import GatewayRunner

    count = await GatewayRunner._redeliver_failed_obligations_for_platform(runner, Platform("buzz"))

    assert count == 5
    by_body = {item["content"].split("\n\n")[-1]: item for item in recording.sent}
    assert "LEGACY-NO-PARENT" not in by_body
    assert by_body["CONTROL-HUMAN-REPORT"]["chat_id"] == HUMAN_CHAT
    assert by_body["CONTROL-HUMAN-REPORT"]["reply_to"] is None
    assert by_body["worker-dm is mentioned only in this body"]["chat_id"] == OTHER_DM
    assert by_body["channel-plain"]["chat_id"] == BASELINE_CHANNEL
    assert by_body[OLD_MARKER]["reply_to"] == OLD_ID
    assert by_body[OLD_MARKER]["metadata"] == {"reply_to_message_id": OLD_ID}
    assert "thread_id" not in by_body[OLD_MARKER]["metadata"]
    threaded_send = by_body["threaded-body"]
    assert threaded_send["reply_to"] == NEW_ID
    assert threaded_send["metadata"]["thread_id"] == "topic-root"
    assert threaded_send["metadata"]["reply_to_message_id"] == NEW_ID
    assert threaded_send["metadata"]["thread_id"] != threaded_send["reply_to"]
    assert all(item["content"].startswith(dl.RECONNECTED_MARKER) for item in recording.sent)
    from gateway.buzz_recovery_quarantine import load_quarantine

    quarantined = load_quarantine(dl.compute_obligation_id(SESSION_KEY, "legacy-id", "LEGACY-NO-PARENT"))
    assert quarantined is not None
    assert quarantined["content"] == "LEGACY-NO-PARENT"
    assert quarantined["published"] is False
    assert quarantined["reply_to_message_id"] is None
    with dl._connect() as conn:
        legacy_row = conn.execute(
            "SELECT content, state, reply_to_message_id FROM delivery_obligations WHERE content = ?",
            ("LEGACY-NO-PARENT",),
        ).fetchone()
    assert legacy_row == ("LEGACY-NO-PARENT", "quarantined", None)


def _control_root() -> "object":
    import os
    from pathlib import Path

    configured = os.environ.get("HERMES_FLEET_CONTROL_ROOT", "")
    path = Path(configured) if configured else Path("/agent/repos/hermes-fleet-control")
    return path if path.is_dir() else None


@pytest.mark.asyncio
async def test_recovered_delegations_keep_their_parents_through_restart_and_handoff(
    monkeypatch, tmp_path,
):
    """Two delegations, one queued follow-up, failed sends, then a worker restart.

    Operator and Control use different state directories. The worker config
    names ``worker-dm`` as the coordination chat. Only the parentless row
    addressed there is quarantined. Control's report to the human Buzz chat,
    another DM, a baseline channel, a parented worker result, and telegram
    are published. Control's own home has no coordination setting.
    Each parented recovered body is handed to the human session stored for
    its own parent. Control sends no worker acknowledgement.
    """
    import hashlib
    import os
    import subprocess
    import sys

    control = _control_root()
    if control is None:
        pytest.skip("HERMES_FLEET_CONTROL_ROOT is required")
    if str(control) not in sys.path:
        sys.path.insert(0, str(control))

    from gateway import delivery_ledger as dl

    worker_home = tmp_path / "vm1035-operator"
    control_home = tmp_path / "vm1005-control"
    worker_home.mkdir()
    control_home.mkdir()
    _write_coordination_chat(
        worker_home,
        "worker-dm",
        extra="buzz:\n  baseline_channels:\n    - " + BASELINE_CHANNEL + "\n",
    )
    monkeypatch.setattr(dl, "_db_path", lambda: worker_home / "state.db")
    journal = control_home / "delegations"
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        check=True, capture_output=True,
    )
    python = venv / "bin" / "python"
    real = os.path.realpath(sys.executable)
    if os.path.realpath(python) != real:
        python.unlink()
        python.symlink_to(real)
    monkeypatch.setenv("HERMES_FLEET_CONTROL_ROOT", str(control))
    monkeypatch.setenv("HERMES_FLEET_CONTROL_INTEGRATION", "1")
    monkeypatch.setenv("FLEET_DELEGATION_JOURNAL_DIR", str(journal))
    monkeypatch.setenv("FLEET_CONTROL_PYTHON", str(python))
    monkeypatch.setenv("HERMES_HOME", str(worker_home))

    from fleet_control.delegation_reply import plan_delegation_reply
    from fleet_control.journal import ensure_journal_dir, fresh_record, read_record, update_record, write_record

    worker_hex = "cd" * 32
    new_chat, old_chat = "111001", "111002"
    new_task = "reply with exactly " + NEW_MARKER
    old_task = "reply with exactly " + OLD_MARKER
    ensure_journal_dir(journal)

    def accept(delegation_id: str, event_id: str, chat_id: str, message_id: str, task: str) -> None:
        record = fresh_record(
            delegation_id=delegation_id,
            worker="operator",
            worker_public_key_hex=worker_hex,
            task_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
            state="accepted",
            task=task,
            origin_platform="telegram",
            origin_chat_id=chat_id,
            origin_session_key=f"agent:main:telegram:dm:{chat_id}",
            origin_thread_id="",
            origin_message_id=message_id,
            origin_chat_type="dm",
            origin_scope_id="",
            origin_user_id="42",
        )
        write_record(journal, update_record(record, channel_id="worker-dm", event_id=event_id))

    accept("dlg_" + "11" * 16, NEW_ID, new_chat, "555", new_task)
    accept("dlg_" + "22" * 16, OLD_ID, old_chat, "556", old_task)

    GatewayRunner, runner = _runner()
    del runner._deliver_queued_first_response
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._pop_post_delivery_callback = lambda *_a, **_k: None
    source = _buzz_source()
    runner._pinned_channel_inputs = _pinned_inputs(source)
    transport = _failing_transport()
    history = [{"role": "user", "content": old_task}, {"role": "assistant", "content": NEW_MARKER}]
    seen = {}

    async def _model(**kwargs):
        seen["history"] = kwargs["history"]
        seen["message"] = kwargs["message"]
        seen["event_message_id"] = kwargs["event_message_id"]
        return {"final_response": OLD_MARKER, "messages": list(kwargs["history"])}

    runner._run_agent = _model
    pending = MessageEvent(text=old_task, source=source, message_id=OLD_ID)
    turn = _turn(source, event_message_id=NEW_ID, inbound_message_id=NEW_ID, history=history)
    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn, adapter=transport, pending=pending.text, pending_event=pending,
        response={"final_response": NEW_MARKER, "messages": history},
        result={"interrupted": False, "final_response": NEW_MARKER, "messages": history},
        stream_task=None,
    )
    assert seen["event_message_id"] == OLD_ID
    assert seen["message"] == old_task
    assert any(NEW_MARKER in (row.get("content") or "") for row in seen["history"])
    assert merged["final_response"] == OLD_MARKER
    assert merged["queued_terminal_reply_anchor"] == OLD_ID

    opening = MessageEvent(text=new_task, source=source, message_id=NEW_ID)
    runner._apply_queued_terminal_delivery(opening, merged)
    await transport._send_final_text(opening, SESSION_KEY, merged["final_response"], {"notify": True}, False, 0, lambda _r: None)

    from gateway.delivery_ledger import compute_obligation_id, mark_failed, record_obligation

    record_obligation(
        obligation_id=compute_obligation_id(SESSION_KEY, "legacy-inbound", "LEGACY-NO-PARENT"),
        session_key=SESSION_KEY, platform="buzz", chat_id="worker-dm", thread_id=None,
        content="LEGACY-NO-PARENT", reply_to_message_id=None,
    )
    mark_failed(
        compute_obligation_id(SESSION_KEY, "legacy-inbound", "LEGACY-NO-PARENT"),
        "permanent refusal",
    )
    preserved = (
        (f"agent:main:buzz:dm:{HUMAN_CHAT}", "human-report", "buzz", HUMAN_CHAT, "CONTROL-HUMAN-REPORT"),
        (f"agent:main:buzz:dm:{OTHER_DM}", "other-dm", "buzz", OTHER_DM, "worker-dm is mentioned only in this body"),
        (f"agent:main:buzz:channel:{BASELINE_CHANNEL}", "channel", "buzz", BASELINE_CHANNEL, "channel-plain"),
        ("agent:main:telegram:dm:4242", "telegram", "telegram", "4242", "telegram-plain"),
    )
    for session, ref, platform, chat, content in preserved:
        obligation_id = compute_obligation_id(session, ref, content)
        record_obligation(
            obligation_id=obligation_id, session_key=session, platform=platform,
            chat_id=chat, thread_id=None, content=content, reply_to_message_id=None,
        )
        mark_failed(obligation_id, "permanent refusal")
    with dl._connect() as conn:
        rows = conn.execute(
            "SELECT content, reply_to_message_id, thread_id, state FROM delivery_obligations"
        ).fetchall()
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
        )
    stored = {row[0]: row for row in rows}
    assert stored[NEW_MARKER][1] == NEW_ID
    assert stored[OLD_MARKER][1] == OLD_ID
    assert stored["LEGACY-NO-PARENT"][1] is None
    assert all(row[2] is None and row[3] == "failed" for row in rows)

    recording = _RecordingSend()
    restarted = _redelivery_runner(recording)
    restarted.adapters[Platform.TELEGRAM] = recording
    count = await GatewayRunner._redeliver_pending_obligations(restarted)
    assert count == 6
    published = "\n".join(item["content"] for item in recording.sent)
    assert "LEGACY-NO-PARENT" not in published
    by_parent = {item["reply_to"]: item for item in recording.sent if item["reply_to"]}
    assert set(by_parent) == {NEW_ID, OLD_ID}
    bare = {item["content"].split("\n\n")[-1]: item for item in recording.sent if not item["reply_to"]}
    assert bare["CONTROL-HUMAN-REPORT"]["chat_id"] == HUMAN_CHAT
    assert bare["worker-dm is mentioned only in this body"]["chat_id"] == OTHER_DM
    assert bare["channel-plain"]["chat_id"] == BASELINE_CHANNEL
    assert bare["telegram-plain"]["chat_id"] == "4242"
    assert all(item["reply_to"] is None for item in bare.values())
    assert by_parent[NEW_ID]["content"].startswith(dl.RECOVERED_MARKER)
    assert by_parent[NEW_ID]["content"].endswith(NEW_MARKER)
    assert OLD_MARKER not in by_parent[NEW_ID]["content"]
    assert by_parent[NEW_ID]["metadata"] == {"reply_to_message_id": NEW_ID}
    assert by_parent[OLD_ID]["content"].endswith(OLD_MARKER)
    assert NEW_MARKER not in by_parent[OLD_ID]["content"]
    assert by_parent[OLD_ID]["metadata"] == {"reply_to_message_id": OLD_ID}
    legacy_id = compute_obligation_id(SESSION_KEY, "legacy-inbound", "LEGACY-NO-PARENT")
    from gateway.buzz_recovery_quarantine import load_quarantine

    quarantined = load_quarantine(legacy_id)
    assert quarantined is not None
    assert quarantined["content"] == "LEGACY-NO-PARENT"
    assert quarantined["published"] is False
    assert quarantined["reply_to_message_id"] is None
    assert quarantined["chat_id"] == "worker-dm"
    for session, ref, _platform, _chat, content in preserved:
        assert load_quarantine(compute_obligation_id(session, ref, content)) is None
    assert (worker_home / "buzz-recovery-quarantine").is_dir()
    assert not (control_home / "buzz-recovery-quarantine").exists()
    assert not (control_home / "state.db").exists()
    with dl._connect() as conn:
        legacy_row = conn.execute(
            "SELECT content, state, reply_to_message_id FROM delivery_obligations WHERE obligation_id = ?",
            (legacy_id,),
        ).fetchone()
    assert legacy_row == ("LEGACY-NO-PARENT", "quarantined", None)
    monkeypatch.setenv("HERMES_HOME", str(control_home))

    new_plan = plan_delegation_reply(
        read_record(journal, "dlg_" + "11" * 16),
        reply_to_message_id=NEW_ID, reply_to_text=new_task,
        inbound_text=by_parent[NEW_ID]["content"], sender_public_key_hex=worker_hex,
    )
    old_plan = plan_delegation_reply(
        read_record(journal, "dlg_" + "22" * 16),
        reply_to_message_id=OLD_ID, reply_to_text=old_task,
        inbound_text=by_parent[OLD_ID]["content"], sender_public_key_hex=worker_hex,
    )
    assert new_plan.acknowledge_worker is False
    assert old_plan.acknowledge_worker is False
    assert new_plan.report_to.chat_id == new_chat
    assert old_plan.report_to.chat_id == old_chat
    assert new_plan.wake_text is not None and NEW_MARKER in new_plan.wake_text
    assert OLD_MARKER not in new_plan.wake_text
    assert old_plan.wake_text is not None and OLD_MARKER in old_plan.wake_text
    assert NEW_MARKER not in old_plan.wake_text
    from gateway.config import PlatformConfig
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageType
    from plugins.platforms.buzz.adapter import BuzzAdapter

    class _Human(BasePlatformAdapter):
        def __init__(self) -> None:
            super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
            self.sent = []

        @property
        def name(self) -> str:
            return "telegram"

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append({"chat_id": str(chat_id), "content": content, "reply_to": reply_to})
            return SendResult(success=True, message_id="human-1")

        async def get_chat_info(self, chat_id):
            return {"id": chat_id, "type": "dm"}

    human = _Human()
    wakes = []

    async def evaluate(event):
        wakes.append(event)
        return f"report for {event.source.chat_id}"

    human.set_message_handler(evaluate)
    worker_turns = []

    async def worker_model(event):
        worker_turns.append(event.text)
        return "automatic worker acknowledgement"

    worker = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "wss://buzz.example/"}))
    worker._self_pubkey = "aa" * 32
    worker.sent = []
    worker.reactions = []

    async def _worker_send(*args, **kwargs):
        worker.sent.append((args, kwargs))
        return SendResult(success=True, message_id="worker-ack")

    async def _worker_react(*args, **kwargs):
        worker.reactions.append((args, kwargs))
        return True

    worker.send = _worker_send
    worker.send_reaction = _worker_react
    worker.set_message_handler(worker_model)
    worker._resolve_user_name = AsyncMock(return_value="operator")
    worker._localize_inbound_media = AsyncMock(
        side_effect=lambda text, message_id, **kwargs: (text, [], [], MessageType.TEXT))
    gateway = SimpleNamespace(
        adapters={Platform.TELEGRAM: human, Platform("buzz"): worker},
        _profile_adapters={},
        _background_tasks=set(),
        _profile_name_for_source=lambda source, adapter_profile=None: None,
    )
    worker.gateway_runner = gateway
    human.gateway_runner = gateway

    async def settle() -> None:
        tasks = [
            task
            for adapter in (human, worker)
            for task in list(adapter._session_tasks.values())
            if hasattr(task, "done")
        ]
        if tasks:
            await __import__("asyncio").gather(*tasks)

    async def deliver(event_id: str, parent: str | None, text: str) -> None:
        tags = [["e", parent, "", "reply"]] if parent else []
        await worker._handle_event(
            "worker-dm",
            {"seen": {}, "last_ts": 0, "chat_type": "dm", "event_meta": {}},
            {
                "id": event_id, "pubkey": worker_hex, "kind": 9, "content": text,
                "created_at": 1_700_000_000, "tags": tags,
            },
        )
        await settle()

    await deliver("ab" * 32, NEW_ID, by_parent[NEW_ID]["content"])
    await deliver("ba" * 32, OLD_ID, by_parent[OLD_ID]["content"])
    assert worker_turns == []
    assert worker.sent == []
    assert worker.reactions == []
    assert "LEGACY-NO-PARENT" not in "\n".join(item["content"] for item in human.sent)
    assert {item.source.chat_id for item in wakes} == {new_chat, old_chat}
    by_chat = {item.source.chat_id: item.text for item in wakes}
    assert NEW_MARKER in by_chat[new_chat]
    assert OLD_MARKER not in by_chat[new_chat]
    assert OLD_MARKER in by_chat[old_chat]
    assert NEW_MARKER not in by_chat[old_chat]
    assert [item["chat_id"] for item in human.sent] == [item.source.chat_id for item in wakes]
    assert all(item["content"].startswith("report for ") for item in human.sent)
    assert read_record(journal, "dlg_" + "11" * 16)["reply_deliveries"][0]["state"] == "completed"
    assert read_record(journal, "dlg_" + "22" * 16)["reply_deliveries"][0]["state"] == "completed"

    monkeypatch.setattr(dl, "_db_path", lambda: control_home / "state.db")
    from gateway.buzz_recovery_quarantine import coordination_chat_id

    assert coordination_chat_id() is None
    report = "CONTROL-ORIGIN-REPORT"
    report_id = compute_obligation_id(f"agent:main:buzz:dm:{new_chat}", "control-report", report)
    record_obligation(
        obligation_id=report_id, session_key=f"agent:main:buzz:dm:{new_chat}",
        platform="buzz", chat_id=new_chat, thread_id=None, content=report,
        reply_to_message_id=None,
    )
    mark_failed(report_id, "permanent refusal")
    with dl._connect() as conn:
        conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1")
    control_recording = _RecordingSend()
    control_runner = _redelivery_runner(control_recording)
    assert await GatewayRunner._redeliver_pending_obligations(control_runner) == 1
    assert control_recording.sent[0]["chat_id"] == new_chat
    assert control_recording.sent[0]["content"].endswith(report)
    assert control_recording.sent[0]["reply_to"] is None
    assert load_quarantine(report_id) is None
    assert not (control_home / "buzz-recovery-quarantine").exists()
    assert not (control_home / "config.yaml").exists()
