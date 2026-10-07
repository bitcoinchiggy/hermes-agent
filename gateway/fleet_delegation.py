"""Control Fleet delegation: trusted origin on delegate_worker, and the inbound handoff.

The MCP stdio child is long-lived. ``run_coroutine_threadsafe`` copies the MCP
loop's context, not the gateway turn's, so ``get_session_env`` inside
``_call_tool_racing_stdio_death`` cannot see the human conversation.
``fleet_delegate_meta`` reads the turn's ContextVars in the tool handler,
before that hop, and only when this process is Control's Fleet integration.

A worker reply is a result. Its text is not a route. Intake runs only for
this integration. A positive ``not_delegation_reply`` keeps the normal path.
An unavailable intake holds a reply-parent message in the profile journal
directory: no model dispatch, no worker reaction, and no worker error.
Gateway startup runs recovery once, then a bounded background retry until
shutdown. A human send that succeeds, or this handoff's own delivery-ledger
obligation, is not woken again. Another turn's success does not complete
this handoff. A model failure or a shutdown before that obligation exists
leaves the reply pending. Task completion is not delivery.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import secrets
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

INTEGRATION_ENV = "HERMES_FLEET_CONTROL_INTEGRATION"
MCP_SERVER_ENV = "HERMES_FLEET_MCP_SERVER"
CONTROL_ROOT_ENV = "HERMES_FLEET_CONTROL_ROOT"
_JOURNAL_DIR_ENV = "FLEET_DELEGATION_JOURNAL_DIR"
_PROFILE_ENV = "FLEET_CONTROL_HERMES_PROFILE"
_PROFILES_ROOT_ENV = "FLEET_CONTROL_PROFILES_ROOT"
_RECOVERY_INTERVAL_ENV = "HERMES_FLEET_DELEGATION_RECOVERY_INTERVAL"
_DEFAULT_PROFILES_ROOT = "/home/hermes/.hermes/profiles"
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
_INTAKE_TIMEOUT_SECONDS = 20.0
_DEFAULT_RECOVERY_INTERVAL = 30.0
_MIN_RECOVERY_INTERVAL = 0.05
_MAX_RECOVERY_INTERVAL = 300.0
_CHAT_TYPES = frozenset({"dm", "group", "channel", "thread"})
_GATEWAY_OWNER = secrets.token_hex(16)
_IN_RECOVERY = contextvars.ContextVar("fleet_delegation_recovery", default=False)
_MAX_HOLDS = 64
_recovery_active = False
_LEDGER_OWNED_STATES = frozenset({"pending", "attempting", "failed", "delivered"})


def integration_enabled() -> bool:
    """True only for the Control profile that owns the Fleet journal."""
    return os.environ.get(INTEGRATION_ENV) == "1"


def fleet_delegate_meta(server_name: str, tool_name: str) -> Optional[dict]:
    """Per-call ``_meta`` for Fleet ``delegate_worker``, or None.

    None means the caller must not pass ``meta``. Other servers, other tools,
    and a process that is not the Control integration stay on the normal MCP path.
    Tool arguments are not consulted.
    """
    if not integration_enabled():
        return None
    expected = os.environ.get(MCP_SERVER_ENV, "")
    if not expected or server_name != expected or tool_name != "delegate_worker":
        return None
    origin = _bound_origin()
    if origin is None:
        return None
    return {"hermes.fleet.origin": origin}


async def maybe_handoff_worker_reply(
    adapter: Any,
    *,
    channel_id: str,
    sender_public_key_hex: str,
    inbound_text: str,
    inbound_event_id: str,
    reply_to_message_id: Optional[str],
    reply_to_text: Optional[str],
    chat_type: str = "dm",
    created_at: int = 0,
) -> bool:
    """Return True when this event must not be dispatched on the worker DM.

    False is a positive unrelated message, a top-level message, or a gateway
    that is not Control's Fleet integration. Helper failure holds a reply
    instead of dispatching it.
    """
    if not integration_enabled():
        return False
    if not _IN_RECOVERY.get() and _begin_recovery():
        token = _IN_RECOVERY.set(True)
        try:
            await retry_fleet_delegation_recovery(adapter)
        finally:
            _IN_RECOVERY.reset(token)
            _end_recovery()
    if not reply_to_message_id:
        return False
    payload = _observation(
        channel_id=channel_id,
        sender_public_key_hex=sender_public_key_hex,
        inbound_text=inbound_text,
        inbound_event_id=inbound_event_id,
        reply_to_message_id=reply_to_message_id,
        reply_to_text=reply_to_text,
        chat_type=chat_type,
        created_at=created_at,
    )
    result = await asyncio.to_thread(_run_intake, payload)
    status = result.get("status")
    if status == "unrelated":
        return False
    if status != "correlated":
        _persist_unavailable_hold(payload)
        logger.warning("fleet delegation intake unavailable; holding the reply for recovery")
        return True
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    applied = await _apply_correlated(adapter, plan)
    if applied:
        await _post_thread_evaluation(adapter, channel_id, plan)
    return applied


async def retry_fleet_delegation_recovery(adapter: Any) -> None:
    """Retry held observations and unfinished human reports. Does not message the worker."""
    await _retry_unavailable_holds(adapter)
    await _recover_unfinished(adapter)


def start_fleet_delegation_recovery(runner: Any) -> None:
    """Start one recovery loop after adapters connect.

    No-op unless this process is Control's Fleet integration. A second call
    while the loop is running does not start another task.
    """
    if not integration_enabled():
        return
    existing = getattr(runner, "_fleet_delegation_recovery_task", None)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(_recovery_loop(runner), name="fleet_delegation_recovery")
    runner._fleet_delegation_recovery_task = task
    background = getattr(runner, "_background_tasks", None)
    if isinstance(background, set):
        background.add(task)
        task.add_done_callback(background.discard)


async def stop_fleet_delegation_recovery(runner: Any) -> None:
    """Cancel the recovery loop and wait until that task has finished."""
    task = getattr(runner, "_fleet_delegation_recovery_task", None)
    runner._fleet_delegation_recovery_task = None
    if task is None:
        return
    background = getattr(runner, "_background_tasks", None)
    if isinstance(background, set):
        background.discard(task)
    if task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return


def _begin_recovery() -> bool:
    """Claim the single in-flight recovery pass. Synchronous, so ticks cannot overlap."""
    global _recovery_active
    if _recovery_active:
        return False
    _recovery_active = True
    return True


def _end_recovery() -> None:
    global _recovery_active
    _recovery_active = False


def _recovery_interval() -> float:
    raw = os.environ.get(_RECOVERY_INTERVAL_ENV, "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_RECOVERY_INTERVAL
    except ValueError:
        value = _DEFAULT_RECOVERY_INTERVAL
    if value < _MIN_RECOVERY_INTERVAL:
        return _MIN_RECOVERY_INTERVAL
    if value > _MAX_RECOVERY_INTERVAL:
        return _MAX_RECOVERY_INTERVAL
    return value


async def _recovery_loop(runner: Any) -> None:
    """One immediate pass, then a bounded sleep. Cancellation ends the loop."""
    while True:
        await _run_recovery_once(runner)
        await asyncio.sleep(_recovery_interval())


async def _run_recovery_once(runner: Any) -> None:
    if not _begin_recovery():
        return
    token = _IN_RECOVERY.set(True)
    try:
        adapter = _buzz_adapter(runner)
        if adapter is not None:
            await retry_fleet_delegation_recovery(adapter)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("fleet delegation recovery pass failed", exc_info=True)
    finally:
        _IN_RECOVERY.reset(token)
        _end_recovery()


def _buzz_adapter(runner: Any) -> Any:
    """The Buzz adapter this gateway connected, including a secondary profile."""
    mappings = [getattr(runner, "adapters", None) or {}]
    for mapping in (getattr(runner, "_profile_adapters", None) or {}).values():
        if isinstance(mapping, dict):
            mappings.append(mapping)
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key, adapter in mapping.items():
            if getattr(key, "value", None) == "buzz" or type(adapter).__name__ == "BuzzAdapter":
                return adapter
    return None


def _bound_origin() -> Optional[dict]:
    """Snapshot the turn ContextVars. Never fall back to ``os.environ``."""
    from gateway.session_context import (
        _SESSION_CHAT_ID,
        _SESSION_CHAT_TYPE,
        _SESSION_KEY,
        _SESSION_MESSAGE_ID,
        _SESSION_PLATFORM,
        _SESSION_PROFILE,
        _SESSION_SCOPE_ID,
        _SESSION_THREAD_ID,
        _SESSION_USER_ID,
        _UNSET,
    )

    def bound(var: Any) -> Optional[str]:
        value = var.get()
        if value is _UNSET or not isinstance(value, str):
            return None
        return value

    platform = bound(_SESSION_PLATFORM)
    chat_id = bound(_SESSION_CHAT_ID)
    session_key = bound(_SESSION_KEY)
    chat_type = bound(_SESSION_CHAT_TYPE)
    if not platform or not chat_id or not session_key or chat_type not in _CHAT_TYPES:
        return None
    thread_id = bound(_SESSION_THREAD_ID)
    message_id = bound(_SESSION_MESSAGE_ID)
    scope_id = bound(_SESSION_SCOPE_ID)
    user_id = bound(_SESSION_USER_ID)
    origin = {
        "platform": platform,
        "chat_id": chat_id,
        "session_key": session_key,
        "thread_id": thread_id or "",
        "message_id": message_id or "",
        "chat_type": chat_type,
        "scope_id": scope_id or "",
        "user_id": user_id or "",
    }
    if not _session_key_matches(origin, bound(_SESSION_PROFILE) or ""):
        return None
    return origin


def _session_key_matches(origin: dict, profile: str) -> bool:
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    try:
        platform = Platform(origin["platform"])
    except ValueError:
        return False
    source = SessionSource(
        platform=platform,
        chat_id=origin["chat_id"],
        chat_type=origin["chat_type"],
        thread_id=origin["thread_id"] or None,
        message_id=origin["message_id"] or None,
        scope_id=origin["scope_id"] or None,
        user_id=origin["user_id"] or None,
    )
    derived = build_session_key(source, profile=profile or None)
    return derived == origin["session_key"]


def _run_intake(payload: dict) -> dict:
    """Run ``fleet-delegation-intake``.

    ``status`` is ``unrelated`` only when intake positively says this parent
    is not a delegation. Any failure is ``unavailable``.
    """
    root = os.environ.get(CONTROL_ROOT_ENV, "")
    if not root or not os.path.isabs(root):
        logger.warning("fleet delegation intake skipped: %s is not an absolute path", CONTROL_ROOT_ENV)
        return {"status": "unavailable"}
    binary = os.path.join(root, "fleet-delegation-intake")
    if not os.path.isfile(binary):
        logger.warning("fleet delegation intake skipped: helper is missing")
        return {"status": "unavailable"}
    allowed = {
        "reply_to_message_id",
        "sender_public_key_hex",
        "inbound_text",
        "inbound_event_id",
        "reply_to_text",
        "action",
        "claim_id",
        "delegation_id",
        "gateway_owner",
        "gateway_pid",
    }
    body = {key: value for key, value in payload.items() if key in allowed}
    body.setdefault("gateway_owner", _GATEWAY_OWNER)
    body.setdefault("gateway_pid", os.getpid())
    try:
        from tools.environments.local import build_subprocess_env

        # Scrub credentials and, when a routed home is active, drop the launch
        # profile's dotenv before the helper reads the journal. A builder
        # failure is the same as a missing helper: no dispatch.
        child_env = build_subprocess_env(strip_launch_profile=True)
    except Exception:
        logger.warning("fleet delegation intake environment was not built", exc_info=True)
        return {"status": "unavailable"}
    try:
        proc = subprocess.run(
            [binary],
            input=json.dumps(body).encode("utf-8"),
            capture_output=True,
            timeout=_INTAKE_TIMEOUT_SECONDS,
            env=child_env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("fleet delegation intake failed", exc_info=True)
        return {"status": "unavailable"}
    if proc.returncode != 0 or not proc.stdout:
        logger.warning("fleet delegation intake rejected the observation")
        return {"status": "unavailable"}
    try:
        plan = json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        logger.warning("fleet delegation intake returned unusable output")
        return {"status": "unavailable"}
    return _classify_intake(plan)


def _classify_intake(plan: object) -> dict:
    if not isinstance(plan, dict):
        return {"status": "unavailable"}
    kind = plan.get("kind")
    if kind == "not_delegation_reply" and plan.get("reason") is None:
        return {"status": "unrelated", "plan": plan}
    if kind == "delegation_reply" and plan.get("suppress_worker_delivery") is True:
        return {"status": "correlated", "plan": plan}
    if kind == "recovery" and isinstance(plan.get("plans"), list):
        return {"status": "recovery", "plan": plan}
    if kind == "delivery_recorded":
        return {"status": "recorded", "plan": plan}
    return {"status": "unavailable"}


def _observation(**fields: object) -> dict:
    payload = {
        "reply_to_message_id": fields["reply_to_message_id"],
        "sender_public_key_hex": fields["sender_public_key_hex"],
        "inbound_text": fields["inbound_text"],
        "inbound_event_id": fields["inbound_event_id"],
        "channel_id": fields["channel_id"],
        "chat_type": fields["chat_type"],
        "created_at": fields["created_at"],
    }
    if fields.get("reply_to_text") is not None:
        payload["reply_to_text"] = fields["reply_to_text"]
    return payload


def _journal_base() -> Optional[Path]:
    """Same directory the helper journal uses. Never the Control checkout.

    ``FLEET_DELEGATION_JOURNAL_DIR`` wins when it is absolute. Otherwise the
    path is ``<profiles root>/<profile>/fleet-delegations``. The profiles root
    defaults to ``/home/hermes/.hermes/profiles``. ``HERMES_FLEET_CONTROL_ROOT``
    is only where the intake binary lives.
    """
    journal = os.environ.get(_JOURNAL_DIR_ENV, "").strip()
    if journal:
        if os.path.isabs(journal):
            return Path(journal)
        logger.warning("fleet delegation journal override is not absolute")
        return None
    profile = os.environ.get(_PROFILE_ENV, "").strip()
    if not _PROFILE_RE.fullmatch(profile):
        logger.warning("fleet delegation profile is unusable; hold was not stored")
        return None
    root = os.environ.get(_PROFILES_ROOT_ENV, "").strip() or _DEFAULT_PROFILES_ROOT
    if not os.path.isabs(root):
        logger.warning("fleet delegation profiles root is not absolute")
        return None
    return Path(root) / profile / "fleet-delegations"


def _hold_directory() -> Optional[Path]:
    base = _journal_base()
    if base is None:
        return None
    path = base / "unavailable-holds"
    if path.is_symlink() or base.is_symlink():
        return None
    try:
        if not base.exists():
            base.mkdir(mode=0o700)
            os.chmod(base, 0o700)
        if not base.is_dir():
            return None
        path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:
        logger.warning("fleet delegation could not store an unavailable-intake hold", exc_info=True)
        return None
    return path


def _persist_unavailable_hold(payload: dict) -> None:
    folder = _hold_directory()
    event_id = payload.get("inbound_event_id")
    if folder is None or not isinstance(event_id, str) or len(event_id) != 64 or any(char not in "0123456789abcdefABCDEF" for char in event_id):
        logger.warning("fleet delegation hold was not stored; recovery needs the event again")
        return
    existing = [path for path in folder.glob("*.json") if path.is_file() and not path.is_symlink()]
    target = folder / f"{event_id}.json"
    if target not in existing and len(existing) >= _MAX_HOLDS:
        logger.warning("fleet delegation hold cap reached; this reply stays undispatched until redelivery")
        return
    temp = folder / f".{event_id}.{secrets.token_hex(4)}.tmp"
    try:
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, target)
        os.chmod(target, 0o600)
    except OSError:
        logger.warning("fleet delegation hold write failed", exc_info=True)


async def _retry_unavailable_holds(adapter: Any) -> None:
    folder = _hold_directory()
    if folder is None:
        return
    for path in sorted(folder.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        result = await asyncio.to_thread(_run_intake, payload)
        status = result.get("status")
        if status == "unavailable":
            continue
        try:
            path.unlink()
        except OSError:
            logger.warning("fleet delegation hold could not be cleared", exc_info=True)
            continue
        if status == "unrelated":
            await _dispatch_held(adapter, payload)
        elif status == "correlated":
            await _apply_correlated(adapter, result.get("plan") or {})


async def _recover_unfinished(adapter: Any) -> None:
    result = await asyncio.to_thread(_run_intake, {"action": "recover"})
    if result.get("status") != "recovery":
        return
    plans = (result.get("plan") or {}).get("plans") or []
    for plan in plans:
        if isinstance(plan, dict):
            await _apply_correlated(adapter, plan)


async def _apply_correlated(adapter: Any, plan: dict) -> bool:
    """Suppress the worker DM. Wake the human only for a new delivery claim."""
    if plan.get("kind") != "delegation_reply" or plan.get("suppress_worker_delivery") is not True:
        return False
    delivery = plan.get("delivery")
    wake = plan.get("wake_text")
    if delivery in {"completed", "in_flight"} or not isinstance(wake, str) or not wake:
        return True
    keep = False
    try:
        keep = await _resume_human_session(adapter, plan)
    except asyncio.CancelledError:
        await _record_claim_during_shutdown(plan, _ledger_owns_report(plan))
        raise
    except Exception:
        logger.warning("fleet delegation handoff could not resume the human session", exc_info=True)
        keep = False
    await _record_claim(plan, keep)
    return True


async def _record_claim(plan: dict, keep: bool) -> None:
    """Persist the claim. ``keep`` means the model must not run again.

    ``keep`` covers a successful human send and this handoff's own ledger
    obligation. It does not mean the human has the message yet.
    """
    claim_id = plan.get("claim_id")
    delegation_id = plan.get("delegation_id")
    event_id = plan.get("inbound_event_id")
    if not (isinstance(claim_id, str) and isinstance(delegation_id, str) and isinstance(event_id, str)):
        return
    action = "complete" if keep else "release"
    await asyncio.to_thread(
        _run_intake,
        {
            "action": action,
            "delegation_id": delegation_id,
            "inbound_event_id": event_id,
            "claim_id": claim_id,
        },
    )


async def _record_claim_during_shutdown(plan: dict, keep: bool) -> None:
    """Write the claim while this task is already cancelled, then stay cancelled."""
    current = asyncio.current_task()
    if current is not None:
        while current.cancelling():
            current.uncancel()
    try:
        await _record_claim(plan, keep)
    finally:
        if current is not None:
            current.cancel()


async def _dispatch_held(adapter: Any, payload: dict) -> None:
    text = payload.get("inbound_text")
    chat_id = payload.get("channel_id")
    user_id = payload.get("sender_public_key_hex")
    message_id = payload.get("inbound_event_id")
    if not all(isinstance(item, str) and item for item in (text, chat_id, user_id, message_id)):
        return
    created = payload.get("created_at")
    await adapter._dispatch_message(
        text=text,
        chat_id=chat_id,
        chat_type=payload.get("chat_type") if payload.get("chat_type") in _CHAT_TYPES else "dm",
        user_id=user_id,
        user_name="",
        message_id=message_id,
        created_at=created if isinstance(created, int) else 0,
        reply_to_message_id=payload.get("reply_to_message_id"),
        reply_to_text=payload.get("reply_to_text") if isinstance(payload.get("reply_to_text"), str) else None,
    )


async def _resume_human_session(worker_adapter: Any, plan: dict) -> bool:
    """Start one turn on the stored origin adapter. Worker text stays inside ``wake_text``.

    True means the model must not run again: the human send succeeded, or this
    handoff's own ledger obligation exists. False leaves the claim pending.
    Finishing the adapter session task is not itself delivery. Completion is
    observed on this event, not by replacing the adapter callback.
    """
    report = plan.get("report_to")
    wake = plan.get("wake_text")
    if not isinstance(report, dict) or not isinstance(wake, str) or not wake:
        return False
    origin_adapter = _origin_adapter(worker_adapter, report.get("platform"))
    if origin_adapter is None:
        logger.warning("fleet delegation handoff has no adapter for %s", report.get("platform"))
        return False
    source = _source_for(origin_adapter, report)
    if source is None:
        return False
    derived = origin_adapter._source_session_key(source)
    if derived != report.get("session_key"):
        logger.warning("fleet delegation handoff session key does not match the origin adapter")
        return False
    from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome

    message_id = report.get("message_id") if isinstance(report.get("message_id"), str) else ""
    event = MessageEvent(
        text=wake,
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        reply_to_message_id=message_id or None,
        metadata={"gateway_session_key": derived, "notification_category": "result"},
    )
    observed: dict[str, Any] = {}

    async def _watch(watched: MessageEvent, outcome: ProcessingOutcome) -> None:
        if watched is not event:
            return
        observed["outcome"] = outcome

    plan["_fleet_event"] = event
    event._on_processing_complete = _watch
    try:
        await origin_adapter.handle_message(event)
        if not getattr(event, "_gateway_accepted", False):
            return False
        session_key = str((event.metadata or {}).get("gateway_session_key") or "")
        task = getattr(origin_adapter, "_session_tasks", {}).get(session_key)
        if task is not None and not task.done():
            await task
    finally:
        if getattr(event, "_on_processing_complete", None) is _watch:
            event._on_processing_complete = None
    if observed.get("outcome") == ProcessingOutcome.SUCCESS or _ledger_owns_report(plan):
        text = getattr(event, "_streamed_final_response", "")
        plan["_evaluation_text"] = text if isinstance(text, str) else ""
        plan["_evaluation_ready"] = True
        return True
    logger.warning("fleet delegation handoff produced no report; the reply stays pending")
    return False


async def _post_thread_evaluation(adapter: Any, channel_id: str, plan: dict) -> None:
    """Copy one accepted evaluation into the coordination thread.

    Failure here does not release the human-report claim and does not
    start another model turn. The legacy coordination DM is not a target.
    """
    from gateway.fleet_coordination import evaluation_body, should_post_evaluation

    if not should_post_evaluation(channel_id, plan):
        return
    body = evaluation_body(str(plan.get("delegation_id") or ""), plan.get("_evaluation_text"))
    if body is None:
        return
    try:
        await adapter.send(channel_id, body, reply_to=plan.get("inbound_event_id"))
    except Exception:
        logger.warning("fleet coordination evaluation was not posted", exc_info=True)


def _ledger_owns_report(plan: dict) -> bool:
    """True when this handoff's own report obligation is in the delivery ledger."""
    event = plan.get("_fleet_event")
    obligation_id = getattr(event, "_delivery_obligation_id", None)
    if not isinstance(obligation_id, str) or not obligation_id:
        return False
    try:
        from gateway.delivery_ledger import _connect

        with _connect() as conn:
            row = conn.execute(
                "SELECT state FROM delivery_obligations WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
    except Exception:
        logger.warning("fleet delegation could not read the delivery ledger", exc_info=True)
        return False
    return row is not None and row[0] in _LEDGER_OWNED_STATES


def _origin_adapter(worker_adapter: Any, platform_name: object) -> Any:
    if not isinstance(platform_name, str) or not platform_name:
        return None
    runner = getattr(worker_adapter, "gateway_runner", None)
    if runner is None:
        return None
    from gateway.config import Platform

    try:
        platform = Platform(platform_name)
    except ValueError:
        return None
    adapters = getattr(runner, "adapters", None) or {}
    found = adapters.get(platform)
    if found is not None:
        return found
    for mapping in (getattr(runner, "_profile_adapters", None) or {}).values():
        if isinstance(mapping, dict) and platform in mapping:
            return mapping[platform]
    return None


def _source_for(adapter: Any, report: dict) -> Any:
    chat_id = report.get("chat_id")
    session_key = report.get("session_key")
    if not isinstance(chat_id, str) or not chat_id or not isinstance(session_key, str) or not session_key:
        return None
    chat_type = report.get("chat_type") if report.get("chat_type") in _CHAT_TYPES else _chat_type_from_key(
        session_key, report.get("platform")
    )
    if chat_type not in _CHAT_TYPES:
        return None

    def optional(name: str) -> Optional[str]:
        value = report.get(name)
        return value if isinstance(value, str) and value else None

    return adapter.build_source(
        chat_id=chat_id,
        chat_type=chat_type,
        thread_id=optional("thread_id"),
        message_id=optional("message_id"),
        scope_id=optional("scope_id"),
        user_id=optional("user_id"),
    )


def _chat_type_from_key(session_key: str, platform: object) -> str:
    """Chat type slot of a key this process stored. Not a parse of worker text."""
    if not isinstance(platform, str) or not platform:
        return ""
    parts = session_key.split(":")
    if len(parts) >= 4 and parts[2] == platform and parts[3] in _CHAT_TYPES:
        return parts[3]
    return ""
