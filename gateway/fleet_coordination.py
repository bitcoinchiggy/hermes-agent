"""Which Buzz events in a Fleet coordination destination may start work.

Buzz ``private`` is membership gating, not encryption. Kind 9 stream
messages are plaintext. Every current member receives every message; a
``p`` tag addresses one pubkey and does not hide the event from the
others. A removed member fails later REQ and live fan-out, and keeps
any plaintext already delivered.

Coordination destinations are ``fleet.coordination_channel_id`` and the
legacy ``fleet.control_coordination_chat_id`` in this home's
``config.yaml``. Other chats are unchanged. Channel membership is not
an input here and does not grant permission to assign work.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ASSIGNMENT_PREFIX = "[fleet-delegation "
EVALUATION_PREFIX = "[fleet-evaluation "
_DELEGATION_ID_RE = re.compile(r"dlg_[0-9a-f]{32}")
_ASSIGNMENT_RE = re.compile(r"^\[fleet-delegation (dlg_[0-9a-f]{32})\](?:\n|\Z)")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
# Kind 9 is what ``buzz messages send`` publishes. Kind 40002 is the
# other stored stream kind. Reactions (7) and system rows (40099) are not.
_STREAM_KINDS = frozenset({9, 40002})
_EVALUATION_TEXT_MAX = 6000
REQUEST_DUMP_KEEP = 20
REQUEST_DUMP_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class CoordinationSettings:
    """Public destinations and the only pubkey whose top-level marker is work."""

    channel_id: Optional[str] = None
    legacy_dm_id: Optional[str] = None
    assigner_pubkey: Optional[str] = None

    def destination(self, chat_id: str) -> Optional[str]:
        """``channel``, ``legacy``, or None when this chat is not coordination."""
        text = str(chat_id or "").strip().lower()
        if self.channel_id and text == self.channel_id:
            return "channel"
        if self.legacy_dm_id and text == self.legacy_dm_id.lower():
            return "legacy"
        return None


def settings_path() -> Optional[Path]:
    """Config beside the delivery ledger. Missing or unreadable is no destination."""
    try:
        from gateway.delivery_ledger import _db_path

        path = _db_path().parent / "config.yaml"
    except Exception:
        logger.warning("Coordination settings path was not resolved", exc_info=True)
        return None
    if path.is_symlink() or not path.is_file():
        return None
    return path


def load_coordination_settings(path: Optional[Path] = None) -> CoordinationSettings:
    """Read the public Fleet keys. An unreadable file identifies nothing."""
    target = path if path is not None else settings_path()
    if target is None or target.is_symlink() or not target.is_file():
        return CoordinationSettings()
    try:
        import hermes_yaml as yaml

        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Coordination settings were not read", exc_info=True)
        return CoordinationSettings()
    if not isinstance(loaded, dict):
        return CoordinationSettings()
    fleet = loaded.get("fleet")
    if not isinstance(fleet, dict):
        return CoordinationSettings()
    return CoordinationSettings(
        channel_id=_uuid(fleet.get("coordination_channel_id")),
        legacy_dm_id=_chat_id(fleet.get("control_coordination_chat_id")),
        assigner_pubkey=_pubkey(fleet.get("control_assigner_pubkey")),
    )


def coordination_disposition(
    *,
    channel_id: str,
    event: dict,
    self_pubkey: str,
    control_integration: bool,
    settings: Optional[CoordinationSettings] = None,
) -> str:
    """``normal``, ``assignment``, ``control``, or ``suppress``.

    ``control`` means Control may correlate a reply and must not start a
    model turn in this chat. ``assignment`` is the one worker execution.
    ``suppress`` drops notices, errors, acknowledgements, reactions,
    startup and shutdown rows, evaluations, and every other member.
    """
    loaded = settings if settings is not None else load_coordination_settings()
    if loaded.destination(channel_id) is None:
        return "normal"
    if control_integration:
        return "control"
    if worker_should_execute(event, self_pubkey, loaded.assigner_pubkey):
        return "assignment"
    return "suppress"


def worker_should_execute(event: dict, self_pubkey: str, assigner_pubkey: Optional[str]) -> bool:
    """True only for one top-level assignment addressed to this pubkey.

    The author must be the configured assigner. A ``p`` tag alone is not
    authority. A reply, including an evaluation, is never an assignment.
    """
    if not assigner_pubkey or not _pubkey(self_pubkey):
        return False
    if int(event.get("kind") or 0) not in _STREAM_KINDS:
        return False
    if _reply_parent(event):
        return False
    content = event.get("content")
    if not isinstance(content, str) or _contains_secret(content):
        return False
    if content.strip() == "!shutdown":
        return False
    if not _ASSIGNMENT_RE.match(content):
        return False
    author = _pubkey(event.get("pubkey"))
    if author != assigner_pubkey or author == self_pubkey.lower():
        return False
    mentioned = _p_tags(event)
    return mentioned == {self_pubkey.lower()}


def should_post_evaluation(channel_id: str, plan: dict, settings: Optional[CoordinationSettings] = None) -> bool:
    """True when a new human report should also be copied into the channel thread.

    The legacy coordination DM is not a thread target. A duplicate or
    in-flight claim does not set ``_evaluation_ready``.
    """
    if not isinstance(plan, dict) or plan.get("_evaluation_ready") is not True:
        return False
    loaded = settings if settings is not None else load_coordination_settings()
    if loaded.destination(channel_id) != "channel":
        return False
    event_id = plan.get("inbound_event_id")
    if not isinstance(event_id, str) or not _HEX64_RE.fullmatch(event_id):
        return False
    delegation_id = plan.get("delegation_id")
    return isinstance(delegation_id, str) and evaluation_body(delegation_id, plan.get("_evaluation_text")) is not None


def evaluation_body(delegation_id: str, model_text: object) -> Optional[str]:
    """Thread copy of Control's evaluation. Not an assignment and not a secret."""
    if not isinstance(delegation_id, str) or not _DELEGATION_ID_RE.fullmatch(delegation_id):
        return None
    notice = (
        f"{EVALUATION_PREFIX}{delegation_id}]\n"
        "Control reported the recorded worker result to the originating human session. "
        "This notice is not an assignment.\n"
    )
    text = model_text if isinstance(model_text, str) else ""
    if _contains_secret(text) or "\x00" in text:
        return notice
    text = text.strip()
    if not text:
        return notice
    if len(text) > _EVALUATION_TEXT_MAX:
        text = text[:_EVALUATION_TEXT_MAX]
    body = notice + "\n" + text
    if body.startswith(ASSIGNMENT_PREFIX) or _contains_secret(body):
        return notice
    return body


def _reply_parent(event: dict) -> Optional[str]:
    tags = event.get("tags")
    if not isinstance(tags, list):
        return None
    reply_id = root_id = last_e = None
    for tag in tags:
        if not isinstance(tag, (list, tuple)) or len(tag) < 2 or str(tag[0]) != "e":
            continue
        target = str(tag[1] or "").strip()
        if not target:
            continue
        last_e = target
        marker = str(tag[3]) if len(tag) > 3 else ""
        if marker == "reply":
            reply_id = target
        elif marker == "root":
            root_id = target
    return reply_id or root_id or last_e


def _p_tags(event: dict) -> set[str]:
    tags = event.get("tags")
    found: set[str] = set()
    if not isinstance(tags, list):
        return found
    for tag in tags:
        if (
            isinstance(tag, (list, tuple))
            and len(tag) > 1
            and tag[0] == "p"
            and _pubkey(tag[1])
        ):
            found.add(str(tag[1]).lower())
    return found


def _contains_secret(text: str) -> bool:
    lowered = text.lower()
    return "nsec1" in lowered or "buzz_private_key" in lowered


def _pubkey(value: object) -> Optional[str]:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value):
        return None
    return value.lower()


def _uuid(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if not _UUID_RE.fullmatch(text):
        return None
    return text


def _chat_id(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or any(char in text for char in "\n\r\x00"):
        return None
    return text
