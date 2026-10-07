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

import json
import logging
import os
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
    """Public destinations and the only pubkey whose top-level marker is work.

    ``status`` is ``ready`` or ``absent`` when the file can be trusted.
    ``unreadable`` and ``malformed`` keep previously known chat ids and
    do not authorize a new assignment.
    """

    channel_id: Optional[str] = None
    legacy_dm_id: Optional[str] = None
    assigner_pubkey: Optional[str] = None
    status: str = "ready"
    remembered: tuple[str, ...] = ()

    def destination(self, chat_id: str) -> Optional[str]:
        """``channel``, ``legacy``, or None when this chat is not coordination."""
        text = _chat_key(chat_id)
        if self.channel_id and text == self.channel_id:
            return "channel"
        if self.legacy_dm_id and text == self.legacy_dm_id.lower():
            return "legacy"
        return None

    def known_chats(self) -> frozenset[str]:
        """Chat ids that are coordination destinations by identity, not by text."""
        found = {item for item in self.remembered if item}
        if self.channel_id:
            found.add(self.channel_id)
        if self.legacy_dm_id:
            found.add(self.legacy_dm_id.lower())
        return frozenset(found)

    @property
    def degraded(self) -> bool:
        return self.status in {"unreadable", "malformed"}


def _home() -> Optional[Path]:
    try:
        from gateway.delivery_ledger import _db_path

        return _db_path().parent
    except Exception:
        logger.warning("Coordination settings path was not resolved", exc_info=True)
        return None


def _read_config_mapping(path: Path) -> dict:
    """Effective user config for this home.

    Parse errors and a non-mapping root raise. Callers turn that into an
    unidentified destination and keep the last ready chat ids. A UTF-8 BOM
    is accepted. Defaults are not merged, so a missing fleet key stays absent.
    """
    from hermes_cli.config_effective import load_user_config_effective

    loaded = load_user_config_effective(path, fail_closed=True, reject_non_mapping=True)
    if not isinstance(loaded, dict):
        raise ValueError("coordination config is not a mapping")
    return loaded


def settings_path() -> Optional[Path]:
    """Config beside the delivery ledger. A missing file is not a destination."""
    home = _home()
    if home is None:
        return None
    path = home / "config.yaml"
    if path.is_symlink() or not path.is_file():
        return None
    return path


def _known_path(home: Path) -> Path:
    return home / "fleet-coordination-known.json"


def _read_known(home: Optional[Path]) -> tuple[str, ...]:
    """Public ids from the last ready load. A bad cache adds nothing."""
    if home is None:
        return ()
    path = _known_path(home)
    if path.is_symlink() or not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Coordination destination cache was not read", exc_info=True)
        return ()
    if not isinstance(payload, dict):
        return ()
    found: list[str] = []
    channel = _uuid(payload.get("channel_id"))
    legacy = _chat_id(payload.get("legacy_dm_id"))
    if channel:
        found.append(channel)
    if legacy:
        found.append(legacy.lower())
    return tuple(found)


def _write_known(home: Optional[Path], settings: CoordinationSettings) -> None:
    if home is None or settings.degraded or not settings.known_chats():
        return
    path = _known_path(home)
    if path.is_symlink():
        logger.warning("Coordination destination cache is a symlink; not writing it")
        return
    payload = {
        "channel_id": settings.channel_id,
        "legacy_dm_id": settings.legacy_dm_id,
        "assigner_pubkey": settings.assigner_pubkey,
    }
    temp = home / f".fleet-coordination-known.{os.getpid()}.tmp"
    try:
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, path)
        os.chmod(path, 0o600)
    except OSError:
        logger.warning("Coordination destination cache was not written", exc_info=True)
        try:
            temp.unlink()
        except OSError:
            pass


def load_coordination_settings(path: Optional[Path] = None) -> CoordinationSettings:
    """Read the public Fleet keys.

    A missing file or a fleet section with none of these keys is
    ``absent``: nothing is a coordination chat. An unreadable file, a
    symlink, or a present key that is not a valid id is degraded. The
    last ready ids stay known. Message text is not a destination.
    """
    home = _home()
    target = path if path is not None else (None if home is None else home / "config.yaml")
    remembered = _read_known(home if path is None else (target.parent if target is not None else None))
    if target is None or target.is_symlink() or not target.is_file():
        if target is not None and target.is_symlink():
            return CoordinationSettings(status="unreadable", remembered=remembered)
        return CoordinationSettings(status="absent", remembered=())
    try:
        loaded = _read_config_mapping(target)
    except Exception:
        logger.warning("Coordination settings were not read", exc_info=True)
        return CoordinationSettings(status="unreadable", remembered=remembered)
    if not isinstance(loaded, dict):
        return CoordinationSettings(status="malformed", remembered=remembered)
    if "fleet" not in loaded or loaded.get("fleet") is None:
        return CoordinationSettings(status="absent", remembered=())
    fleet = loaded.get("fleet")
    if not isinstance(fleet, dict):
        return CoordinationSettings(status="malformed", remembered=remembered)
    present = [key for key in (
        "coordination_channel_id",
        "control_coordination_chat_id",
        "control_assigner_pubkey",
    ) if key in fleet and fleet.get(key) not in (None, "")]
    if not present:
        return CoordinationSettings(status="absent", remembered=())
    channel = _uuid(fleet.get("coordination_channel_id")) if "coordination_channel_id" in fleet else None
    legacy = _chat_id(fleet.get("control_coordination_chat_id")) if "control_coordination_chat_id" in fleet else None
    assigner = _pubkey(fleet.get("control_assigner_pubkey")) if "control_assigner_pubkey" in fleet else None
    malformed = (
        ("coordination_channel_id" in fleet and fleet.get("coordination_channel_id") not in (None, "") and channel is None)
        or ("control_coordination_chat_id" in fleet and fleet.get("control_coordination_chat_id") not in (None, "") and legacy is None)
        or ("control_assigner_pubkey" in fleet and fleet.get("control_assigner_pubkey") not in (None, "") and assigner is None)
    )
    if malformed:
        partial = CoordinationSettings(
            channel_id=channel,
            legacy_dm_id=legacy,
            assigner_pubkey=None,
            status="malformed",
            remembered=remembered,
        )
        return partial
    ready = CoordinationSettings(
        channel_id=channel,
        legacy_dm_id=legacy,
        assigner_pubkey=assigner,
        status="ready",
        remembered=remembered,
    )
    _write_known(target.parent, ready)
    return ready


def journal_chat_ids(directory: Optional[Path] = None) -> frozenset[str]:
    """Channel ids stored on delegation records. Text is not consulted."""
    folder = directory
    if folder is None:
        try:
            from gateway.fleet_delegation import _journal_base

            folder = _journal_base()
        except Exception:
            logger.warning("Coordination journal was not resolved", exc_info=True)
            return frozenset()
    if folder is None or folder.is_symlink() or not folder.is_dir():
        return frozenset()
    found: set[str] = set()
    try:
        paths = list(folder.glob("*.json"))
    except OSError:
        return frozenset()
    for path in paths:
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("delegation_id"), str):
            continue
        channel = _uuid(payload.get("channel_id")) or _chat_id(payload.get("channel_id"))
        if channel:
            found.add(channel.lower())
    return frozenset(found)


def held_chat_ids(home: Optional[Path] = None) -> frozenset[str]:
    """Chats a quarantine hold removed from ordinary dispatch."""
    root = home if home is not None else _home()
    if root is None:
        return frozenset()
    path = root / "buzz" / "coordination-dispatch-hold.json"
    if path.is_symlink() or not path.is_file():
        return frozenset()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        logger.warning("Coordination dispatch hold was not read", exc_info=True)
        return frozenset()
    chats = payload.get("chats") if isinstance(payload, dict) else None
    if not isinstance(chats, list):
        return frozenset()
    found: set[str] = set()
    for item in chats:
        channel = _uuid(item) or _chat_id(item)
        if channel:
            found.add(channel.lower())
    return frozenset(found)


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

    A degraded config or a quarantine hold identifies chats by id. It
    does not read the message body. Human chats stay on ``normal``.
    """
    loaded = settings if settings is not None else load_coordination_settings()
    key = _chat_key(channel_id)
    if settings is None and key in held_chat_ids():
        return "control" if control_integration else "suppress"
    if loaded.degraded:
        known = set(loaded.known_chats())
        if control_integration:
            known |= set(journal_chat_ids())
        if key not in known:
            return "normal"
        return "control" if control_integration else "suppress"
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
    if loaded.degraded or loaded.destination(channel_id) != "channel":
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
    if not text or any(char in text for char in "\n\r\x00") or len(text) > 128:
        return None
    return text


def _chat_key(value: object) -> str:
    return str(value or "").strip().lower()
