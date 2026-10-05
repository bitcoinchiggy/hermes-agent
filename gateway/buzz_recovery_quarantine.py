"""Quarantine for one legacy Buzz recovery addressed to the coordination DM.

Operator and Control do not share a filesystem. A missing reply parent is not
enough to decide that a Buzz row is that DM: Control's reports, other DMs,
and channels also recover without one. The destination is this Hermes home's
``config.yaml`` key ``fleet.control_coordination_chat_id`` compared with the
row's ``chat_id``. Message text, allowlists, and baseline channel ids are not
that destination. When the key is absent, the row is published normally.

The ledger row stays in ``quarantined``. That is not delivery. Nothing in
this module fills ``reply_to_message_id``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_RECORDS = 64
_FIELDS = (
    "obligation_id",
    "platform",
    "chat_id",
    "thread_id",
    "content",
    "reply_to_message_id",
    "published",
)


def coordination_chat_id() -> Optional[str]:
    """Return this home's coordination DM id, or None when it is not configured.

    The value is ``fleet.control_coordination_chat_id`` in the config.yaml next
    to the delivery ledger. Another host's files are not read. A missing,
    unreadable, or non-string value means the destination is not identified.
    """
    from gateway.delivery_ledger import _db_path

    path = _db_path().parent / "config.yaml"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        import hermes_yaml as yaml

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        # An unreadable setting does not identify the coordination DM.
        # Recovery publishes the row instead of quarantining every Buzz chat.
        logger.warning("Coordination chat id was not read", exc_info=True)
        return None
    if not isinstance(loaded, dict):
        return None
    fleet = loaded.get("fleet")
    if not isinstance(fleet, dict):
        return None
    value = fleet.get("control_coordination_chat_id")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or any(char in text for char in "\n\r\x00"):
        return None
    return text


def unanchored_coordination_destination(row: dict) -> bool:
    """True for a parentless Buzz row whose chat is a known coordination destination.

    A missing configuration publishes the row. An unreadable or malformed
    configuration quarantines only chats already known by id, including
    journal channel ids. Other chats, including human reports, are published.
    Message text is not the destination.
    """
    if str(row.get("platform") or "") != "buzz":
        return False
    parent = row.get("reply_to_message_id")
    if isinstance(parent, str):
        if parent.strip():
            return False
    elif parent is not None:
        return False
    from gateway.fleet_coordination import journal_chat_ids, load_coordination_settings

    loaded = load_coordination_settings()
    chat = str(row.get("chat_id") or "").strip().lower()
    if not chat:
        return False
    if loaded.degraded:
        return chat in loaded.known_chats() or chat in journal_chat_ids()
    if loaded.status != "ready":
        return False
    return loaded.destination(chat) is not None


def quarantine_unanchored_buzz_result(
    *,
    obligation_id: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
) -> bool:
    """Store one unanchored Buzz body on this worker. Returns False when it was not stored."""
    if not obligation_id or not chat_id or not content:
        return False
    folder = _directory(create=True)
    if folder is None:
        return False
    target = _path(folder, obligation_id)
    if not target.exists():
        current = [path for path in folder.glob("*.json") if path.is_file() and not path.is_symlink()]
        if len(current) >= _MAX_RECORDS:
            logger.warning("Buzz recovery quarantine is full; obligation %s was not stored", obligation_id)
            return False
    payload = {
        "obligation_id": obligation_id,
        "platform": "buzz",
        "chat_id": chat_id,
        "thread_id": thread_id or None,
        "content": content,
        "reply_to_message_id": None,
        "published": False,
    }
    return _write(folder, target, payload)


def load_quarantine(obligation_id: str) -> Optional[dict]:
    """Return this worker's quarantine record, or None."""
    folder = _directory(create=False)
    if folder is None or not obligation_id:
        return None
    record = _read_file(_path(folder, obligation_id))
    if record is None or record.get("obligation_id") != obligation_id:
        return None
    if record.get("platform") != "buzz" or record.get("published") is not False:
        return None
    if record.get("reply_to_message_id") not in (None, ""):
        return None
    return record


def _directory(*, create: bool) -> Optional[Path]:
    from gateway.delivery_ledger import _db_path

    base = _db_path().parent
    path = base / "buzz-recovery-quarantine"
    if path.is_symlink():
        logger.warning("Buzz recovery quarantine directory is a symlink; not using it")
        return None
    if not create:
        if path.is_dir():
            return path
        return None
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:
        logger.warning("Buzz recovery quarantine directory was not created", exc_info=True)
        return None
    if path.is_symlink() or not path.is_dir():
        return None
    return path


def _path(folder: Path, obligation_id: str) -> Path:
    digest = hashlib.sha256(obligation_id.encode("utf-8")).hexdigest()
    return folder / f"{digest}.json"


def _read_file(path: Path) -> Optional[dict]:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {key: payload.get(key) for key in _FIELDS}


def _write(folder: Path, target: Path, payload: dict) -> bool:
    if target.is_symlink():
        return False
    temp = folder / f".{target.stem}.{os.getpid()}.tmp"
    try:
        temp.write_text(json.dumps({key: payload.get(key) for key in _FIELDS}), encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, target)
        os.chmod(target, 0o600)
    except OSError:
        logger.warning("Buzz recovery quarantine write failed", exc_info=True)
        try:
            temp.unlink()
        except OSError:
            pass
        return False
    return True
