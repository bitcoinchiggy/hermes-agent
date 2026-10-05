"""Backed-up quarantine for one legacy coordination DM.

Startup redelivery sends every ``pending``, ``attempting``, and ``failed``
Buzz obligation. Startup also resumes sessions marked ``resume_pending``.
The Buzz cursor fetches relay events newer than ``last_ts`` that are not
in ``seen``. Those three stores can revive a coordination-DM reply loop
even when the parent is not a delegation. The database copy is a SQLite
backup, so uncheckpointed WAL commits are in that file.

Selection is the chat id. Startup notices, error sends, and ordinary
responses in that chat are all included. Message text only labels the
dry-run report. Unrelated chats and delegation journal files are not
modified. The default entry point refuses a live Hermes home.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Optional

_REPLAY_STATES = frozenset({"pending", "attempting", "failed"})
_LIVE_PREFIX = "/home/hermes"


class QuarantineError(RuntimeError):
    """The plan was not applied. Live state was not changed."""


def label_outbound(content: object) -> str:
    """Report label only. Not a reason to select or skip a row."""
    text = content.lower() if isinstance(content, str) else ""
    if "http 402" in text or "interruption" in text or text.startswith("error"):
        return "error"
    if (
        "!shutdown" in text
        or "gateway is online" in text
        or "startup" in text
        or "shutting down" in text
    ):
        return "startup"
    return "response"


def plan_quarantine(
    home: Path,
    *,
    chats: list[str],
    journal_dir: Optional[Path] = None,
    loop_parent: str = "",
) -> dict:
    """Read-only report. Does not copy, update, or delete."""
    root = _root(home)
    targets = _targets(chats)
    obligations = _obligations(root / "state.db", targets, loop_parent)
    sessions = _sessions(root, targets)
    cursors = _cursors(root / "buzz" / "channel-cursors.json", targets)
    return {
        "dry_run": True,
        "home": str(root),
        "chats": sorted(targets),
        "loop_parent": loop_parent,
        "outbound": obligations["selected"],
        "preserved_outbound": obligations["preserved"],
        "inbound_sessions": sessions["selected"],
        "preserved_sessions": sessions["preserved"],
        "inbound_cursors": cursors,
        "preserved_delegations": _delegations(journal_dir),
        "apply": False,
    }


def apply_quarantine(home: Path, report: dict, *, backup_dir: Path) -> dict:
    """Copy the selected stores, then quarantine only the reported chats.

    Delegation journal files are not in the backup set and are not
    rewritten. A failed backup leaves the home unchanged.
    """
    if not isinstance(report, dict) or not report.get("chats"):
        raise QuarantineError("quarantine report has no chats")
    root = _root(home)
    targets = _targets(list(report["chats"]))
    if set(targets) != set(report["chats"]):
        raise QuarantineError("quarantine report chats do not match")
    backup = _backup(root, backup_dir)
    now = time.time()
    outbound = _quarantine_obligations(root / "state.db", targets, now)
    sessions = _clear_resume(root, targets)
    hold = _write_hold(root, targets)
    applied = dict(report)
    applied["dry_run"] = False
    applied["apply"] = True
    applied["backup"] = str(backup)
    applied["quarantined_obligations"] = outbound
    applied["cleared_sessions"] = sessions
    applied["hold"] = str(hold)
    return applied


def _root(home: Path) -> Path:
    if not isinstance(home, Path):
        raise QuarantineError("quarantine home is unusable")
    try:
        root = home.resolve()
    except OSError as exc:
        raise QuarantineError("quarantine home is unusable") from exc
    if not root.is_dir() or root.is_symlink():
        raise QuarantineError("quarantine home is unusable")
    if str(root) == _LIVE_PREFIX or str(root).startswith(_LIVE_PREFIX + "/"):
        raise QuarantineError("refusing the live Hermes home")
    return root


def _targets(chats: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in chats:
        if not isinstance(item, str):
            raise QuarantineError("quarantine chat is unusable")
        text = item.strip().lower()
        if not text or any(char in text for char in "\n\r\x00"):
            raise QuarantineError("quarantine chat is unusable")
        if text not in cleaned:
            cleaned.append(text)
    if not cleaned:
        raise QuarantineError("quarantine chat is unusable")
    return cleaned


def _obligations(path: Path, targets: list[str], loop_parent: str) -> dict:
    selected: list[dict] = []
    preserved = 0
    if path.is_symlink() or not path.is_file():
        return {"selected": selected, "preserved": preserved}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "delivery_obligations" not in tables:
            return {"selected": selected, "preserved": preserved}
        rows = conn.execute(
            """SELECT obligation_id, platform, chat_id, state, content, reply_to_message_id
               FROM delivery_obligations"""
        )
        for obligation_id, platform, chat_id, state, content, reply_to in rows:
            chat = str(chat_id or "").strip().lower()
            replayable = str(state or "") in _REPLAY_STATES and str(platform or "") == "buzz"
            if chat in targets and replayable:
                selected.append(
                    {
                        "obligation_id": obligation_id,
                        "chat_id": chat,
                        "state": state,
                        "label": label_outbound(content),
                        "replies_to_loop_parent": bool(loop_parent) and reply_to == loop_parent,
                    }
                )
            elif replayable:
                preserved += 1
    finally:
        conn.close()
    return {"selected": selected, "preserved": preserved}


def _session_chat(entry: dict) -> str:
    origin = entry.get("origin")
    if isinstance(origin, dict) and origin.get("chat_id"):
        return str(origin.get("chat_id") or "").strip().lower()
    return str(entry.get("chat_id") or "").strip().lower()


def _sessions(root: Path, targets: list[str]) -> dict:
    selected: list[dict] = []
    preserved = 0
    for entry_id, entry in _routing_entries(root):
        if not isinstance(entry, dict) or entry.get("resume_pending") is not True:
            continue
        chat = _session_chat(entry)
        if chat in targets:
            selected.append(
                {
                    "session_key": entry_id,
                    "chat_id": chat,
                    "resume_pending": True,
                    "resume_reason": entry.get("resume_reason"),
                }
            )
        else:
            preserved += 1
    return {"selected": selected, "preserved": preserved}


def _routing_entries(root: Path) -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    db = root / "state.db"
    if db.is_file() and not db.is_symlink():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "gateway_routing" in tables:
                for session_key, entry_json in conn.execute(
                    "SELECT session_key, entry_json FROM gateway_routing"
                ):
                    try:
                        payload = json.loads(entry_json)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if isinstance(payload, dict):
                        found.append((str(session_key), payload))
        finally:
            conn.close()
    mirror = root / "sessions" / "sessions.json"
    if mirror.is_file() and not mirror.is_symlink() and not found:
        try:
            payload = json.loads(mirror.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            for key, entry in payload.items():
                if str(key).startswith("_") or not isinstance(entry, dict):
                    continue
                found.append((str(key), entry))
    return found


def _cursors(path: Path, targets: list[str]) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    channels = payload.get("channels") if isinstance(payload, dict) else None
    if not isinstance(channels, dict):
        return []
    rows: list[dict] = []
    for chat_id, entry in channels.items():
        if str(chat_id).strip().lower() not in targets or not isinstance(entry, dict):
            continue
        seen = entry.get("seen") if isinstance(entry.get("seen"), list) else []
        rows.append(
            {
                "chat_id": str(chat_id).strip().lower(),
                "last_ts": entry.get("last_ts"),
                "seen": len(seen),
                "restart_replays_relay_history": True,
            }
        )
    return rows


def _delegations(journal_dir: Optional[Path]) -> list[dict]:
    if journal_dir is None or journal_dir.is_symlink() or not journal_dir.is_dir():
        return []
    kept: list[dict] = []
    for path in sorted(journal_dir.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        state = payload.get("state")
        if state not in {"pending", "accepted", "ambiguous", "bound"}:
            continue
        kept.append(
            {
                "delegation_id": payload.get("delegation_id"),
                "state": state,
                "channel_id": payload.get("channel_id"),
                "origin_chat_id": payload.get("origin_chat_id"),
                "preserved": True,
            }
        )
    return kept


def _backup(root: Path, backup_dir: Path) -> Path:
    if not isinstance(backup_dir, Path):
        raise QuarantineError("quarantine backup is unusable")
    try:
        destination = backup_dir.resolve()
    except OSError as exc:
        raise QuarantineError("quarantine backup is unusable") from exc
    if destination == root or root in destination.parents or destination in root.parents:
        raise QuarantineError("quarantine backup must be outside the home")
    if destination.exists():
        raise QuarantineError("quarantine backup already exists")
    destination.mkdir(parents=False)
    database = root / "state.db"
    if database.is_file() and not database.is_symlink():
        _backup_sqlite(database, destination / "state.db")
    for relative in (
        Path("sessions") / "sessions.json",
        Path("buzz") / "channel-cursors.json",
    ):
        source = root / relative
        if source.is_symlink() or not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def _backup_sqlite(source: Path, target: Path) -> None:
    """One consistent file, including commits that are still in the WAL."""
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dest = sqlite3.connect(target)
        try:
            src.backup(dest)
        finally:
            dest.close()
    finally:
        src.close()


def _quarantine_obligations(path: Path, targets: list[str], now: float) -> int:
    if path.is_symlink() or not path.is_file():
        return 0
    conn = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "delivery_obligations" not in tables:
            return 0
        changed = 0
        for chat in targets:
            cursor = conn.execute(
                """UPDATE delivery_obligations
                   SET state='quarantined', updated_at=?, last_error=?
                   WHERE platform='buzz' AND lower(chat_id)=? AND state IN ('pending', 'attempting', 'failed')""",
                (
                    now,
                    "quarantined: legacy coordination replay; not published",
                    chat,
                ),
            )
            changed += cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        conn.commit()
        return changed
    finally:
        conn.close()


def _clear_resume(root: Path, targets: list[str]) -> int:
    cleared = 0
    db = root / "state.db"
    if db.is_file() and not db.is_symlink():
        conn = sqlite3.connect(db)
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "gateway_routing" in tables:
                rows = list(conn.execute("SELECT rowid, entry_json FROM gateway_routing"))
                for rowid, entry_json in rows:
                    try:
                        payload = json.loads(entry_json)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if not isinstance(payload, dict) or payload.get("resume_pending") is not True:
                        continue
                    if _session_chat(payload) not in targets:
                        continue
                    payload["resume_pending"] = False
                    payload["resume_reason"] = "coordination-quarantine"
                    conn.execute(
                        "UPDATE gateway_routing SET entry_json=? WHERE rowid=?",
                        (json.dumps(payload), rowid),
                    )
                    cleared += 1
                conn.commit()
        finally:
            conn.close()
    mirror = root / "sessions" / "sessions.json"
    if mirror.is_file() and not mirror.is_symlink():
        try:
            payload = json.loads(mirror.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            changed = False
            for key, entry in payload.items():
                if str(key).startswith("_") or not isinstance(entry, dict):
                    continue
                if entry.get("resume_pending") is not True or _session_chat(entry) not in targets:
                    continue
                entry["resume_pending"] = False
                entry["resume_reason"] = "coordination-quarantine"
                changed = True
                if not (db.is_file() and not db.is_symlink()):
                    cleared += 1
            if changed:
                mirror.write_text(json.dumps(payload), encoding="utf-8")
    return cleared


def _write_hold(root: Path, targets: list[str]) -> Path:
    folder = root / "buzz"
    folder.mkdir(mode=0o700, exist_ok=True)
    path = folder / "coordination-dispatch-hold.json"
    if path.is_symlink():
        raise QuarantineError("coordination dispatch hold is a symlink")
    path.write_text(
        json.dumps({"chats": targets, "reason": "legacy-coordination-replay"}),
        encoding="utf-8",
    )
    return path
