"""Durable, fail-closed event outbox for RealtimeSTT journal delivery."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable


class OutboxDeliveryError(RuntimeError):
    """Raised when the journal does not acknowledge the exact queued event."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class EventOutbox:
    """SQLite-backed outbox with exact-response and sequence validation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS pending_events (
                    event_id TEXT PRIMARY KEY,
                    event_json TEXT NOT NULL,
                    delivered INTEGER NOT NULL DEFAULT 0,
                    assigned_sequence INTEGER,
                    response_json TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    last_attempt_at TEXT,
                    last_error TEXT
                )"""
            )
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        columns = {
            str(row[1])
            for row in db.execute("PRAGMA table_info(pending_events)").fetchall()
        }
        additions = {
            "attempt_count": "INTEGER NOT NULL DEFAULT 0",
            "last_attempt_at": "TEXT",
            "last_error": "TEXT",
        }
        for name, ddl in additions.items():
            if name not in columns:
                db.execute(f"ALTER TABLE pending_events ADD COLUMN {name} {ddl}")

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def put(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise ValueError("outbox_event_invalid")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("outbox_event_id_missing")
        if "sequence" in event:
            raise ValueError("outbox_producer_sequence_forbidden")
        encoded = _canonical_json(event)
        with self.connect() as db:
            row = db.execute(
                "SELECT event_json FROM pending_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row and row[0] != encoded:
                raise ValueError("outbox_event_id_conflict")
            db.execute(
                "INSERT OR IGNORE INTO pending_events(event_id,event_json) VALUES(?,?)",
                (event_id, encoded),
            )

    @staticmethod
    def _validate_response(
        event_id: str,
        original: dict[str, Any],
        response: dict[str, Any] | None,
    ) -> tuple[int, dict[str, Any]]:
        if not isinstance(response, dict):
            raise OutboxDeliveryError("outbox_journal_response_invalid")
        stored = response.get("event", response)
        if not isinstance(stored, dict):
            raise OutboxDeliveryError("outbox_journal_event_missing")
        assigned_sequence = stored.get("sequence")
        if not isinstance(assigned_sequence, int) or assigned_sequence < 1:
            raise OutboxDeliveryError("outbox_journal_sequence_missing")
        if stored.get("event_id") != event_id:
            raise OutboxDeliveryError("outbox_journal_event_id_mismatch")
        for key, expected in original.items():
            if stored.get(key) != expected:
                raise OutboxDeliveryError(f"outbox_journal_event_conflict:{key}")
        return assigned_sequence, stored

    def _mark_failure(self, event_id: str, error: Exception) -> None:
        category = str(error) or type(error).__name__
        with self.connect() as db:
            db.execute(
                """UPDATE pending_events
                   SET attempt_count=attempt_count+1,last_attempt_at=?,last_error=?
                   WHERE event_id=?""",
                (_utc_now(), category, event_id),
            )

    def _mark_success(
        self,
        event_id: str,
        assigned_sequence: int,
        response: dict[str, Any],
    ) -> None:
        with self.connect() as db:
            db.execute(
                """UPDATE pending_events
                   SET delivered=1,assigned_sequence=?,response_json=?,
                       attempt_count=attempt_count+1,last_attempt_at=?,last_error=NULL
                   WHERE event_id=?""",
                (
                    assigned_sequence,
                    _canonical_json(response),
                    _utc_now(),
                    event_id,
                ),
            )

    def flush(self, deliver: Callable[[dict[str, Any]], dict[str, Any] | None]) -> int:
        """Deliver every pending event, stopping at the first fail-closed error."""
        with self.connect() as db:
            rows = db.execute(
                "SELECT event_id,event_json FROM pending_events "
                "WHERE delivered=0 ORDER BY rowid"
            ).fetchall()
        delivered_count = 0
        for event_id, encoded in rows:
            original = json.loads(encoded)
            try:
                response = deliver(original)
                assigned_sequence, _stored = self._validate_response(
                    event_id,
                    original,
                    response,
                )
            except Exception as exc:
                self._mark_failure(event_id, exc)
                raise
            self._mark_success(event_id, assigned_sequence, response or {})
            delivered_count += 1
        return delivered_count

    def pending_count(self) -> int:
        with self.connect() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM pending_events WHERE delivered=0"
                ).fetchone()[0]
            )

    def delivered_count(self) -> int:
        with self.connect() as db:
            return int(
                db.execute(
                    "SELECT COUNT(*) FROM pending_events WHERE delivered=1"
                ).fetchone()[0]
            )

    def get(self, event_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT event_json,delivered,assigned_sequence,response_json,
                          attempt_count,last_attempt_at,last_error
                   FROM pending_events WHERE event_id=?""",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "event": json.loads(row[0]),
            "delivered": bool(row[1]),
            "assigned_sequence": row[2],
            "response": json.loads(row[3]) if row[3] else None,
            "attempt_count": int(row[4] or 0),
            "last_attempt_at": row[5],
            "last_error": row[6],
        }

    def latest_delivered(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT event_id,assigned_sequence,response_json
                   FROM pending_events WHERE delivered=1
                   ORDER BY rowid DESC LIMIT 1"""
            ).fetchone()
        if row is None:
            return None
        return {
            "event_id": row[0],
            "assigned_sequence": row[1],
            "response": json.loads(row[2]) if row[2] else None,
        }

    def last_error(self) -> str | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT last_error FROM pending_events
                   WHERE delivered=0 AND last_error IS NOT NULL
                   ORDER BY rowid DESC LIMIT 1"""
            ).fetchone()
        return str(row[0]) if row else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "pending_count": self.pending_count(),
            "delivered_count": self.delivered_count(),
            "last_error": self.last_error(),
        }
