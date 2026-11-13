"""Concurrency-safe in-memory and optional SQLite Ledger for ClaimLayer v3."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, Mapping

from claimlayer.models import (
    Assertion,
    FactRecord,
    Policy,
    ResolutionRecord,
    Source,
    Verdict,
    normalize_entity_key,
    normalize_key,
    normalize_predicate_key,
)
from claimlayer.verifiers.precedence import is_source_active

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subj_key TEXT NOT NULL,
    pred_key TEXT NOT NULL,
    source_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_lookup ON facts(subj_key, pred_key);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_id);

CREATE TABLE IF NOT EXISTS resolutions (
    subj_key TEXT NOT NULL,
    pred_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (subj_key, pred_key)
);

CREATE TABLE IF NOT EXISTS verdicts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    verdict_id TEXT NOT NULL,
    status TEXT NOT NULL,
    triage_state TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    payload_json TEXT NOT NULL,
    resolution_json TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_verdicts_id ON verdicts(verdict_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_triage ON verdicts(triage_state);

CREATE TABLE IF NOT EXISTS ledger_meta (
    meta_key TEXT PRIMARY KEY,
    meta_value TEXT NOT NULL
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    """Authoritative store for sources, known facts, verdicts, and human resolutions."""

    def __init__(
        self,
        sources: Iterable[Source | dict[str, Any]] | None = None,
        facts: Iterable[FactRecord | dict[str, Any]] | None = None,
        *,
        db_path: str | Path | None = None,
        signing_key: str | bytes | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._db_path = Path(db_path) if db_path is not None else None
        self.signing_key = signing_key
        self._sources: dict[str, Source] = {}
        self._facts: list[FactRecord] = []
        self._fact_set: set[FactRecord] = set()
        self._facts_by_source: dict[str, list[FactRecord]] = {}
        self._facts_by_pred: dict[str, list[FactRecord]] = {}
        self._resolutions: dict[tuple[str, str], ResolutionRecord] = {}
        self._verdict_log: list[dict[str, Any]] = []
        self._last_db_sig: tuple[int, int] | None = None

        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
            self._load_sqlite()

        if sources:
            for src in sources:
                self.add_source(src)
        if facts:
            for fact in facts:
                self.add_fact(fact)

    @contextmanager
    def _db_session(self) -> Generator[sqlite3.Connection, None, None]:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _current_db_sig(self) -> tuple[int, int] | None:
        if self._db_path is None or not self._db_path.exists():
            return None
        st = self._db_path.stat()
        return (st.st_mtime_ns, st.st_size)

    def _sync_from_sqlite_if_changed(self) -> None:
        """Reload state if another Ledger instance or process modified the SQLite file (L1)."""
        if self._db_path is None:
            return
        sig = self._current_db_sig()
        if sig is not None and sig != self._last_db_sig:
            self._load_sqlite()

    def _init_sqlite(self) -> None:
        with self._db_session() as conn:
            conn.executescript(_SQLITE_SCHEMA)
        self._last_db_sig = self._current_db_sig()

    def _index_fact_in_memory(self, rec: FactRecord) -> bool:
        if rec in self._fact_set:
            return False
        self._fact_set.add(rec)
        self._facts.append(rec)
        self._facts_by_source.setdefault(rec.source_id, []).append(rec)
        pk = normalize_predicate_key(rec.predicate)
        raw_pk = normalize_key(rec.predicate)
        self._facts_by_pred.setdefault(pk, []).append(rec)
        if raw_pk != pk:
            self._facts_by_pred.setdefault(raw_pk, []).append(rec)
        return True

    def _load_sqlite(self) -> None:
        self._sources.clear()
        self._facts.clear()
        self._fact_set.clear()
        self._facts_by_source.clear()
        self._facts_by_pred.clear()
        self._resolutions.clear()
        self._verdict_log.clear()

        with self._db_session() as conn:
            for row in conn.execute("SELECT payload_json FROM sources ORDER BY rowid ASC"):
                data = json.loads(row["payload_json"])
                data.pop("sha256", None)
                src = Source(**data)
                self._sources[src.id] = src

            for row in conn.execute("SELECT payload_json FROM facts ORDER BY id ASC"):
                data = json.loads(row["payload_json"])
                self._index_fact_in_memory(FactRecord(**data))

            for row in conn.execute("SELECT subj_key, pred_key, payload_json FROM resolutions"):
                data = json.loads(row["payload_json"])
                self._resolutions[(row["subj_key"], row["pred_key"])] = ResolutionRecord(**data)

            for row in conn.execute("SELECT * FROM verdicts ORDER BY seq ASC"):
                self._verdict_log.append(
                    {
                        "seq": row["seq"],
                        "verdict_id": row["verdict_id"],
                        "status": row["status"],
                        "triage_state": row["triage_state"],
                        "agent_id": row["agent_id"],
                        "policy_id": row["policy_id"],
                        "confidence": row["confidence"],
                        "verdict": json.loads(row["payload_json"]),
                        "resolution": (
                            json.loads(row["resolution_json"]) if row["resolution_json"] else None
                        ),
                        "created_at": row["created_at"],
                        "resolved_at": row["resolved_at"],
                    }
                )
        self._last_db_sig = self._current_db_sig()

    def add_source(
        self,
        source: Source | dict[str, Any],
        *,
        allow_overwrite: bool = True,
    ) -> Source:
        src = source if isinstance(source, Source) else Source(**source)
        with self._lock:
            self._sync_from_sqlite_if_changed()
            if not allow_overwrite and src.id in self._sources:
                existing = self._sources[src.id]
                if existing.sha256 != src.sha256 or existing.rank != src.rank:
                    raise ValueError(
                        f"Source '{src.id}' is already registered and cannot be overwritten."
                    )
            self._sources[src.id] = src
            if self._db_path is not None:
                payload = src.to_dict()
                payload.pop("sha256", None)
                with self._db_session() as conn:
                    conn.execute(
                        """
                        INSERT INTO sources(id, payload_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            payload_json = excluded.payload_json,
                            updated_at = excluded.updated_at
                        """,
                        (src.id, json.dumps(payload, ensure_ascii=False), _utc_now_iso()),
                    )
                self._last_db_sig = self._current_db_sig()
        return src

    def get_source(self, source_id: str) -> Source | None:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return self._sources.get(source_id)

    def list_sources(self) -> list[Source]:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return [self._sources[k] for k in sorted(self._sources.keys())]

    def add_fact(self, fact: FactRecord | dict[str, Any]) -> FactRecord:
        rec = fact if isinstance(fact, FactRecord) else FactRecord(**fact)
        with self._lock:
            self._sync_from_sqlite_if_changed()
            if self._index_fact_in_memory(rec):
                if self._db_path is not None:
                    with self._db_session() as conn:
                        conn.execute(
                            "INSERT INTO facts(subj_key, pred_key, source_id, payload_json) VALUES (?, ?, ?, ?)",
                            (
                                normalize_entity_key(rec.subject),
                                normalize_predicate_key(rec.predicate),
                                rec.source_id,
                                json.dumps(rec.to_dict(), ensure_ascii=False),
                            ),
                        )
                    self._last_db_sig = self._current_db_sig()
        return rec

    def get_facts(
        self,
        subject: str,
        predicate: str,
        *,
        aliases: Mapping[str, str] | None = None,
        allow_section_predicate_match: bool = False,
    ) -> list[FactRecord]:
        sk = normalize_entity_key(subject)
        pk = normalize_predicate_key(predicate, aliases)
        raw_pk = normalize_key(predicate)
        with self._lock:
            self._sync_from_sqlite_if_changed()
            candidates = list(self._facts_by_pred.get(pk, []))
            if raw_pk != pk:
                for f in self._facts_by_pred.get(raw_pk, []):
                    if f not in candidates:
                        candidates.append(f)

            exact = [f for f in candidates if normalize_entity_key(f.subject) == sk]
            if exact or not allow_section_predicate_match:
                return exact
            # If predicate is a section identifier or unique across a source, match by predicate (C7c)
            return candidates

    def get_facts_by_source(self, source_id: str) -> list[FactRecord]:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return list(self._facts_by_source.get(source_id, []))

    def list_facts(self) -> list[FactRecord]:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return list(self._facts)

    def superseded_source_ids(
        self,
        *,
        as_of: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> set[str]:
        """Return IDs of sources superseded globally or for a specific (subject, predicate)."""
        superseded: set[str] = set()
        sk = normalize_entity_key(subject) if subject else None
        pk = normalize_predicate_key(predicate) if predicate else None
        with self._lock:
            self._sync_from_sqlite_if_changed()
            for src in self._sources.values():
                active, _, _ = is_source_active(src, as_of)
                if not active or not src.supersedes:
                    continue
                src_facts = self._facts_by_source.get(src.id, [])
                if not src_facts or sk is None or pk is None:
                    superseded.update(src.supersedes)
                else:
                    touches_pair = any(
                        (normalize_entity_key(f.subject) == sk or len(src_facts) == 1)
                        and normalize_predicate_key(f.predicate) == pk
                        for f in src_facts
                    )
                    if touches_pair:
                        superseded.update(src.supersedes)
        return superseded

    def get_resolution(self, subject: str, predicate: str) -> ResolutionRecord | None:
        key = (normalize_entity_key(subject), normalize_predicate_key(predicate))
        raw_key = (normalize_key(subject), normalize_key(predicate))
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return self._resolutions.get(key) or self._resolutions.get(raw_key)

    def list_resolutions(self) -> list[ResolutionRecord]:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            return list(self._resolutions.values())

    def save_sealed_policy(self, policy: Policy) -> None:
        """Persist canonical policy inside SQLite ledger_meta to prevent workspace policy tampering (W3)."""
        if self._db_path is None:
            return
        with self._lock:
            with self._db_session() as conn:
                conn.execute(
                    """
                    INSERT INTO ledger_meta(meta_key, meta_value)
                    VALUES ('sealed_policy', ?)
                    ON CONFLICT(meta_key) DO UPDATE SET meta_value = excluded.meta_value
                    """,
                    (json.dumps(policy.to_dict(), sort_keys=True),),
                )
            self._last_db_sig = self._current_db_sig()

    def load_sealed_policy(self) -> dict[str, Any] | None:
        if self._db_path is None:
            return None
        with self._lock:
            with self._db_session() as conn:
                row = conn.execute(
                    "SELECT meta_value FROM ledger_meta WHERE meta_key = 'sealed_policy'"
                ).fetchone()
                if row is None:
                    return None
                return dict(json.loads(row["meta_value"]))

    def record_verdict(self, verdict: Verdict) -> dict[str, Any]:
        """Persist a verification verdict into the audit and triage log."""
        now = _utc_now_iso()
        triage_state = "OPEN" if verdict.status == "ESCALATE" else "AUTO_CLOSED"
        payload = verdict.to_dict()
        with self._lock:
            self._sync_from_sqlite_if_changed()
            seq = len(self._verdict_log) + 1
            if self._db_path is not None:
                with self._db_session() as conn:
                    cur = conn.execute(
                        """
                        INSERT INTO verdicts(
                            verdict_id, status, triage_state, agent_id, policy_id,
                            confidence, payload_json, resolution_json, created_at, resolved_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                        """,
                        (
                            verdict.verdict_id,
                            verdict.status,
                            triage_state,
                            verdict.agent_id,
                            verdict.policy_id,
                            verdict.confidence,
                            json.dumps(payload, ensure_ascii=False),
                            now,
                        ),
                    )
                    seq = int(cur.lastrowid or seq)
                self._last_db_sig = self._current_db_sig()

            entry = {
                "seq": seq,
                "verdict_id": verdict.verdict_id,
                "status": verdict.status,
                "triage_state": triage_state,
                "agent_id": verdict.agent_id,
                "policy_id": verdict.policy_id,
                "confidence": verdict.confidence,
                "verdict": payload,
                "resolution": None,
                "created_at": now,
                "resolved_at": None,
            }
            self._verdict_log.append(entry)
            return dict(entry)

    def _is_verdict_fully_resolved(self, verdict_payload: dict[str, Any]) -> bool:
        """Check whether ALL escalated (subject, predicate) items in a verdict have human resolutions (F1)."""
        escalated_pairs: set[tuple[str, str]] = set()
        for iss in verdict_payload.get("issues", []):
            if iss.get("disposition") == "ESCALATE":
                s = iss.get("subject") or ""
                p = iss.get("predicate") or ""
                if s or p:
                    escalated_pairs.add((normalize_entity_key(s), normalize_predicate_key(p)))
        if not escalated_pairs:
            return True
        return all(pair in self._resolutions for pair in escalated_pairs)

    def resolve_escalation(
        self,
        *,
        verdict_id: str | None = None,
        subject: str,
        predicate: str,
        selected_value: str,
        resolver: str = "human_supervisor",
        action: str = "APPROVE_VALUE",
        note: str = "",
    ) -> dict[str, Any]:
        """Record a 1-click human triage resolution and close verdicts once all their conflicts are resolved."""
        now = _utc_now_iso()
        rec = ResolutionRecord(
            subject=subject,
            predicate=predicate,
            selected_value=selected_value,
            resolver=resolver,
            note=note,
            verdict_id=verdict_id,
            resolved_at=now,
        )
        key = (normalize_entity_key(subject), normalize_predicate_key(predicate))
        res_payload = {**rec.to_dict(), "action": action}

        with self._lock:
            self._sync_from_sqlite_if_changed()
            if action != "REJECT_TO_AGENT":
                self._resolutions[key] = rec
                if self._db_path is not None:
                    with self._db_session() as conn:
                        conn.execute(
                            """
                            INSERT INTO resolutions(subj_key, pred_key, payload_json)
                            VALUES (?, ?, ?)
                            ON CONFLICT(subj_key, pred_key) DO UPDATE SET
                                payload_json = excluded.payload_json
                            """,
                            (key[0], key[1], json.dumps(rec.to_dict(), ensure_ascii=False)),
                        )

            updated_entries: list[dict[str, Any]] = []
            for entry in self._verdict_log:
                if entry["triage_state"] != "OPEN":
                    continue
                matches_id = verdict_id is not None and entry["verdict_id"] == verdict_id
                matches_pair = any(
                    normalize_entity_key(iss.get("subject") or "") == key[0]
                    and normalize_predicate_key(iss.get("predicate") or "") == key[1]
                    for iss in entry["verdict"].get("issues", [])
                )
                if matches_id or (verdict_id is None and matches_pair):
                    if action == "REJECT_TO_AGENT":
                        entry["triage_state"] = "REJECTED"
                    elif self._is_verdict_fully_resolved(entry["verdict"]):
                        entry["triage_state"] = "RESOLVED"
                    else:
                        # Still has other unresolved ESCALATE issues on the same verdict (F1)
                        entry["triage_state"] = "OPEN"

                    entry["resolution"] = res_payload
                    entry["resolved_at"] = now if entry["triage_state"] != "OPEN" else None
                    updated_entries.append(dict(entry))

                    if self._db_path is not None:
                        with self._db_session() as conn:
                            conn.execute(
                                """
                                UPDATE verdicts
                                SET triage_state = ?, resolution_json = ?, resolved_at = ?
                                WHERE seq = ?
                                """,
                                (
                                    entry["triage_state"],
                                    json.dumps(res_payload, ensure_ascii=False),
                                    entry["resolved_at"],
                                    entry["seq"],
                                ),
                            )
            if self._db_path is not None:
                self._last_db_sig = self._current_db_sig()

        return {
            "resolution": res_payload,
            "updated_verdicts": len(updated_entries),
        }

    def list_verdicts(self, *, triage_state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            items = list(reversed(self._verdict_log))
            if triage_state:
                items = [e for e in items if e["triage_state"] == triage_state]
            return [dict(e) for e in items[:limit]]

    def get_verdict_entry(self, verdict_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._sync_from_sqlite_if_changed()
            for entry in reversed(self._verdict_log):
                if entry["verdict_id"] == verdict_id:
                    return dict(entry)
            return None

    def get_metrics(self) -> dict[str, Any]:
        """Compute live verification and human escalation metrics."""
        with self._lock:
            self._sync_from_sqlite_if_changed()
            total = len(self._verdict_log)
            pass_count = sum(1 for e in self._verdict_log if e["status"] == "PASS")
            retry_count = sum(1 for e in self._verdict_log if e["status"] == "RETRY")
            escalate_count = sum(1 for e in self._verdict_log if e["status"] == "ESCALATE")
            open_escalations = sum(1 for e in self._verdict_log if e["triage_state"] == "OPEN")
            resolved_escalations = sum(
                1 for e in self._verdict_log if e["triage_state"] in {"RESOLVED", "REJECTED"}
            )
            # Do not count unrecovered RETRY loops as human savings when zero proposals have passed (F2)
            if total == 0 or pass_count == 0:
                saved_pct = 0.0
            else:
                autonomous_handled = pass_count + retry_count
                saved_pct = round((autonomous_handled / total) * 100.0, 1)

            return {
                "total_verifications": total,
                "auto_approved_pass": pass_count,
                "agent_self_retry": retry_count,
                "escalated_to_human": escalate_count,
                "open_escalations": open_escalations,
                "resolved_escalations": resolved_escalations,
                "human_supervision_saved_pct": saved_pct,
                "sources_count": len(self._sources),
                "facts_count": len(self._facts),
                "resolutions_count": len(self._resolutions),
            }

    def reset(self) -> None:
        """Clear all state (useful for demos and tests)."""
        with self._lock:
            self._sources.clear()
            self._facts.clear()
            self._fact_set.clear()
            self._facts_by_source.clear()
            self._facts_by_pred.clear()
            self._resolutions.clear()
            self._verdict_log.clear()
            if self._db_path is not None:
                with self._db_session() as conn:
                    conn.executescript(
                        "DELETE FROM sources; DELETE FROM facts; DELETE FROM resolutions; "
                        "DELETE FROM verdicts; DELETE FROM ledger_meta;"
                    )
                self._last_db_sig = self._current_db_sig()

    def verify(
        self,
        assertions: Iterable[Assertion | dict[str, Any]],
        policy: Policy | dict[str, Any] | None = None,
        *,
        as_of: str | None = None,
        agent_id: str = "agent",
        record: bool = False,
        signing_key: str | bytes | None = None,
    ) -> Verdict:
        from claimlayer.verify import verify

        verdict = verify(
            assertions=assertions,
            ledger=self,
            policy=policy,
            as_of=as_of,
            agent_id=agent_id,
            signing_key=signing_key or self.signing_key,
        )
        if record:
            self.record_verdict(verdict)
        return verdict
