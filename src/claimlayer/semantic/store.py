"""Append-only, project-isolated semantic records with content integrity checks."""
from __future__ import annotations
import contextlib
import json
from pathlib import Path
import sqlite3
import threading
from .models import canonical, digest


class _ManagedConnection(sqlite3.Connection):
    """SQLite connection that holds an explicit transaction across statements and closes on exit."""

    def _begin(self, mode: str = 'DEFERRED') -> None:
        self._tx_depth = 1
        self._closed = False
        upper = mode.upper()
        if upper not in {'DEFERRED', 'IMMEDIATE', 'EXCLUSIVE'}:
            raise ValueError(f'Unsupported SQLite transaction mode: {mode!r}')
        self._tx_mode = upper
        super().execute('PRAGMA busy_timeout=30000')
        super().execute(f'BEGIN {upper}')
        if upper == 'DEFERRED':
            # Pin the WAL MVCC read snapshot immediately so SELECTs cannot see concurrent mid-flight commits.
            super().execute('SELECT 1 FROM sqlite_master LIMIT 1').fetchall()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if getattr(self, '_tx_depth', 1) > 1:
            self._tx_depth -= 1
            return False
        try:
            if self.in_transaction:
                super().execute('ROLLBACK' if exc_type is not None else 'COMMIT')
        finally:
            if not getattr(self, '_closed', False):
                self._closed = True
                super().close()
        return False


class VerificationStore:
    def __init__(self, path: str | Path, project_id: str):
        self.path = Path(path)
        if str(self.path) != ':memory:':
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        self._local = threading.local()
        init_db = sqlite3.connect(self.path, timeout=30)
        try:
            if str(self.path) != ':memory:':
                init_db.execute('PRAGMA journal_mode=WAL')
                init_db.execute('PRAGMA synchronous=NORMAL')
            init_db.execute('PRAGMA busy_timeout=30000')
            init_db.executescript('''
                CREATE TABLE IF NOT EXISTS semantic_records (
                  project_id TEXT NOT NULL, kind TEXT NOT NULL, record_id TEXT NOT NULL,
                  payload TEXT NOT NULL, checksum TEXT NOT NULL,
                  PRIMARY KEY(project_id,kind,record_id));
                CREATE TABLE IF NOT EXISTS semantic_heads (
                  project_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                  record_id TEXT NOT NULL, PRIMARY KEY(project_id,kind,name));
            ''')
            init_db.commit()
        finally:
            init_db.close()

    def connect(self, mode: str = 'DEFERRED') -> _ManagedConnection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None, factory=_ManagedConnection)
        try:
            conn._begin(mode)
        except Exception:
            conn.close()
            raise
        return conn

    @contextlib.contextmanager
    def _session(self, mode: str = 'DEFERRED'):
        tx_conn = getattr(self._local, 'tx_conn', None)
        if tx_conn is not None:
            if mode.upper() == 'IMMEDIATE' and tx_conn._tx_mode == 'DEFERRED':
                raise RuntimeError('Cannot enter an IMMEDIATE write session inside a DEFERRED read transaction')
            tx_conn._tx_depth += 1
            try:
                yield tx_conn
            finally:
                tx_conn._tx_depth -= 1
            return
        with self.connect(mode=mode) as db:
            yield db

    @contextlib.contextmanager
    def transaction(self, mode: str = 'IMMEDIATE'):
        """Execute multiple store operations atomically inside one SQLite transaction."""
        tx_conn = getattr(self._local, 'tx_conn', None)
        if tx_conn is not None:
            if mode.upper() == 'IMMEDIATE' and tx_conn._tx_mode == 'DEFERRED':
                raise RuntimeError('Cannot nest an IMMEDIATE transaction inside an active DEFERRED transaction')
            tx_conn._tx_depth += 1
            try:
                yield tx_conn
            finally:
                tx_conn._tx_depth -= 1
            return
        with self.connect(mode=mode) as db:
            self._local.tx_conn = db
            try:
                yield db
            finally:
                self._local.tx_conn = None

    def put(self, kind, record_id, payload, *, head=None):
        encoded = canonical(payload)
        checksum = digest(payload)
        with self._session(mode='IMMEDIATE') as db:
            old = db.execute('SELECT checksum FROM semantic_records WHERE project_id=? AND kind=? AND record_id=?',
                             (self.project_id, kind, record_id)).fetchone()
            if old and old[0] != checksum:
                raise ValueError('Cannot overwrite an immutable semantic record')
            db.execute('INSERT OR IGNORE INTO semantic_records VALUES (?,?,?,?,?)',
                       (self.project_id, kind, record_id, encoded, checksum))
            if head is not None:
                db.execute('INSERT OR REPLACE INTO semantic_heads VALUES (?,?,?,?)',
                           (self.project_id, kind, head, record_id))

    def get(self, kind, record_id):
        with self._session(mode='DEFERRED') as db:
            row = db.execute('SELECT payload,checksum FROM semantic_records WHERE project_id=? AND kind=? AND record_id=?',
                             (self.project_id, kind, record_id)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if digest(payload) != row[1]:
            raise ValueError('Corrupted semantic record')
        return payload

    def head(self, kind, name):
        with self._session(mode='DEFERRED') as db:
            row = db.execute('SELECT record_id FROM semantic_heads WHERE project_id=? AND kind=? AND name=?',
                             (self.project_id, kind, name)).fetchone()
            if row is None:
                return (None, None)
            rec_row = db.execute(
                'SELECT payload,checksum FROM semantic_records WHERE project_id=? AND kind=? AND record_id=?',
                (self.project_id, kind, row[0]),
            ).fetchone()
        if rec_row is None:
            return (row[0], None)
        payload = json.loads(rec_row[0])
        if digest(payload) != rec_row[1]:
            raise ValueError('Corrupted semantic record')
        return (row[0], payload)

    def heads(self, kind):
        with self._session(mode='DEFERRED') as db:
            rows = db.execute(
                'SELECT h.name, h.record_id, r.payload, r.checksum '
                'FROM semantic_heads h '
                'JOIN semantic_records r ON r.project_id=h.project_id AND r.kind=h.kind AND r.record_id=h.record_id '
                'WHERE h.project_id=? AND h.kind=? ORDER BY h.name',
                (self.project_id, kind),
            ).fetchall()
        out = []
        for name, rid, raw_payload, checksum in rows:
            payload = json.loads(raw_payload)
            if digest(payload) != checksum:
                raise ValueError('Corrupted semantic record')
            out.append((name, rid, payload))
        return out

    def by_kind(self, kind):
        with self._session(mode='DEFERRED') as db:
            rows = db.execute(
                'SELECT record_id, payload, checksum FROM semantic_records '
                'WHERE project_id=? AND kind=? ORDER BY rowid',
                (self.project_id, kind),
            ).fetchall()
        out = []
        for rid, raw_payload, checksum in rows:
            payload = json.loads(raw_payload)
            if digest(payload) != checksum:
                raise ValueError('Corrupted semantic record')
            out.append((rid, payload))
        return out

    def snapshot(self):
        """Read all current heads, document revisions, and rejections in one SQLite snapshot transaction."""
        with self._session(mode='DEFERRED') as db:
            head_rows = db.execute(
                'SELECT h.kind, h.name, h.record_id, r.payload, r.checksum '
                'FROM semantic_heads h '
                'JOIN semantic_records r ON r.project_id=h.project_id AND r.kind=h.kind AND r.record_id=h.record_id '
                'WHERE h.project_id=? ORDER BY h.kind, h.name',
                (self.project_id,),
            ).fetchall()
            doc_rows = db.execute(
                'SELECT record_id, payload, checksum FROM semantic_records '
                'WHERE project_id=? AND kind=? ORDER BY rowid',
                (self.project_id, 'document'),
            ).fetchall()
            rej_rows = db.execute(
                'SELECT record_id, payload, checksum FROM semantic_records '
                'WHERE project_id=? AND kind=? ORDER BY rowid',
                (self.project_id, 'rejection'),
            ).fetchall()

        heads_by_kind = {}
        for kind, name, rid, raw_payload, checksum in head_rows:
            payload = json.loads(raw_payload)
            if digest(payload) != checksum:
                raise ValueError('Corrupted semantic record')
            heads_by_kind.setdefault(kind, []).append((name, rid, payload))

        document_revisions = {}
        for rev, raw_payload, checksum in doc_rows:
            payload = json.loads(raw_payload)
            if digest(payload) != checksum:
                raise ValueError('Corrupted semantic record')
            document_revisions[rev] = payload

        rejections = []
        for rid, raw_payload, checksum in rej_rows:
            payload = json.loads(raw_payload)
            if digest(payload) != checksum:
                raise ValueError('Corrupted semantic record')
            rejections.append((rid, payload))

        return {
            'heads': heads_by_kind,
            'document_revisions': document_revisions,
            'rejections': rejections,
        }

    def legacy_present(self):
        with self._session(mode='DEFERRED') as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'documents', 'claims'} <= tables:
                return False
            return db.execute('SELECT 1 FROM documents WHERE project_id=? LIMIT 1', (self.project_id,)).fetchone() is not None
