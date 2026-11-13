"""Append-only, project-isolated semantic records with content integrity checks."""
from __future__ import annotations
import json
from pathlib import Path
import sqlite3
from .models import canonical, digest


class VerificationStore:
    def __init__(self, path: str | Path, project_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS semantic_records (
                  project_id TEXT NOT NULL, kind TEXT NOT NULL, record_id TEXT NOT NULL,
                  payload TEXT NOT NULL, checksum TEXT NOT NULL,
                  PRIMARY KEY(project_id,kind,record_id));
                CREATE TABLE IF NOT EXISTS semantic_heads (
                  project_id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT NOT NULL,
                  record_id TEXT NOT NULL, PRIMARY KEY(project_id,kind,name));
            ''')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def put(self, kind, record_id, payload, *, head=None):
        encoded = canonical(payload)
        checksum = digest(payload)
        with self.connect() as db:
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
        with self.connect() as db:
            row = db.execute('SELECT payload,checksum FROM semantic_records WHERE project_id=? AND kind=? AND record_id=?',
                             (self.project_id, kind, record_id)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if digest(payload) != row[1]:
            raise ValueError('Corrupted semantic record')
        return payload

    def head(self, kind, name):
        with self.connect() as db:
            row = db.execute('SELECT record_id FROM semantic_heads WHERE project_id=? AND kind=? AND name=?',
                             (self.project_id, kind, name)).fetchone()
        return (row[0], self.get(kind, row[0])) if row else (None, None)

    def heads(self, kind):
        with self.connect() as db:
            rows = db.execute('SELECT name,record_id FROM semantic_heads WHERE project_id=? AND kind=? ORDER BY name',
                              (self.project_id, kind)).fetchall()
        return [(name, rid, self.get(kind, rid)) for name, rid in rows]

    def legacy_present(self):
        with self.connect() as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'documents', 'claims'} <= tables:
                return False
            return db.execute('SELECT 1 FROM documents WHERE project_id=? LIMIT 1', (self.project_id,)).fetchone() is not None
