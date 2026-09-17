"""Personal library: saved topics and papers, stored in a local SQLite database."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from papermesh.cache import DATA_DIR

DEFAULT_DB_PATH = DATA_DIR / "papermesh.db"
STATUSES = ("to_read", "reading", "read")

# Bump SCHEMA_VERSION and add a step to MIGRATIONS whenever the schema changes.
SCHEMA_VERSION = 1
MIGRATIONS = {
    1: """
        CREATE TABLE topics (
            query       TEXT PRIMARY KEY,   -- normalized: stripped + lowercased
            display     TEXT NOT NULL,      -- query as the user typed it
            summary     TEXT,
            saved_at    TEXT NOT NULL
        );
        CREATE TABLE papers (
            key             TEXT PRIMARY KEY,   -- "arxiv:<id>" or "s2:<id>"
            title           TEXT NOT NULL,
            authors         TEXT NOT NULL,      -- JSON list
            year            INTEGER,
            link            TEXT NOT NULL,
            citation_count  INTEGER,
            source_query    TEXT,
            status          TEXT NOT NULL DEFAULT 'to_read'
                            CHECK (status IN ('to_read', 'reading', 'read')),
            note            TEXT NOT NULL DEFAULT '',
            saved_at        TEXT NOT NULL
        );
    """,
}


def paper_key(arxiv_id: str | None, s2_id: str | None) -> str:
    """Stable library key: prefer the arXiv ID, fall back to the Semantic Scholar ID."""
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    if s2_id:
        return f"s2:{s2_id}"
    raise ValueError("A paper needs an arXiv or Semantic Scholar ID to be saved.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_query(query: str) -> str:
    return " ".join(query.split()).lower()


class Library:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._migrate(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per operation: Streamlit reruns scripts on different threads.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for target in range(version + 1, SCHEMA_VERSION + 1):
            conn.executescript(MIGRATIONS[target])
            conn.execute(f"PRAGMA user_version = {target}")

    # ---------- Topics ----------

    def save_topic(self, query: str, summary: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO topics (query, display, summary, saved_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (query) DO UPDATE SET display = excluded.display, summary = excluded.summary
                """,
                (_normalize_query(query), " ".join(query.split()), summary, _now()),
            )

    def remove_topic(self, query: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM topics WHERE query = ?", (_normalize_query(query),))

    def is_topic_saved(self, query: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM topics WHERE query = ?", (_normalize_query(query),)).fetchone()
        return row is not None

    def topic_summary(self, query: str) -> str | None:
        """The overview stored with a saved topic, if any."""
        with self._connect() as conn:
            row = conn.execute("SELECT summary FROM topics WHERE query = ?", (_normalize_query(query),)).fetchone()
        return row["summary"] if row else None

    def list_topics(self) -> list[dict]:
        """Saved topics, newest first, with how many saved papers came from each."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.query, t.display, t.summary, t.saved_at,
                       (SELECT COUNT(*) FROM papers p WHERE p.source_query = t.query) AS paper_count
                FROM topics t
                ORDER BY t.saved_at DESC, t.rowid DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    # ---------- Papers ----------

    def save_paper(
        self,
        key: str,
        title: str,
        link: str,
        authors: list[str] | None = None,
        year: int | None = None,
        citation_count: int | None = None,
        source_query: str | None = None,
    ) -> None:
        """Save a paper, or refresh its metadata if already saved (status and note are kept)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (key, title, authors, year, link, citation_count, source_query, saved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (key) DO UPDATE SET
                    title = excluded.title,
                    authors = CASE WHEN excluded.authors = '[]' THEN papers.authors ELSE excluded.authors END,
                    year = COALESCE(excluded.year, papers.year),
                    link = excluded.link,
                    citation_count = COALESCE(excluded.citation_count, papers.citation_count)
                """,
                (
                    key,
                    title,
                    json.dumps(authors or []),
                    year,
                    link,
                    citation_count,
                    _normalize_query(source_query) if source_query else None,
                    _now(),
                ),
            )

    def remove_paper(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM papers WHERE key = ?", (key,))

    def saved_paper_keys(self) -> set[str]:
        with self._connect() as conn:
            return {row["key"] for row in conn.execute("SELECT key FROM papers")}

    def list_papers(self, status: str | None = None) -> list[dict]:
        """Saved papers, newest first, optionally filtered by reading status."""
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}; expected one of {STATUSES}.")
        query = "SELECT * FROM papers"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        with self._connect() as conn:
            rows = conn.execute(query + " ORDER BY saved_at DESC, rowid DESC", params).fetchall()
        return [{**dict(row), "authors": json.loads(row["authors"])} for row in rows]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM papers GROUP BY status").fetchall()
        counts = {status: 0 for status in STATUSES}
        counts.update({row["status"]: row["n"] for row in rows})
        return counts

    def set_status(self, key: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}; expected one of {STATUSES}.")
        with self._connect() as conn:
            conn.execute("UPDATE papers SET status = ? WHERE key = ?", (status, key))

    def set_note(self, key: str, note: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE papers SET note = ? WHERE key = ?", (note, key))
