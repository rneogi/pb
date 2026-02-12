"""
Database Schema and Operations
==============================
SQLite database for metadata, artifacts, chunks, and Phase 2 tables.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "db" / "observer.sqlite"


def get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session():
    """Context manager for database sessions."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Initialize database schema with all tables."""
    with db_session() as conn:
        cursor = conn.cursor()

        # =====================================================================
        # Phase 1 Tables
        # =====================================================================

        # Artifacts table - stores metadata for each crawled document
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT UNIQUE NOT NULL,  -- sha256 hash of canonical_url + content_hash
                canonical_url TEXT NOT NULL,
                content_hash TEXT NOT NULL,  -- sha256 of normalized content
                source_name TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                domain TEXT,
                title TEXT,
                url TEXT NOT NULL,
                week TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                published_at TEXT,
                fetch_mode TEXT,
                http_status INTEGER,
                etag TEXT,
                last_modified TEXT,
                raw_path TEXT,
                clean_path TEXT,
                meta_path TEXT,
                main_text_length INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(canonical_url, week)
            )
        """)

        # Chunks table - stores chunked text for retrieval
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,  -- artifact_id + chunk_index
                artifact_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                start_char INTEGER,
                end_char INTEGER,
                token_count_approx INTEGER,
                week TEXT NOT NULL,
                source_kind TEXT,
                source_name TEXT,
                canonical_url TEXT,
                title TEXT,
                published_at TEXT,
                retrieved_at TEXT,
                embedded BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
            )
        """)

        # Crawl state table - tracks incremental crawl state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS crawl_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_url TEXT UNIQUE NOT NULL,
                last_content_hash TEXT,
                last_etag TEXT,
                last_modified TEXT,
                last_crawled_at TEXT,
                last_week TEXT,
                crawl_count INTEGER DEFAULT 0,
                consecutive_unchanged INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Delta history - tracks changes over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS delta_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week TEXT NOT NULL,
                canonical_url TEXT NOT NULL,
                artifact_id TEXT,
                delta_type TEXT NOT NULL,  -- new, changed, stale, unchanged
                previous_content_hash TEXT,
                new_content_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Evaluation results - bucket assignments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                week TEXT NOT NULL,
                bucket TEXT NOT NULL,  -- deal_signal, investor_graph_change, etc.
                confidence REAL,
                rationale TEXT,
                keywords_matched TEXT,  -- JSON array
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
            )
        """)

        # =====================================================================
        # Phase 2 Tables (Placeholders - empty for now)
        # =====================================================================

        # Entities table - for entity resolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT UNIQUE NOT NULL,
                entity_type TEXT NOT NULL,  -- company, investor, person, fund
                canonical_name TEXT NOT NULL,
                aliases TEXT,  -- JSON array of alternative names
                domain TEXT,
                website TEXT,
                description TEXT,
                first_seen_week TEXT,
                last_seen_week TEXT,
                confidence REAL,
                metadata TEXT,  -- JSON object for additional data
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Claims table - extracted claims from artifacts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT UNIQUE NOT NULL,
                claim_type TEXT NOT NULL,  -- funding_round, acquisition, hire, etc.
                subject_entity_id TEXT,
                object_entity_id TEXT,
                predicate TEXT,
                value TEXT,
                unit TEXT,
                confidence REAL,
                source_artifact_id TEXT,
                source_chunk_id TEXT,
                evidence_text TEXT,
                extracted_at TEXT,
                validated BOOLEAN DEFAULT FALSE,
                validation_notes TEXT,
                week TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (object_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (source_artifact_id) REFERENCES artifacts(artifact_id),
                FOREIGN KEY (source_chunk_id) REFERENCES chunks(chunk_id)
            )
        """)

        # Deal records table - assembled deal records with state machine
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deal_id TEXT UNIQUE NOT NULL,
                deal_type TEXT,  -- seed, series_a, series_b, acquisition, etc.
                company_entity_id TEXT,
                amount REAL,
                currency TEXT DEFAULT 'USD',
                valuation REAL,
                lead_investor_entity_id TEXT,
                investors TEXT,  -- JSON array of entity_ids
                announced_date TEXT,
                closed_date TEXT,
                state TEXT DEFAULT 'draft',  -- draft, pending_review, confirmed, stale
                confidence REAL,
                supporting_claims TEXT,  -- JSON array of claim_ids
                notes TEXT,
                first_seen_week TEXT,
                last_updated_week TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (lead_investor_entity_id) REFERENCES entities(entity_id)
            )
        """)

        # Entity mentions - links entities to artifacts/chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                chunk_id TEXT,
                mention_text TEXT,
                context TEXT,
                position_start INTEGER,
                position_end INTEGER,
                confidence REAL,
                week TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id),
                FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_week ON artifacts(week)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_source_kind ON artifacts(source_kind)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_canonical_url ON artifacts(canonical_url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_artifact_id ON chunks(artifact_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_week ON chunks(week)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_week ON evaluations(week)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_bucket ON evaluations(bucket)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_crawl_state_url ON crawl_state(canonical_url)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_deal_records_state ON deal_records(state)")

        conn.commit()


# =========================================================================
# Artifact Operations
# =========================================================================

def upsert_artifact(artifact: Dict[str, Any]) -> str:
    """Insert or update an artifact record."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO artifacts (
                artifact_id, canonical_url, content_hash, source_name, source_kind,
                domain, title, url, week, retrieved_at, published_at, fetch_mode,
                http_status, etag, last_modified, raw_path, clean_path, meta_path,
                main_text_length, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            artifact["artifact_id"],
            artifact["canonical_url"],
            artifact["content_hash"],
            artifact["source_name"],
            artifact["source_kind"],
            artifact.get("domain"),
            artifact.get("title"),
            artifact["url"],
            artifact["week"],
            artifact["retrieved_at"],
            artifact.get("published_at"),
            artifact.get("fetch_mode"),
            artifact.get("http_status"),
            artifact.get("etag"),
            artifact.get("last_modified"),
            artifact.get("raw_path"),
            artifact.get("clean_path"),
            artifact.get("meta_path"),
            artifact.get("main_text_length"),
            datetime.utcnow().isoformat()
        ))
        return artifact["artifact_id"]


def get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Get an artifact by ID."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_artifacts_by_week(week: str) -> List[Dict[str, Any]]:
    """Get all artifacts for a given week."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE week = ?", (week,))
        return [dict(row) for row in cursor.fetchall()]


# =========================================================================
# Chunk Operations
# =========================================================================

def insert_chunks(chunks: List[Dict[str, Any]]):
    """Insert multiple chunks."""
    with db_session() as conn:
        cursor = conn.cursor()
        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, artifact_id, chunk_index, text, start_char, end_char,
                    token_count_approx, week, source_kind, source_name, canonical_url,
                    title, published_at, retrieved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["chunk_id"],
                chunk["artifact_id"],
                chunk["chunk_index"],
                chunk["text"],
                chunk.get("start_char"),
                chunk.get("end_char"),
                chunk.get("token_count_approx"),
                chunk["week"],
                chunk.get("source_kind"),
                chunk.get("source_name"),
                chunk.get("canonical_url"),
                chunk.get("title"),
                chunk.get("published_at"),
                chunk.get("retrieved_at")
            ))


def get_chunks_by_artifact(artifact_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for an artifact."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chunks WHERE artifact_id = ? ORDER BY chunk_index",
            (artifact_id,)
        )
        return [dict(row) for row in cursor.fetchall()]


def get_chunks_by_week(week: str, embedded_only: bool = False) -> List[Dict[str, Any]]:
    """Get all chunks for a week."""
    with db_session() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM chunks WHERE week = ?"
        if embedded_only:
            query += " AND embedded = TRUE"
        cursor.execute(query, (week,))
        return [dict(row) for row in cursor.fetchall()]


def mark_chunks_embedded(chunk_ids: List[str]):
    """Mark chunks as embedded."""
    with db_session() as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?" for _ in chunk_ids])
        cursor.execute(
            f"UPDATE chunks SET embedded = TRUE WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )


# =========================================================================
# Crawl State Operations
# =========================================================================

def get_crawl_state(canonical_url: str) -> Optional[Dict[str, Any]]:
    """Get crawl state for a URL."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM crawl_state WHERE canonical_url = ?",
            (canonical_url,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def update_crawl_state(canonical_url: str, content_hash: str, week: str,
                       etag: Optional[str] = None, last_modified: Optional[str] = None,
                       unchanged: bool = False):
    """Update crawl state for a URL."""
    with db_session() as conn:
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        existing = get_crawl_state(canonical_url)
        if existing:
            consecutive = existing["consecutive_unchanged"] + 1 if unchanged else 0
            cursor.execute("""
                UPDATE crawl_state SET
                    last_content_hash = ?,
                    last_etag = ?,
                    last_modified = ?,
                    last_crawled_at = ?,
                    last_week = ?,
                    crawl_count = crawl_count + 1,
                    consecutive_unchanged = ?,
                    updated_at = ?
                WHERE canonical_url = ?
            """, (content_hash, etag, last_modified, now, week, consecutive, now, canonical_url))
        else:
            cursor.execute("""
                INSERT INTO crawl_state (
                    canonical_url, last_content_hash, last_etag, last_modified,
                    last_crawled_at, last_week, crawl_count, consecutive_unchanged
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 0)
            """, (canonical_url, content_hash, etag, last_modified, now, week))


# =========================================================================
# Evaluation Operations
# =========================================================================

def insert_evaluation(artifact_id: str, week: str, bucket: str,
                      confidence: float = 1.0, rationale: str = "",
                      keywords_matched: List[str] = None):
    """Insert an evaluation result."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluations (artifact_id, week, bucket, confidence, rationale, keywords_matched)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            artifact_id, week, bucket, confidence, rationale,
            json.dumps(keywords_matched or [])
        ))


def get_evaluations_by_week(week: str) -> List[Dict[str, Any]]:
    """Get all evaluations for a week."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluations WHERE week = ?", (week,))
        return [dict(row) for row in cursor.fetchall()]


# =========================================================================
# Delta History Operations
# =========================================================================

def insert_delta(week: str, canonical_url: str, delta_type: str,
                 artifact_id: str = None, previous_hash: str = None,
                 new_hash: str = None):
    """Insert a delta history record."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO delta_history (
                week, canonical_url, artifact_id, delta_type,
                previous_content_hash, new_content_hash
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (week, canonical_url, artifact_id, delta_type, previous_hash, new_hash))


def get_deltas_by_week(week: str) -> List[Dict[str, Any]]:
    """Get all deltas for a week."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM delta_history WHERE week = ?", (week,))
        return [dict(row) for row in cursor.fetchall()]


if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print(f"Database initialized at {DB_PATH}")
