import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    accuracy REAL NOT NULL,
    precision_macro REAL NOT NULL,
    recall_macro REAL NOT NULL,
    f1_macro REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path):
    with connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def seconds_since_last_submission(db_path, team_name):
    """None if this team has never submitted before."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT created_at FROM submissions WHERE team_name = ? ORDER BY id DESC LIMIT 1",
            (team_name,),
        ).fetchone()
    if row is None:
        return None
    last = datetime.fromisoformat(row["created_at"])
    return (datetime.now(timezone.utc) - last).total_seconds()


def record_submission(db_path, team_name, scores):
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO submissions "
            "(team_name, accuracy, precision_macro, recall_macro, f1_macro, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                team_name,
                scores["accuracy"],
                scores["precision"],
                scores["recall"],
                scores["f1_score"],
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def leaderboard(db_path):
    """Each team's best submission (by accuracy, then macro F1), ranked."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT team_name, accuracy, precision_macro, recall_macro, f1_macro, created_at
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY team_name
                           ORDER BY accuracy DESC, f1_macro DESC, id ASC
                       ) AS rn
                FROM submissions
            )
            WHERE rn = 1
            ORDER BY accuracy DESC, f1_macro DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]
