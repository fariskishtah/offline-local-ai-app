import sqlite3
from datetime import datetime

DB_PATH = "chat_history.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT    NOT NULL DEFAULT 'New chat',
                model      TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role       TEXT    NOT NULL CHECK(role IN ('user','assistant')),
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            );
        """)

def create_session(model: str, title: str = "New chat") -> int:
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO sessions (title, model, created_at) VALUES (?, ?, ?)",
            (title, model, now),
        )
        return cursor.lastrowid

def get_all_sessions() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, model, created_at FROM sessions ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def save_message(session_id: int, role: str, content: str) -> None:
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )

def load_messages(session_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def rename_session(session_id: int, new_title: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (new_title, session_id),
        )

def delete_session(session_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

def auto_title_from_first_message(session_id: int, user_message: str) -> None:
    title = user_message.strip()[:40]
    if len(user_message) > 40:
        title += "…"
    rename_session(session_id, title)
def init_benchmarks_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                model TEXT,
                prompt_id TEXT,
                prompt_text TEXT,
                tokens_per_second REAL,
                time_to_first_token REAL,
                tokens_generated INTEGER,
                prompt_tokens INTEGER,
                generation_duration REAL,
                total_duration REAL,
                ram_used_gb REAL,
                ram_percent REAL,
                cpu_percent REAL,
                model_processor TEXT,
                num_ctx INTEGER,
                status TEXT,
                error_msg TEXT,
                created_at TEXT
            )
        """)
def init_benchmarks_table() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                tokens_per_second REAL,
                time_to_first_token REAL,
                tokens_generated INTEGER,
                prompt_tokens INTEGER,
                generation_duration REAL,
                total_duration REAL,
                ram_used_gb REAL,
                ram_percent REAL,
                cpu_percent REAL,
                model_processor TEXT,
                num_ctx INTEGER,
                status TEXT NOT NULL DEFAULT 'ok',
                error_msg TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_benchmark_result(result: dict) -> None:
    from datetime import datetime
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO benchmarks (
                run_id, model, prompt_id, prompt_text,
                tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, num_ctx, status, error_msg, created_at
            ) VALUES (
                :run_id, :model, :prompt_id, :prompt_text,
                :tokens_per_second, :time_to_first_token,
                :tokens_generated, :prompt_tokens,
                :generation_duration, :total_duration,
                :ram_used_gb, :ram_percent, :cpu_percent,
                :model_processor, :num_ctx, :status, :error_msg, :created_at
            )
        """, {**result, "created_at": now})


def get_benchmark_runs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                run_id,
                MIN(created_at) AS started_at,
                COUNT(DISTINCT model) AS models_tested,
                COUNT(*) AS total_rows,
                ROUND(AVG(tokens_per_second), 1) AS avg_tok_s
            FROM benchmarks
            WHERE status = 'ok'
            GROUP BY run_id
            ORDER BY started_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_benchmark_results(run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                model, prompt_id, prompt_text,
                tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, num_ctx, status, error_msg
            FROM benchmarks
            WHERE run_id = ?
            ORDER BY model, prompt_id
        """, (run_id,)).fetchall()
    return [dict(r) for r in rows]            
def init_benchmarks_table() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                tokens_per_second REAL,
                time_to_first_token REAL,
                tokens_generated INTEGER,
                prompt_tokens INTEGER,
                generation_duration REAL,
                total_duration REAL,
                ram_used_gb REAL,
                ram_percent REAL,
                cpu_percent REAL,
                model_processor TEXT,
                num_ctx INTEGER,
                status TEXT NOT NULL DEFAULT 'ok',
                error_msg TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_benchmark_result(result: dict) -> None:
    from datetime import datetime
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO benchmarks (
                run_id, model, prompt_id, prompt_text,
                tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, num_ctx, status, error_msg, created_at
            ) VALUES (
                :run_id, :model, :prompt_id, :prompt_text,
                :tokens_per_second, :time_to_first_token,
                :tokens_generated, :prompt_tokens,
                :generation_duration, :total_duration,
                :ram_used_gb, :ram_percent, :cpu_percent,
                :model_processor, :num_ctx, :status, :error_msg, :created_at
            )
        """, {**result, "created_at": now})


def get_benchmark_runs() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                run_id,
                MIN(created_at) AS started_at,
                COUNT(DISTINCT model) AS models_tested,
                COUNT(*) AS total_rows,
                ROUND(AVG(tokens_per_second), 1) AS avg_tok_s
            FROM benchmarks
            WHERE status = 'ok'
            GROUP BY run_id
            ORDER BY started_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_benchmark_results(run_id: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                model, prompt_id, prompt_text,
                tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, num_ctx, status, error_msg
            FROM benchmarks
            WHERE run_id = ?
            ORDER BY model, prompt_id
        """, (run_id,)).fetchall()
    return [dict(r) for r in rows]
# ── Append to the bottom of database.py ─────────────────────

def init_ctx_table() -> None:
    """
    Creates the ctx_benchmarks table if it doesn't exist.
    Stores one row per (model, num_ctx) pair per run.
    Safe to call every time ctx_tuner.py starts.
    """
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ctx_benchmarks (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                TEXT    NOT NULL,
                model                 TEXT    NOT NULL,
                num_ctx               INTEGER NOT NULL,
                tokens_per_second     REAL,
                time_to_first_token   REAL,
                tokens_generated      INTEGER,
                prompt_tokens         INTEGER,
                generation_duration   REAL,
                prompt_eval_duration  REAL,
                total_duration        REAL,
                ram_used_gb           REAL,
                ram_percent           REAL,
                cpu_percent           REAL,
                model_processor       TEXT,
                status                TEXT    NOT NULL DEFAULT 'ok',
                error_msg             TEXT,
                created_at            TEXT    NOT NULL
            )
        """)


def save_ctx_result(result: dict) -> None:
    """
    Inserts one ctx_benchmark row.
    `result` is a plain dict matching CtxResult.to_db_dict() from ctx_tuner.py.
    """
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO ctx_benchmarks (
                run_id, model, num_ctx,
                tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, prompt_eval_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, status, error_msg, created_at
            ) VALUES (
                :run_id, :model, :num_ctx,
                :tokens_per_second, :time_to_first_token,
                :tokens_generated, :prompt_tokens,
                :generation_duration, :prompt_eval_duration, :total_duration,
                :ram_used_gb, :ram_percent, :cpu_percent,
                :model_processor, :status, :error_msg, :created_at
            )
        """, {**result, "created_at": now})


def get_ctx_runs() -> list[dict]:
    """
    Returns one summary row per (run_id, model) pair, newest first.
    Used to populate the run selector in the Streamlit tab.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                run_id,
                model,
                MIN(created_at)                          AS started_at,
                COUNT(*)                                 AS ctx_values_tested,
                MIN(num_ctx)                             AS ctx_min,
                MAX(num_ctx)                             AS ctx_max
            FROM ctx_benchmarks
            WHERE status = 'ok'
            GROUP BY run_id, model
            ORDER BY started_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_ctx_results(run_id: str, model: str) -> list[dict]:
    """
    Returns all rows for a specific (run_id, model) pair,
    ordered by num_ctx ascending — ready to plot as a series.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                num_ctx, tokens_per_second, time_to_first_token,
                tokens_generated, prompt_tokens,
                generation_duration, prompt_eval_duration, total_duration,
                ram_used_gb, ram_percent, cpu_percent,
                model_processor, status, error_msg
            FROM ctx_benchmarks
            WHERE run_id = ? AND model = ?
            ORDER BY num_ctx ASC
        """, (run_id, model)).fetchall()
    return [dict(r) for r in rows]