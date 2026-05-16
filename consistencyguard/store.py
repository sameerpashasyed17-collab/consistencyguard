import sqlite3
import json
import os
from datetime import datetime
from consistencyguard.models import LLMCall, ConsistencyViolation


def get_db_path() -> str:
    return os.getenv("DB_PATH", "consistencyguard.db")


def init_db() -> None:
    """Create tables if they don't exist."""
    with sqlite3.connect(get_db_path()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL,
                agent_id TEXT DEFAULT 'default',
                timestamp TEXT NOT NULL,
                prompt_embedding TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id_new INTEGER,
                call_id_ref INTEGER,
                prompt_similarity REAL,
                response_divergence REAL,
                severity TEXT,
                new_prompt TEXT,
                new_response TEXT,
                ref_response TEXT,
                explanation TEXT,
                agent_id TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()


def save_call(call: LLMCall) -> int:
    """Save a call, return its rowid."""
    with sqlite3.connect(get_db_path()) as conn:
        cursor = conn.execute(
            """INSERT INTO llm_calls
               (prompt, response, model, agent_id, timestamp, prompt_embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                call.prompt,
                call.response,
                call.model,
                call.agent_id or "default",
                call.timestamp.isoformat(),
                json.dumps(call.prompt_embedding),
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_all_calls() -> list[LLMCall]:
    """Return all stored calls with embeddings."""
    with sqlite3.connect(get_db_path()) as conn:
        rows = conn.execute(
            "SELECT id, prompt, response, model, agent_id, "
            "timestamp, prompt_embedding FROM llm_calls"
        ).fetchall()
    result = []
    for row in rows:
        result.append(LLMCall(
            id=row[0],
            prompt=row[1],
            response=row[2],
            model=row[3],
            agent_id=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            prompt_embedding=json.loads(row[6]),
        ))
    return result


def save_violation(v: ConsistencyViolation) -> None:
    with sqlite3.connect(get_db_path()) as conn:
        conn.execute(
            """INSERT INTO violations
               (call_id_new, call_id_ref, prompt_similarity,
                response_divergence, severity, new_prompt,
                new_response, ref_response, explanation,
                agent_id, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                v.call_id_new, v.call_id_ref,
                v.prompt_similarity, v.response_divergence,
                v.severity.value, v.new_prompt,
                v.new_response, v.ref_response,
                v.explanation, v.agent_id,
                v.timestamp.isoformat(),
            )
        )
        conn.commit()


def get_all_violations() -> list[dict]:
    with sqlite3.connect(get_db_path()) as conn:
        rows = conn.execute(
            "SELECT * FROM violations ORDER BY timestamp DESC"
        ).fetchall()
    cols = [
        "id", "call_id_new", "call_id_ref", "prompt_similarity",
        "response_divergence", "severity", "new_prompt",
        "new_response", "ref_response", "explanation",
        "agent_id", "timestamp"
    ]
    return [dict(zip(cols, row)) for row in rows]


def get_stats(hours: int = 24) -> dict:
    with sqlite3.connect(get_db_path()) as conn:
        total_calls = conn.execute(
            "SELECT COUNT(*) FROM llm_calls"
        ).fetchone()[0]
        total_violations = conn.execute(
            "SELECT COUNT(*) FROM violations"
        ).fetchone()[0]
        by_severity = conn.execute(
            "SELECT severity, COUNT(*) FROM violations "
            "GROUP BY severity"
        ).fetchall()
    sev_map = dict(by_severity)
    return {
        "total_calls": total_calls,
        "total_violations": total_violations,
        "critical": sev_map.get("critical", 0),
        "warning": sev_map.get("warning", 0),
        "info": sev_map.get("info", 0),
    }


def get_trend_data(hours: int = 24) -> list[dict]:
    """
    Returns violation counts grouped into hourly buckets for the last N hours.
    Each dict: {"bucket": "05-16 14:00", "count": 3, "critical": 1, "warning": 2, "info": 0}
    """
    with sqlite3.connect(get_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT
                strftime('%m-%d %H:00', timestamp) AS bucket,
                COUNT(*) AS count,
                SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical,
                SUM(CASE WHEN severity = 'warning' THEN 1 ELSE 0 END) AS warning,
                SUM(CASE WHEN severity = 'info' THEN 1 ELSE 0 END) AS info
            FROM violations
            WHERE timestamp >= datetime('now', ?)
            GROUP BY bucket
            ORDER BY bucket ASC
            """,
            (f"-{hours} hours",),
        ).fetchall()
    return [
        {"bucket": r[0], "count": r[1], "critical": r[2], "warning": r[3], "info": r[4]}
        for r in rows
    ]


def get_agent_stats(hours: int = 24) -> list[dict]:
    """Per-agent call and violation counts for the last N hours."""
    with sqlite3.connect(get_db_path()) as conn:
        call_counts = dict(conn.execute(
            "SELECT agent_id, COUNT(*) FROM llm_calls GROUP BY agent_id"
        ).fetchall())

        viol_rows = conn.execute(
            """
            SELECT
                agent_id,
                COUNT(*) AS total,
                SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical,
                SUM(CASE WHEN severity = 'warning' THEN 1 ELSE 0 END) AS warning,
                SUM(CASE WHEN severity = 'info' THEN 1 ELSE 0 END) AS info,
                MAX(timestamp) AS last_violation
            FROM violations
            WHERE timestamp >= datetime('now', ?)
            GROUP BY agent_id
            """,
            (f"-{hours} hours",),
        ).fetchall()

    result = []
    for row in viol_rows:
        agent_id, total, critical, warning, info, last_v = row
        calls = call_counts.get(agent_id, 0)
        result.append({
            "agent_id": agent_id,
            "total_calls": calls,
            "total_violations": total,
            "critical": critical,
            "warning": warning,
            "info": info,
            "violation_rate": round(total / calls * 100, 1) if calls > 0 else 0.0,
            "last_violation": last_v,
        })
    return sorted(result, key=lambda x: x["total_violations"], reverse=True)


def get_violations_filtered(
    agent_id: str = None,
    severity: str = None,
    since_hours: int = None,
    limit: int = 100,
) -> list[dict]:
    """Filtered version of get_all_violations with optional agent/severity/time filters."""
    conditions: list[str] = []
    params: list = []

    if agent_id:
        conditions.append("agent_id = ?")
        params.append(agent_id)
    if severity:
        conditions.append("severity = ?")
        params.append(severity.lower())
    if since_hours:
        conditions.append("timestamp >= datetime('now', ?)")
        params.append(f"-{since_hours} hours")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    with sqlite3.connect(get_db_path()) as conn:
        rows = conn.execute(
            f"SELECT * FROM violations {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

    cols = [
        "id", "call_id_new", "call_id_ref", "prompt_similarity",
        "response_divergence", "severity", "new_prompt",
        "new_response", "ref_response", "explanation",
        "agent_id", "timestamp",
    ]
    return [dict(zip(cols, row)) for row in rows]
