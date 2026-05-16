import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sessions.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with _conn() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                province TEXT DEFAULT '辽宁',
                category TEXT DEFAULT '',
                score INTEGER DEFAULT 0,
                rank INTEGER DEFAULT 0,
                preferred_majors TEXT DEFAULT '[]',
                avoid_majors TEXT DEFAULT '[]',
                preferred_regions TEXT DEFAULT '[]',
                avoid_regions TEXT DEFAULT '[]',
                target_level TEXT DEFAULT '',
                stage TEXT DEFAULT 'collecting',
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            )
        """)


def row_to_profile(row: tuple) -> dict:
    """将 DB 行转为 user_profile dict。列顺序与表定义一致。"""
    if not row:
        return None
    return {
        "province": row[1] or "辽宁",
        "category": row[2] or "",
        "score": row[3] or 0,
        "rank": row[4] or 0,
        "preferred_majors": json.loads(row[5]) if row[5] else [],
        "avoid_majors": json.loads(row[6]) if row[6] else [],
        "preferred_regions": json.loads(row[7]) if row[7] else [],
        "avoid_regions": json.loads(row[8]) if row[8] else [],
        "target_level": row[9] or "",
    }


def load_session(session_id: str) -> dict | None:
    """加载 session 的 user_profile，不存在返回 None。"""
    with _conn() as db:
        row = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if not row:
        return None
    profile = row_to_profile(row)
    profile["_stage"] = row[10] or "collecting"
    return profile


def save_session(session_id: str, profile: dict, stage: str = "collecting"):
    """插入或更新 session 的关键字段。"""
    with _conn() as db:
        db.execute("""
            INSERT INTO sessions (session_id, province, category, score, rank,
                preferred_majors, avoid_majors, preferred_regions, avoid_regions,
                target_level, stage, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            ON CONFLICT(session_id) DO UPDATE SET
                province=excluded.province,
                category=excluded.category,
                score=excluded.score,
                rank=excluded.rank,
                preferred_majors=excluded.preferred_majors,
                avoid_majors=excluded.avoid_majors,
                preferred_regions=excluded.preferred_regions,
                avoid_regions=excluded.avoid_regions,
                target_level=excluded.target_level,
                stage=excluded.stage,
                updated_at=datetime('now','localtime')
        """, (
            session_id,
            profile.get("province", "辽宁") or "辽宁",
            profile.get("category", "") or "",
            profile.get("score", 0) or 0,
            profile.get("rank", 0) or 0,
            json.dumps(profile.get("preferred_majors", []) or [], ensure_ascii=False),
            json.dumps(profile.get("avoid_majors", []) or [], ensure_ascii=False),
            json.dumps(profile.get("preferred_regions", []) or [], ensure_ascii=False),
            json.dumps(profile.get("avoid_regions", []) or [], ensure_ascii=False),
            profile.get("target_level", "") or "",
            stage,
        ))


def delete_session(session_id: str):
    with _conn() as db:
        db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
