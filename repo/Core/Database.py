"""Core/Database.py — Gestão da base de dados SQLite"""
import sqlite3, json
from contextlib import contextmanager
from Config.Settings import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
                name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, rul INTEGER, status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, action TEXT, details TEXT, ip TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS kb_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, filename TEXT, chunks INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        """)

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
