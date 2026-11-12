"""
Persistent storage for NEXUS AI.

Everything that must survive a server restart lives here:
  - user accounts and admin accounts
  - the AI name / tagline (branding)
  - the chunks (text + embedding) of every PDF an admin uploads, and which built-in
    PDFs an admin deleted, so the vector index can be rebuilt on start-up
  - a generated signing secret (when SECRET_KEY is not set)

Backends, chosen by the DATABASE_URL environment variable:
  - not set                       -> JSON files next to the app (the original behaviour; fine
                                     for local development, but wiped on free hosting)
  - postgresql://user:pw@host/db  -> hosted Postgres (Neon, Supabase, Render Postgres, ...)
  - sqlite:///nexus.db            -> a local SQLite file (handy for testing)

The rest of the app only uses the `storage` object at the bottom of this file.
"""

import json
import os
import secrets
import sqlite3
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_BRANDING = {"name": "NEXUS", "tagline": ""}
USER_CACHE_SECONDS = 30


# ====================================================================== file backend
class FileBackend:
    """Original JSON-file behaviour. Nothing is persisted across restarts on free hosting."""
    name = "files"
    persistent = False

    def __init__(self):
        self.users_file = os.path.join(BASE_DIR, "users.json")
        self.admins_file = os.path.join(BASE_DIR, "admins.json")
        self.branding_file = os.path.join(BASE_DIR, "branding.json")
        self._lock = threading.Lock()

    # -- helpers
    @staticmethod
    def _read(path, default):
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write(path, data):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    # -- users
    def get_user(self, username):
        return self._read(self.users_file, {}).get(username)

    def create_user(self, username, record):
        with self._lock:
            users = self._read(self.users_file, {})
            if username in users:
                return False
            users[username] = record
            self._write(self.users_file, users)
            return True

    def user_exists(self, username):
        return self.get_user(username) is not None

    # -- admins
    def get_admins(self):
        return self._read(self.admins_file, {})

    def create_admin(self, username, record):
        with self._lock:
            admins = self._read(self.admins_file, {})
            if username in admins:
                return False
            admins[username] = record
            self._write(self.admins_file, admins)
            return True

    # -- branding
    def get_branding(self):
        data = self._read(self.branding_file, {})
        return {
            "name": data.get("name") or DEFAULT_BRANDING["name"],
            "tagline": data.get("tagline", DEFAULT_BRANDING["tagline"]),
        }

    def set_branding(self, name, tagline):
        data = {"name": name, "tagline": tagline}
        self._write(self.branding_file, data)
        return data

    # -- settings (a generated secret cannot be kept without a database)
    def get_or_create_secret(self):
        return secrets.token_hex(32)

    # -- documents (index data is not persisted in this mode)
    def save_chunks(self, rows):
        pass

    def remove_chunks(self, source):
        pass

    def mark_deleted(self, source):
        pass

    def unmark_deleted(self, source):
        pass

    def list_deleted(self):
        return []

    def iter_chunks(self, batch_size=200):
        return iter(())


# ====================================================================== SQL backend
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        enrol TEXT NOT NULL DEFAULT '',
        course TEXT NOT NULL DEFAULT '',
        created BIGINT NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS admins (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'admin'
    )""",
    """CREATE TABLE IF NOT EXISTS branding (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        tagline TEXT NOT NULL DEFAULT ''
    )""",
    """CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS doc_chunks (
        chunk_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        page INTEGER NOT NULL DEFAULT 0,
        page_end INTEGER NOT NULL DEFAULT 0,
        course_code TEXT,
        body TEXT NOT NULL,
        embedding TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS doc_chunks_source_idx ON doc_chunks (source)",
    """CREATE TABLE IF NOT EXISTS deleted_sources (
        source TEXT PRIMARY KEY
    )""",
]

UPSERT_CHUNK = (
    "INSERT INTO doc_chunks (chunk_id, source, page, page_end, course_code, body, embedding) "
    "VALUES {values} "
    "ON CONFLICT (chunk_id) DO UPDATE SET source = EXCLUDED.source, page = EXCLUDED.page, "
    "page_end = EXCLUDED.page_end, course_code = EXCLUDED.course_code, "
    "body = EXCLUDED.body, embedding = EXCLUDED.embedding"
)


class SqlBackend:
    """Postgres (production) or SQLite (testing). One connection per process, re-opened
    after a fork or when the database closes an idle connection."""
    persistent = True

    def __init__(self, url):
        if url.startswith("postgres://"):  # some providers still hand out the old scheme
            url = "postgresql://" + url[len("postgres://"):]
        self.url = url
        self.is_sqlite = url.startswith("sqlite:")
        self.name = "sqlite" if self.is_sqlite else "postgres"
        self._conn = None
        self._pid = None
        self._lock = threading.RLock()
        self._ready_pid = None
        self._user_cache = {}

    # -- connection handling
    def _connect(self):
        if self.is_sqlite:
            path = self.url.split("sqlite:///", 1)[1] if "sqlite:///" in self.url else self.url[7:]
            if not os.path.isabs(path):
                path = os.path.join(BASE_DIR, path)
            return sqlite3.connect(path, check_same_thread=False, timeout=30)
        import psycopg2
        return psycopg2.connect(self.url, connect_timeout=10)

    def _get_conn(self):
        if self._conn is None or self._pid != os.getpid():
            self._conn = self._connect()
            self._pid = os.getpid()
        return self._conn

    def _reset(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    @staticmethod
    def _is_connection_error(exc):
        try:
            import psycopg2
            return isinstance(exc, (psycopg2.OperationalError, psycopg2.InterfaceError))
        except ImportError:
            return False

    def _fmt(self, sql):
        return sql.replace("%s", "?") if self.is_sqlite else sql

    def _ensure_schema(self):
        if self._ready_pid == os.getpid():
            return
        for statement in SCHEMA:
            try:
                self._run(statement, ready_check=False)
            except Exception as exc:  # two workers creating tables at once is harmless
                print(f"[storage] schema note: {exc}", flush=True)
        self._ready_pid = os.getpid()
        print(f"[storage] using {self.name} database", flush=True)

    def _run(self, sql, params=(), fetch=None, ready_check=True):
        if ready_check:
            self._ensure_schema()
        sql = self._fmt(sql)
        for attempt in (1, 2):
            with self._lock:
                conn = None
                try:
                    conn = self._get_conn()
                    cur = conn.cursor()
                    cur.execute(sql, params)
                    if fetch == "one":
                        result = cur.fetchone()
                    elif fetch == "all":
                        result = cur.fetchall()
                    else:
                        result = cur.rowcount
                    conn.commit()
                    cur.close()
                    return result
                except Exception as exc:
                    try:
                        if conn is not None:
                            conn.rollback()
                    except Exception:
                        pass
                    if attempt == 1 and self._is_connection_error(exc):
                        self._reset()
                        continue
                    raise

    # -- users
    def get_user(self, username):
        row = self._run(
            "SELECT password_hash, name, enrol, course, created FROM users WHERE username = %s",
            (username,), fetch="one")
        if not row:
            return None
        return {"password_hash": row[0], "name": row[1], "enrol": row[2],
                "course": row[3], "created": row[4]}

    def create_user(self, username, record):
        created = self._run(
            "INSERT INTO users (username, password_hash, name, enrol, course, created) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (username) DO NOTHING",
            (username, record["password_hash"], record.get("name", ""), record.get("enrol", ""),
             record.get("course", ""), int(record.get("created") or time.time())))
        if created:
            self._user_cache[username] = time.time() + USER_CACHE_SECONDS
        return bool(created)

    def user_exists(self, username):
        """Checked on every AI request, so remember a positive answer for a short while."""
        if self._user_cache.get(username, 0) > time.time():
            return True
        row = self._run("SELECT 1 FROM users WHERE username = %s", (username,), fetch="one")
        if row:
            self._user_cache[username] = time.time() + USER_CACHE_SECONDS
            return True
        self._user_cache.pop(username, None)
        return False

    # -- admins
    def get_admins(self):
        rows = self._run("SELECT username, password_hash, role FROM admins", fetch="all")
        return {u: {"password_hash": h, "role": r} for u, h, r in rows}

    def create_admin(self, username, record):
        return bool(self._run(
            "INSERT INTO admins (username, password_hash, role) VALUES (%s, %s, %s) "
            "ON CONFLICT (username) DO NOTHING",
            (username, record["password_hash"], record.get("role", "admin"))))

    # -- branding
    def get_branding(self):
        row = self._run("SELECT name, tagline FROM branding WHERE id = 1", fetch="one")
        if not row:
            return dict(DEFAULT_BRANDING)
        return {"name": row[0] or DEFAULT_BRANDING["name"], "tagline": row[1] or ""}

    def set_branding(self, name, tagline):
        self._run(
            "INSERT INTO branding (id, name, tagline) VALUES (1, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, tagline = EXCLUDED.tagline",
            (name, tagline))
        return {"name": name, "tagline": tagline}

    # -- settings
    def get_or_create_secret(self):
        row = self._run("SELECT value FROM settings WHERE key = 'secret_key'", fetch="one")
        if row:
            return row[0]
        self._run("INSERT INTO settings (key, value) VALUES ('secret_key', %s) "
                  "ON CONFLICT (key) DO NOTHING", (secrets.token_hex(32),))
        return self._run("SELECT value FROM settings WHERE key = 'secret_key'", fetch="one")[0]

    # -- documents
    def save_chunks(self, rows):
        """rows: dicts with id, source, page, page_end, course_code, text, embedding."""
        if not rows:
            return
        self._ensure_schema()
        values = [(r["id"], r["source"], int(r.get("page") or 0), int(r.get("page_end") or 0),
                   r.get("course_code"), r["text"],
                   json.dumps([round(float(x), 6) for x in r["embedding"]]))
                  for r in rows]
        for start in range(0, len(values), 100):
            part = values[start:start + 100]
            for attempt in (1, 2):
                with self._lock:
                    conn = None
                    try:
                        conn = self._get_conn()
                        cur = conn.cursor()
                        if self.is_sqlite:
                            cur.executemany(
                                UPSERT_CHUNK.format(values="(?, ?, ?, ?, ?, ?, ?)"), part)
                        else:
                            from psycopg2.extras import execute_values
                            execute_values(cur, UPSERT_CHUNK.format(values="%s"), part)
                        conn.commit()
                        cur.close()
                        break
                    except Exception as exc:
                        try:
                            if conn is not None:
                                conn.rollback()
                        except Exception:
                            pass
                        if attempt == 1 and self._is_connection_error(exc):
                            self._reset()
                            continue
                        raise

    def remove_chunks(self, source):
        self._run("DELETE FROM doc_chunks WHERE source = %s", (source,))

    def mark_deleted(self, source):
        self._run("INSERT INTO deleted_sources (source) VALUES (%s) ON CONFLICT (source) DO NOTHING",
                  (source,))

    def unmark_deleted(self, source):
        self._run("DELETE FROM deleted_sources WHERE source = %s", (source,))

    def list_deleted(self):
        return [r[0] for r in self._run("SELECT source FROM deleted_sources", fetch="all")]

    def iter_chunks(self, batch_size=200):
        """Yield lists of chunk dicts, in a stable order, without loading everything at once."""
        last = ""
        while True:
            rows = self._run(
                "SELECT chunk_id, source, page, page_end, course_code, body, embedding "
                "FROM doc_chunks WHERE chunk_id > %s ORDER BY chunk_id LIMIT %s",
                (last, batch_size), fetch="all")
            if not rows:
                return
            yield [{"id": r[0], "source": r[1], "page": r[2], "page_end": r[3],
                    "course_code": r[4], "text": r[5], "embedding": json.loads(r[6])}
                   for r in rows]
            last = rows[-1][0]


# ====================================================================== choose a backend
def _make_backend():
    url = (os.getenv("DATABASE_URL") or "").strip()
    return SqlBackend(url) if url else FileBackend()


storage = _make_backend()
