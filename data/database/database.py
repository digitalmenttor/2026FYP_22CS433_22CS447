"""
Smart Classroom — MySQL Database Layer
Saari tables aur CRUD operations yahan hain.

Tables:
  students          — registered students (name, roll_number, department, face embedding blob)
  sessions          — monitoring sessions (start/end time, stats)
  cheating_events   — har cheating incident ka record
  alert_history     — per-session yaw/peeking/mobile/paper alerts
  attendance        — session mein kaun present tha
  engagement_stats  — per-student engagement scores
"""

import mysql.connector
from mysql.connector import pooling
import json
import pickle
import numpy as np
from datetime import datetime
import os

# ──────────────────────────────────────────────
# CONFIG  (env vars se override hota hai)
# ──────────────────────────────────────────────
DB_CONFIG = {
    'host':     os.getenv('MYSQL_HOST',     'localhost'),
    'port':     int(os.getenv('MYSQL_PORT', 3306)),
    'user':     os.getenv('MYSQL_USER',     'root'),
    'password': os.getenv('MYSQL_PASSWORD', '1234'),
    'database': os.getenv('MYSQL_DATABASE', 'smart_classroom'),
    'autocommit': True,
    'charset':  'utf8mb4',
}

# Connection pool (5 connections)
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="sc_pool",
            pool_size=5,
            **DB_CONFIG
        )
    return _pool

def get_conn():
    return get_pool().get_connection()

# ──────────────────────────────────────────────
# INIT — saari tables banao agar exist nahi karte
# ──────────────────────────────────────────────
def init_db():
    """Database aur tables create karo."""
    # Pehle database without db name se connect karo
    cfg = {k: v for k, v in DB_CONFIG.items() if k != 'database'}
    cfg['autocommit'] = True
    conn = mysql.connector.connect(**cfg)
    cur  = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.close()

    conn = get_conn()
    cur  = conn.cursor()

    # ── students table (roll_number + department included) ────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id               INT            PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            roll_number      VARCHAR(100),
            department       VARCHAR(255),
            registered_at    DATETIME       DEFAULT CURRENT_TIMESTAMP,
            face_embeddings  LONGBLOB,
            total_sessions   INT            DEFAULT 0,
            total_alerts     INT            DEFAULT 0,
            notes            TEXT,
            is_active        TINYINT(1)     DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Migration: purani table mein nayi columns add karo ───────────────
    for col, definition in [
        ('roll_number', 'VARCHAR(100)'),
        ('department',  'VARCHAR(255)'),
    ]:
        try:
            cur.execute(f"ALTER TABLE students ADD COLUMN {col} {definition}")
            print(f"✅ Migration: column '{col}' added to students table")
        except Exception:
            pass  # Column pehle se exist karta hai — ignore

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            started_at       DATETIME       NOT NULL,
            ended_at         DATETIME,
            duration_sec     INT,
            total_frames     INT            DEFAULT 0,
            students_present INT            DEFAULT 0,
            total_yaw_alerts     INT        DEFAULT 0,
            total_peeking_alerts INT        DEFAULT 0,
            total_mobile_alerts  INT        DEFAULT 0,
            total_paper_exchanges INT       DEFAULT 0,
            exam_name        VARCHAR(200),
            exam_type        ENUM('quiz','midterm','final','practice') DEFAULT 'quiz',
            classroom_id     INT,
            notes            TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    # ── Migration: sessions table mein exam + classroom columns add karo ──
    for col, definition in [
        ('exam_name',    'VARCHAR(200)'),
        ('exam_type',    "ENUM('quiz','midterm','final','practice') DEFAULT 'quiz'"),
        ('classroom_id', 'INT'),
    ]:
        try:
            cur.execute(f"ALTER TABLE sessions ADD COLUMN {col} {definition}")
            print(f"✅ Migration: column '{col}' added to sessions table")
        except Exception:
            pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cheating_events (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            session_id       INT,
            student_id       INT,
            student_name     VARCHAR(255),
            event_type       ENUM('yaw','peeking','mobile','paper_exchange','zone_cross','hand_proximity','book') NOT NULL,
            severity         ENUM('LOW','MEDIUM','HIGH') DEFAULT 'MEDIUM',
            direction        VARCHAR(50),
            device_class     VARCHAR(100),
            paper_from       INT,
            paper_to         INT,
            confidence       FLOAT,
            frame_number     INT,
            timestamp_sec    FLOAT,
            occurred_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
            INDEX idx_student (student_id),
            INDEX idx_session (session_id),
            INDEX idx_type    (event_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── Migration: cheating_events event_type mein 'book' add karo ───────
    try:
        cur.execute("""
            ALTER TABLE cheating_events
            MODIFY COLUMN event_type
            ENUM('yaw','peeking','mobile','paper_exchange','zone_cross','hand_proximity','book') NOT NULL
        """)
        print("✅ Migration: 'book' added to cheating_events event_type enum")
    except Exception:
        pass

    cur.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            session_id       INT,
            student_id       INT,
            student_name     VARCHAR(255),
            alert_type       VARCHAR(100)   NOT NULL,
            detail           TEXT,
            occurred_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
            INDEX idx_session  (session_id),
            INDEX idx_student  (student_id),
            INDEX idx_type     (alert_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            session_id       INT            NOT NULL,
            student_id       INT            NOT NULL,
            student_name     VARCHAR(255),
            first_seen_at    DATETIME       DEFAULT CURRENT_TIMESTAMP,
            last_seen_at     DATETIME       DEFAULT CURRENT_TIMESTAMP,
            frames_present   INT            DEFAULT 0,
            UNIQUE KEY uq_session_student (session_id, student_id),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            INDEX idx_student (student_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS engagement_stats (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            session_id       INT,
            student_id       INT            NOT NULL,
            student_name     VARCHAR(255),
            engagement_score FLOAT          DEFAULT 100.0,
            looking_away_frames INT         DEFAULT 0,
            total_frames     INT            DEFAULT 0,
            recorded_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_session_student (session_id, student_id),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
            INDEX idx_student (student_id),
            INDEX idx_session (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.close()
    print("✅ MySQL tables initialized successfully!")
    create_classrooms_tables()  # ✅ YAHAN — sab tables ban jaane ke baad


# ──────────────────────────────────────────────
# STUDENTS
# ──────────────────────────────────────────────

def save_student(student_id: int, name: str, embeddings: list,
                 roll_number: str = None, department: str = None):
    """
    Student register/update karo.
    embeddings: list of numpy arrays → pickle → BLOB
    roll_number: student ka roll number (optional)
    department:  student ka department (optional)
    """
    emb_blob = pickle.dumps(embeddings) if embeddings else None
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO students
            (id, name, roll_number, department, face_embeddings, registered_at, is_active)
        VALUES (%s, %s, %s, %s, %s, %s, 1)
        ON DUPLICATE KEY UPDATE
            name            = VALUES(name),
            roll_number     = VALUES(roll_number),
            department      = VALUES(department),
            face_embeddings = VALUES(face_embeddings),
            is_active       = 1
    """, (student_id, name, roll_number, department, emb_blob, datetime.now()))
    conn.close()


def load_all_students() -> dict:
    """
    Saare students load karo.
    Returns: {id: {'name': str, 'roll_number': str, 'department': str, 'embeddings': [np.array, ...]}}
    """
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, name, roll_number, department, face_embeddings
        FROM students WHERE is_active=1
    """)
    rows = cur.fetchall()
    conn.close()

    result = {}
    for row in rows:
        embs = pickle.loads(row['face_embeddings']) if row['face_embeddings'] else []
        result[row['id']] = {
            'name':        row['name'],
            'roll_number': row['roll_number'] or '',
            'department':  row['department']  or '',
            'embeddings':  embs,
        }
    return result


def add_embedding_to_student(student_id: int, new_embedding: np.ndarray):
    """Existing student mein ek nayi embedding add karo."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT face_embeddings FROM students WHERE id=%s", (student_id,))
    row = cur.fetchone()
    if row:
        embs = pickle.loads(row['face_embeddings']) if row['face_embeddings'] else []
        embs.append(new_embedding)
        cur.execute("UPDATE students SET face_embeddings=%s WHERE id=%s",
                    (pickle.dumps(embs), student_id))
    conn.close()


def delete_student(student_id: int):
    """Student ko soft-delete karo (is_active=0)."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE students SET is_active=0 WHERE id=%s", (student_id,))
    conn.close()


def update_student_alert_count(student_id: int, increment: int = 1):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE students SET total_alerts = total_alerts + %s WHERE id=%s",
                (increment, student_id))
    conn.close()


def get_student_full_detail(student_id: int) -> dict:
    """Ek student ki poori detail — cheating history, sessions, engagement."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)

    # Basic info (roll_number + department bhi)
    cur.execute("""
        SELECT id, name, roll_number, department,
               registered_at, total_sessions, total_alerts, notes, is_active
        FROM students WHERE id=%s
    """, (student_id,))
    student = cur.fetchone()
    if not student:
        conn.close()
        return {}

    # Cheating events
    cur.execute("""
        SELECT event_type, severity, direction, device_class,
               paper_from, paper_to, occurred_at
        FROM cheating_events WHERE student_id=%s
        ORDER BY occurred_at DESC LIMIT 50
    """, (student_id,))
    events = cur.fetchall()

    # Engagement history
    cur.execute("""
        SELECT session_id, engagement_score, looking_away_frames,
               total_frames, recorded_at
        FROM engagement_stats WHERE student_id=%s
        ORDER BY recorded_at DESC LIMIT 20
    """, (student_id,))
    engagement = cur.fetchall()

    # Alert summary
    cur.execute("""
        SELECT alert_type, COUNT(*) as cnt
        FROM alert_history WHERE student_id=%s
        GROUP BY alert_type
    """, (student_id,))
    alert_summary = {r['alert_type']: r['cnt'] for r in cur.fetchall()}

    # Attendance count
    cur.execute("SELECT COUNT(*) as cnt FROM attendance WHERE student_id=%s", (student_id,))
    attendance_count = cur.fetchone()['cnt']

    conn.close()

    # Datetime → string
    def dt_str(v):
        return v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, datetime) else str(v) if v else None

    student['registered_at'] = dt_str(student['registered_at'])
    for e in events:
        e['occurred_at'] = dt_str(e['occurred_at'])
    for eg in engagement:
        eg['recorded_at'] = dt_str(eg['recorded_at'])

    return {
        'student':           student,
        'cheating_events':   events,
        'engagement':        engagement,
        'alert_summary':     alert_summary,
        'sessions_attended': attendance_count,
    }


def get_all_students_summary() -> list:
    """Dashboard ke liye saare students ka summary (roll_number + department bhi)."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.roll_number, s.department,
               s.registered_at, s.total_sessions,
               s.total_alerts, s.is_active,
               COUNT(DISTINCT a.session_id)      AS sessions_attended,
               ROUND(AVG(e.engagement_score), 1) AS avg_engagement
        FROM students s
        LEFT JOIN attendance      a ON a.student_id = s.id
        LEFT JOIN engagement_stats e ON e.student_id = s.id
        WHERE s.is_active = 1
        GROUP BY s.id
        ORDER BY s.registered_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('registered_at'), datetime):
            r['registered_at'] = r['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# SESSIONS
# ──────────────────────────────────────────────

def start_session(exam_name: str = None, exam_type: str = 'quiz',
                  classroom_id: int = None) -> int:
    """Nayi session shuru karo, session_id return karo."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO sessions (started_at, exam_name, exam_type, classroom_id)
        VALUES (%s, %s, %s, %s)
    """, (datetime.now(), exam_name or None, exam_type, classroom_id or None))
    sid = cur.lastrowid
    conn.close()
    return sid


def end_session(session_id: int, stats: dict):
    """Session band karo aur final stats save karo."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE sessions SET
            ended_at              = %s,
            duration_sec          = %s,
            total_frames          = %s,
            students_present      = %s,
            total_yaw_alerts      = %s,
            total_peeking_alerts  = %s,
            total_mobile_alerts   = %s,
            total_paper_exchanges = %s
        WHERE id=%s
    """, (
        datetime.now(),
        stats.get('session_time', 0),
        stats.get('total_frames', 0),
        stats.get('person_count', 0),
        stats.get('total_yaw_alerts', 0),
        stats.get('total_peeking_alerts', 0),
        stats.get('total_mobile_detections', 0),
        stats.get('total_paper_exchanges', 0),
        session_id,
    ))
    conn.close()


def get_sessions_list(limit: int = 20) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, started_at, ended_at, duration_sec,
               students_present, total_yaw_alerts, total_peeking_alerts,
               total_mobile_alerts, total_paper_exchanges,
               exam_name, exam_type, classroom_id
        FROM sessions ORDER BY started_at DESC LIMIT %s
    """, (limit,))  
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        for k in ('started_at', 'ended_at'):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# CHEATING EVENTS
# ──────────────────────────────────────────────

def log_cheating_event(session_id, student_id, student_name,
                        event_type, severity='MEDIUM', **kwargs):
    """
    Ek cheating event log karo.
    kwargs: direction, device_class, paper_from, paper_to, confidence,
            frame_number, timestamp_sec
    """
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO cheating_events
            (session_id, student_id, student_name, event_type, severity,
             direction, device_class, paper_from, paper_to,
             confidence, frame_number, timestamp_sec, occurred_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        session_id, student_id, student_name, event_type, severity,
        kwargs.get('direction'),
        kwargs.get('device_class'),
        kwargs.get('paper_from'),
        kwargs.get('paper_to'),
        kwargs.get('confidence'),
        kwargs.get('frame_number'),
        kwargs.get('timestamp_sec'),
        datetime.now(),
    ))
    conn.close()
    if student_id:
        update_student_alert_count(student_id)


def get_cheating_events(session_id=None, student_id=None,
                         event_type=None, limit=100) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    where, params = [], []
    if session_id:  where.append("session_id=%s");  params.append(session_id)
    if student_id:  where.append("student_id=%s");  params.append(student_id)
    if event_type:  where.append("event_type=%s");  params.append(event_type)
    sql = "SELECT * FROM cheating_events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY occurred_at DESC LIMIT %s"
    params.append(limit)
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('occurred_at'), datetime):
            r['occurred_at'] = r['occurred_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# ALERT HISTORY
# ──────────────────────────────────────────────

def log_alert(session_id, student_id, student_name, alert_type, detail=''):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO alert_history
            (session_id, student_id, student_name, alert_type, detail, occurred_at)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, (session_id, student_id, student_name, alert_type, detail, datetime.now()))
    conn.close()


# ──────────────────────────────────────────────
# ATTENDANCE
# ──────────────────────────────────────────────

def mark_attendance(session_id: int, student_id, student_name: str):
    """Student ko is session mein present mark karo (upsert)."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO attendance (session_id, student_id, student_name, frames_present)
        VALUES (%s,%s,%s,1)
        ON DUPLICATE KEY UPDATE
            last_seen_at   = CURRENT_TIMESTAMP,
            frames_present = frames_present + 1
    """, (session_id, student_id, student_name))
    conn.close()


def get_session_attendance(session_id: int) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT student_id, student_name, first_seen_at, last_seen_at, frames_present
        FROM attendance WHERE session_id=%s ORDER BY first_seen_at
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        for k in ('first_seen_at', 'last_seen_at'):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# ENGAGEMENT STATS
# ──────────────────────────────────────────────

def save_engagement(session_id, student_id, student_name,
                    engagement_score, looking_away_frames, total_frames):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO engagement_stats
            (session_id, student_id, student_name,
             engagement_score, looking_away_frames, total_frames, recorded_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            engagement_score    = VALUES(engagement_score),
            looking_away_frames = VALUES(looking_away_frames),
            total_frames        = VALUES(total_frames),
            recorded_at         = VALUES(recorded_at)
    """, (session_id, student_id, student_name,
          engagement_score, looking_away_frames, total_frames, datetime.now()))
    conn.close()


def get_student_engagement_trend(student_id, limit=10) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT session_id, engagement_score, recorded_at
        FROM engagement_stats WHERE student_id=%s
        ORDER BY recorded_at DESC LIMIT %s
    """, (student_id, limit))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('recorded_at'), datetime):
            r['recorded_at'] = r['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# REPORTS
# ──────────────────────────────────────────────

def export_session_report_csv(session_id: int, path: str):
    """Ek session ki poori report CSV mein export karo."""
    import csv
    events = get_cheating_events(session_id=session_id, limit=10000)
    attend = get_session_attendance(session_id)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)

        w.writerow(['=== ATTENDANCE ==='])
        w.writerow(['Student ID', 'Name', 'First Seen', 'Last Seen', 'Frames Present'])
        for a in attend:
            w.writerow([a['student_id'], a['student_name'],
                        a['first_seen_at'], a['last_seen_at'], a['frames_present']])

        w.writerow([])
        w.writerow(['=== CHEATING EVENTS ==='])
        w.writerow(['Time', 'Type', 'Student ID', 'Name', 'Severity', 'Direction',
                    'Device', 'Paper From', 'Paper To', 'Confidence'])
        for e in events:
            w.writerow([e['occurred_at'], e['event_type'], e['student_id'],
                        e['student_name'], e['severity'], e['direction'],
                        e['device_class'], e['paper_from'], e['paper_to'],
                        e['confidence']])
    return path


def get_dashboard_summary() -> dict:
    """Home dashboard ke liye aggregate stats."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)

    cur.execute("SELECT COUNT(*) as cnt FROM students WHERE is_active=1")
    total_students = cur.fetchone()['cnt']

    cur.execute("SELECT COUNT(*) as cnt FROM sessions")
    total_sessions = cur.fetchone()['cnt']

    cur.execute("SELECT COUNT(*) as cnt FROM cheating_events")
    total_events = cur.fetchone()['cnt']

    cur.execute("""
        SELECT event_type, COUNT(*) as cnt
        FROM cheating_events GROUP BY event_type
    """)
    event_breakdown = {r['event_type']: r['cnt'] for r in cur.fetchall()}

    cur.execute("""
        SELECT s.name, s.roll_number, s.department, COUNT(ce.id) as alert_count
        FROM students s
        LEFT JOIN cheating_events ce ON ce.student_id = s.id
        WHERE s.is_active=1
        GROUP BY s.id ORDER BY alert_count DESC LIMIT 5
    """)
    top_offenders = cur.fetchall()

    cur.execute("""
        SELECT id, started_at, ended_at, students_present,
               total_yaw_alerts + total_peeking_alerts +
               total_mobile_alerts + total_paper_exchanges AS total_alerts
        FROM sessions ORDER BY started_at DESC LIMIT 5
    """)
    recent_sessions = cur.fetchall()
    for r in recent_sessions:
        for k in ('started_at', 'ended_at'):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')

    conn.close()
    return {
        'total_students':  total_students,
        'total_sessions':  total_sessions,
        'total_events':    total_events,
        'event_breakdown': event_breakdown,
        'top_offenders':   top_offenders,
        'recent_sessions': recent_sessions,
    }
# database.py ke end mein add karo

# ──────────────────────────────────────────────
# CLASSROOMS
# ──────────────────────────────────────────────

def create_classrooms_tables():
    """Classroom tables banao — init_db() ke andar call karo."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classrooms (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            name       VARCHAR(100) NOT NULL,
            exam_date  DATE,
            notes      TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active  TINYINT(1) DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_classroom (
            student_id   INT NOT NULL,
            classroom_id INT NOT NULL,
            assigned_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (student_id, classroom_id),
            FOREIGN KEY (student_id)   REFERENCES students(id)    ON DELETE CASCADE,
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id)  ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)
    conn.close()
    print("✅ Classroom tables ready")


def get_all_classrooms() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT c.id, c.name, c.exam_date, c.notes,
               COUNT(sc.student_id) AS student_count
        FROM classrooms c
        LEFT JOIN student_classroom sc ON sc.classroom_id = c.id
        WHERE c.is_active = 1
        GROUP BY c.id
        ORDER BY c.name
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if r.get('exam_date'):
            r['exam_date'] = str(r['exam_date'])
    return rows


def create_classroom(name: str, exam_date=None, notes='') -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO classrooms (name, exam_date, notes) VALUES (%s, %s, %s)",
        (name, exam_date or None, notes)
    )
    cid = cur.lastrowid
    conn.close()
    return cid


def delete_classroom(classroom_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE classrooms SET is_active=0 WHERE id=%s", (classroom_id,))
    conn.close()


def assign_student_to_classroom(student_id: int, classroom_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT IGNORE INTO student_classroom (student_id, classroom_id)
        VALUES (%s, %s)
    """, (student_id, classroom_id))
    conn.close()


def remove_student_from_classroom(student_id: int, classroom_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        DELETE FROM student_classroom
        WHERE student_id=%s AND classroom_id=%s
    """, (student_id, classroom_id))
    conn.close()


def get_classroom_students(classroom_id: int) -> list:
    """Is classroom ke assigned students."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.roll_number, s.department
        FROM students s
        JOIN student_classroom sc ON sc.student_id = s.id
        WHERE sc.classroom_id = %s AND s.is_active = 1
        ORDER BY s.name
    """, (classroom_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def check_student_in_classroom(student_id, classroom_id) -> bool:
    """True agar student is classroom mein assigned hai."""
    if not classroom_id:
        return True  # classroom set nahi — strict mode off
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT 1 FROM student_classroom
        WHERE student_id=%s AND classroom_id=%s LIMIT 1
    """, (student_id, classroom_id))
    found = cur.fetchone() is not None
    conn.close()
    return found


def get_student_classroom(student_id) -> dict:
    """Student kis classroom mein hai."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT c.id, c.name FROM classrooms c
        JOIN student_classroom sc ON sc.classroom_id = c.id
        WHERE sc.student_id = %s AND c.is_active = 1
        LIMIT 1
    """, (student_id,))
    row = cur.fetchone()
    conn.close()
    return row or {}