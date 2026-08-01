"""
Smart Classroom — MySQL Database Layer
MySQL + Risk Scoring Engine ke liye saari tables aur CRUD operations.

Tables (EXISTING - same rahenge):
  students          — registered students (name, reg date, face embedding blob)
  sessions          — monitoring sessions (start/end time, stats)
  cheating_events   — har cheating incident ka record
  alert_history     — per-session alerts
  attendance        — session mein kaun present tha
  engagement_stats  — per-student engagement scores

Tables (NAYE - add ho rahe hain):
  departments       — department ka naam aur code
  classes           — class ka naam, department, camera URL
  sections          — section ka naam, class ke saath linked
  subjects          — subject ka naam, class ke saath linked
  teachers          — naam, email, password, assigned class aur subjects
  student_subjects  — student aur subject ka many-to-many mapping
  admin_users       — admin accounts
  notifications     — admin ke liye pending student notifications
"""

import mysql.connector
from mysql.connector import pooling
import json
import pickle
import numpy as np
from datetime import datetime
import os
import hashlib

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

# Connection pool — size 20 kar diya (requests zyada hain)
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="sc_pool",
            pool_size=20,          # 5 se 20 — pool exhausted fix
            pool_reset_session=True,
            **DB_CONFIG
        )
    return _pool

def get_conn():
    return get_pool().get_connection()

class _ConnCtx:
    """
    Context manager taake 'with get_conn() as conn:' kaam kare
    aur connection hamesha close ho — chahe error aaye ya na aaye.
    """
    def __init__(self):
        self._conn = get_pool().get_connection()
    def __enter__(self):
        return self._conn
    def __exit__(self, *_):
        try:
            self._conn.close()
        except Exception:
            pass

def conn_ctx():
    """Use: with conn_ctx() as conn: — guaranteed close."""
    return _ConnCtx()

def hash_password(password: str) -> str:
    """Simple SHA-256 password hash."""
    return hashlib.sha256(password.encode()).hexdigest()


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

    # ── EXISTING TABLES (same rahenge) ────────────────────────────────────────

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id               INT            PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            registered_at    DATETIME       DEFAULT CURRENT_TIMESTAMP,
            face_embeddings  LONGBLOB,
            total_sessions   INT            DEFAULT 0,
            total_alerts     INT            DEFAULT 0,
            notes            TEXT,
            is_active        TINYINT(1)     DEFAULT 1,
            -- NAYE COLUMNS (existing table mein add)
            email            VARCHAR(255)   UNIQUE,
            password_hash    VARCHAR(255),
            roll_no          VARCHAR(50),
            date_of_birth    DATE,
            class_id         INT,
            section_id       INT,
            status           ENUM('pending','active','rejected') DEFAULT 'pending'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # Agar table pehle se exist karti hai toh naye columns add karo (safe ALTER)
    _safe_add_column(cur, 'students', 'email',         "VARCHAR(255)")
    _safe_add_column(cur, 'students', 'password_hash', "VARCHAR(255)")
    _safe_add_column(cur, 'students', 'roll_no',       "VARCHAR(50)")
    _safe_add_column(cur, 'students', 'date_of_birth', "DATE")
    _safe_add_column(cur, 'students', 'class_id',      "INT")
    _safe_add_column(cur, 'students', 'section_id',    "INT")
    _safe_add_column(cur, 'students', 'status',        "ENUM('pending','active','rejected') DEFAULT 'pending'")

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
            notes            TEXT,
            class_id         INT,
            teacher_id       INT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

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
            occurred_at      DATETIME           DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
            INDEX idx_student (student_id),
            INDEX idx_session (session_id),
            INDEX idx_type    (event_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

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
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL,
            INDEX idx_student (student_id),
            INDEX idx_session (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    # ── NAYE TABLES ────────────────────────────────────────────────────────────

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            email            VARCHAR(255)   UNIQUE NOT NULL,
            password_hash    VARCHAR(255)   NOT NULL,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            code             VARCHAR(50)    UNIQUE NOT NULL,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            department_id    INT,
            camera_url       VARCHAR(500),
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1,
            FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE SET NULL,
            INDEX idx_department (department_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(50)    NOT NULL,
            class_id         INT            NOT NULL,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1,
            FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE,
            INDEX idx_class (class_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            class_id         INT            NOT NULL,
            teacher_id       INT,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1,
            FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE,
            INDEX idx_class   (class_id),
            INDEX idx_teacher (teacher_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            name             VARCHAR(255)   NOT NULL,
            email            VARCHAR(255)   UNIQUE NOT NULL,
            password_hash    VARCHAR(255)   NOT NULL,
            assigned_class_id INT,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            is_active        TINYINT(1)     DEFAULT 1,
            FOREIGN KEY (assigned_class_id) REFERENCES classes(id) ON DELETE SET NULL,
            INDEX idx_class (assigned_class_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            teacher_id       INT            NOT NULL,
            subject_id       INT            NOT NULL,
            UNIQUE KEY uq_teacher_subject (teacher_id, subject_id),
            FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_subjects (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            student_id       INT            NOT NULL,
            subject_id       INT            NOT NULL,
            UNIQUE KEY uq_student_subject (student_id, subject_id),
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            type             VARCHAR(100)   NOT NULL,
            message          TEXT           NOT NULL,
            student_id       INT,
            is_read          TINYINT(1)     DEFAULT 0,
            created_at       DATETIME       DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_read   (is_read)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.close()

    # Default admin account banana (pehli baar)
    _create_default_admin()

    print("✅ MySQL tables initialized successfully!")


def _safe_add_column(cur, table, column, col_def):
    """Column add karo agar exist nahi karta."""
    try:
        cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {col_def}")
    except Exception:
        pass  # Column pehle se exist karta hai


def _create_default_admin():
    """Agar koi admin nahi hai toh default admin banao."""
    try:
        conn = get_conn()
        cur  = conn.cursor(dictionary=True)
        cur.execute("SELECT id FROM admin_users LIMIT 1")
        row = cur.fetchone()
        if not row:
            cur.execute("""
                INSERT INTO admin_users (name, email, password_hash)
                VALUES (%s, %s, %s)
            """, ('Admin', 'admin@smartclass.com', hash_password('admin123')))
            print("✅ Default admin created: admin@smartclass.com / admin123")
        conn.close()
    except Exception as e:
        print(f"⚠️  Could not create default admin: {e}")


# ──────────────────────────────────────────────
# ADMIN AUTH
# ──────────────────────────────────────────────

def admin_login(email: str, password: str) -> dict:
    """Admin login verify karo. Returns admin dict ya None."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, name, email FROM admin_users
        WHERE email=%s AND password_hash=%s AND is_active=1
    """, (email, hash_password(password)))
    row = cur.fetchone()
    conn.close()
    return row


# ──────────────────────────────────────────────
# DEPARTMENTS
# ──────────────────────────────────────────────

def create_department(name: str, code: str) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("INSERT INTO departments (name, code) VALUES (%s, %s)", (name, code))
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_all_departments() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM departments WHERE is_active=1 ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_department(dept_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE departments SET is_active=0 WHERE id=%s", (dept_id,))
    conn.close()


# ──────────────────────────────────────────────
# CLASSES
# ──────────────────────────────────────────────

def create_class(name: str, department_id: int, camera_url: str = None) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO classes (name, department_id, camera_url)
        VALUES (%s, %s, %s)
    """, (name, department_id, camera_url))
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_all_classes() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT c.*, d.name as department_name
        FROM classes c
        LEFT JOIN departments d ON d.id = c.department_id
        WHERE c.is_active=1
        ORDER BY c.name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def update_class_camera(class_id: int, camera_url: str):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE classes SET camera_url=%s WHERE id=%s", (camera_url, class_id))
    conn.close()


def delete_class(class_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE classes SET is_active=0 WHERE id=%s", (class_id,))
    conn.close()


def get_class_by_id(class_id: int) -> dict:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT c.*, d.name as department_name
        FROM classes c
        LEFT JOIN departments d ON d.id = c.department_id
        WHERE c.id=%s AND c.is_active=1
    """, (class_id,))
    row = cur.fetchone()
    conn.close()
    return row


# ──────────────────────────────────────────────
# SECTIONS
# ──────────────────────────────────────────────

def create_section(name: str, class_id: int) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("INSERT INTO sections (name, class_id) VALUES (%s, %s)", (name, class_id))
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_sections_by_class(class_id: int) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.*, c.name as class_name
        FROM sections s
        JOIN classes c ON c.id = s.class_id
        WHERE s.class_id=%s AND s.is_active=1
        ORDER BY s.name
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_sections() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.*, c.name as class_name
        FROM sections s
        JOIN classes c ON c.id = s.class_id
        WHERE s.is_active=1
        ORDER BY c.name, s.name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_section(section_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE sections SET is_active=0 WHERE id=%s", (section_id,))
    conn.close()


# ──────────────────────────────────────────────
# SUBJECTS
# ──────────────────────────────────────────────

def create_subject(name: str, class_id: int) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("INSERT INTO subjects (name, class_id) VALUES (%s, %s)", (name, class_id))
    new_id = cur.lastrowid
    conn.close()
    return new_id


def get_subjects_by_class(class_id: int) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT sub.*, t.name as teacher_name
        FROM subjects sub
        LEFT JOIN teachers t ON t.id = sub.teacher_id
        WHERE sub.class_id=%s AND sub.is_active=1
        ORDER BY sub.name
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_all_subjects() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT sub.*, c.name as class_name, t.name as teacher_name
        FROM subjects sub
        JOIN classes c ON c.id = sub.class_id
        LEFT JOIN teachers t ON t.id = sub.teacher_id
        WHERE sub.is_active=1
        ORDER BY c.name, sub.name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def assign_teacher_to_subject(subject_id: int, teacher_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE subjects SET teacher_id=%s WHERE id=%s", (teacher_id, subject_id))
    conn.close()


def delete_subject(subject_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE subjects SET is_active=0 WHERE id=%s", (subject_id,))
    conn.close()


# ──────────────────────────────────────────────
# TEACHERS
# ──────────────────────────────────────────────

def create_teacher(name: str, email: str, password: str, class_id: int = None) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO teachers (name, email, password_hash, assigned_class_id)
        VALUES (%s, %s, %s, %s)
    """, (name, email, hash_password(password), class_id))
    new_id = cur.lastrowid
    conn.close()
    return new_id


def teacher_login(email: str, password: str) -> dict:
    """Teacher login verify karo."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT t.id, t.name, t.email, t.assigned_class_id,
               c.name as class_name, c.camera_url
        FROM teachers t
        LEFT JOIN classes c ON c.id = t.assigned_class_id
        WHERE t.email=%s AND t.password_hash=%s AND t.is_active=1
    """, (email, hash_password(password)))
    row = cur.fetchone()
    conn.close()
    return row


def get_all_teachers() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT t.id, t.name, t.email, t.assigned_class_id,
               t.created_at, t.is_active,
               c.name as class_name
        FROM teachers t
        LEFT JOIN classes c ON c.id = t.assigned_class_id
        WHERE t.is_active=1
        ORDER BY t.name
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


def assign_teacher_to_class(teacher_id: int, class_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE teachers SET assigned_class_id=%s WHERE id=%s", (class_id, teacher_id))
    conn.close()


def assign_teacher_subjects(teacher_id: int, subject_ids: list):
    """Teacher ko multiple subjects assign karo (purane replace ho jayenge)."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM teacher_subjects WHERE teacher_id=%s", (teacher_id,))
    for sid in subject_ids:
        cur.execute("""
            INSERT IGNORE INTO teacher_subjects (teacher_id, subject_id)
            VALUES (%s, %s)
        """, (teacher_id, sid))
    conn.close()


def get_teacher_subjects(teacher_id: int) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT sub.id, sub.name, sub.class_id, c.name as class_name
        FROM teacher_subjects ts
        JOIN subjects sub ON sub.id = ts.subject_id
        JOIN classes c ON c.id = sub.class_id
        WHERE ts.teacher_id=%s
    """, (teacher_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_teacher(teacher_id: int):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE teachers SET is_active=0 WHERE id=%s", (teacher_id,))
    conn.close()


def get_teacher_by_id(teacher_id: int) -> dict:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT t.*, c.name as class_name, c.camera_url
        FROM teachers t
        LEFT JOIN classes c ON c.id = t.assigned_class_id
        WHERE t.id=%s AND t.is_active=1
    """, (teacher_id,))
    row = cur.fetchone()
    conn.close()
    return row


# ──────────────────────────────────────────────
# STUDENT REGISTRATION (NAYA FLOW)
# ──────────────────────────────────────────────

def register_student_pending(
    name: str, email: str, password: str,
    roll_no: str, date_of_birth: str,
    class_id: int, section_id: int,
    subject_ids: list,
    face_embeddings: list = None
) -> int:
    """
    Student register karo with 'pending' status.
    Returns new student_id.
    """
    # Next available ID find karo
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM students")
    row     = cur.fetchone()
    next_id = row[0] if row else 1

    emb_blob = pickle.dumps(face_embeddings) if face_embeddings else None

    cur.execute("""
        INSERT INTO students
            (id, name, email, password_hash, roll_no, date_of_birth,
             class_id, section_id, status, face_embeddings, registered_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)
    """, (next_id, name, email, hash_password(password),
          roll_no, date_of_birth, class_id, section_id,
          emb_blob, datetime.now()))

    # Subjects assign karo
    for sid in subject_ids:
        try:
            cur.execute("""
                INSERT IGNORE INTO student_subjects (student_id, subject_id)
                VALUES (%s, %s)
            """, (next_id, sid))
        except Exception:
            pass

    conn.close()

    # Admin ko notification bhejo
    _add_notification(
        'new_student',
        f"Naya student register hua: {name} (Roll: {roll_no}) — approval pending",
        next_id
    )

    return next_id


def approve_student(
    student_id: int,
    class_id: int,
    section_id: int,
    subject_ids: list
) -> bool:
    """Admin student ko approve kare aur class/section/subjects confirm kare."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE students
        SET status='active', class_id=%s, section_id=%s, is_active=1
        WHERE id=%s
    """, (class_id, section_id, student_id))

    # Purane subjects replace karo
    cur.execute("DELETE FROM student_subjects WHERE student_id=%s", (student_id,))
    for sid in subject_ids:
        try:
            cur.execute("""
                INSERT IGNORE INTO student_subjects (student_id, subject_id)
                VALUES (%s, %s)
            """, (student_id, sid))
        except Exception:
            pass

    conn.close()
    return True


def reject_student(student_id: int) -> bool:
    """Admin student ko reject kare."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE students SET status='rejected' WHERE id=%s", (student_id,))
    conn.close()
    return True


def get_pending_students() -> list:
    """Admin panel ke liye pending approval wale students."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.email, s.roll_no, s.date_of_birth,
               s.class_id, s.section_id, s.registered_at,
               c.name as class_name,
               sec.name as section_name
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        LEFT JOIN sections sec ON sec.id = s.section_id
        WHERE s.status='pending'
        ORDER BY s.registered_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('registered_at'), datetime):
            r['registered_at'] = r['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
        # Us student ke subjects bhi fetch karo
        r['subjects'] = get_student_subjects(r['id'])
    return rows


def get_student_subjects(student_id: int) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT sub.id, sub.name
        FROM student_subjects ss
        JOIN subjects sub ON sub.id = ss.subject_id
        WHERE ss.student_id=%s
    """, (student_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


def student_login(email: str, password: str) -> dict:
    """Student login verify karo."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.email, s.roll_no, s.status,
               s.class_id, s.section_id,
               c.name as class_name,
               sec.name as section_name
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        LEFT JOIN sections sec ON sec.id = s.section_id
        WHERE s.email=%s AND s.password_hash=%s
    """, (email, hash_password(password)))
    row = cur.fetchone()
    conn.close()
    return row


# ──────────────────────────────────────────────
# STUDENT SUBJECTS
# ──────────────────────────────────────────────

def assign_student_subjects(student_id: int, subject_ids: list):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("DELETE FROM student_subjects WHERE student_id=%s", (student_id,))
    for sid in subject_ids:
        cur.execute("""
            INSERT IGNORE INTO student_subjects (student_id, subject_id)
            VALUES (%s, %s)
        """, (student_id, sid))
    conn.close()


# ──────────────────────────────────────────────
# NOTIFICATIONS
# ──────────────────────────────────────────────

def _add_notification(ntype: str, message: str, student_id: int = None):
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO notifications (type, message, student_id)
            VALUES (%s, %s, %s)
        """, (ntype, message, student_id))
        conn.close()
    except Exception:
        pass


def get_unread_notifications() -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT * FROM notifications
        WHERE is_read=0
        ORDER BY created_at DESC
        LIMIT 50
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('created_at'), datetime):
            r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


def mark_notifications_read():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("UPDATE notifications SET is_read=1 WHERE is_read=0")
    conn.close()


def get_notification_count() -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM notifications WHERE is_read=0")
    count = cur.fetchone()[0]
    conn.close()
    return count


# ──────────────────────────────────────────────
# ADMIN — ALL STUDENTS VIEW (active + pending)
# ──────────────────────────────────────────────

def get_all_students_admin() -> list:
    """Admin ke liye saare students (active + pending + rejected)."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.email, s.roll_no, s.status,
               s.registered_at, s.total_alerts,
               c.name as class_name, sec.name as section_name
        FROM students s
        LEFT JOIN classes c ON c.id = s.class_id
        LEFT JOIN sections sec ON sec.id = s.section_id
        ORDER BY s.registered_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('registered_at'), datetime):
            r['registered_at'] = r['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# TEACHER PORTAL — apni class ki info
# ──────────────────────────────────────────────

def get_class_students(class_id: int) -> list:
    """Teacher apni class ke active students dekhe."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.roll_no, s.email,
               sec.name as section_name,
               s.total_alerts, s.registered_at
        FROM students s
        LEFT JOIN sections sec ON sec.id = s.section_id
        WHERE s.class_id=%s AND s.status='active'
        ORDER BY s.name
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        if isinstance(r.get('registered_at'), datetime):
            r['registered_at'] = r['registered_at'].strftime('%Y-%m-%d %H:%M:%S')
    return rows


def get_class_sessions(class_id: int, limit: int = 10) -> list:
    """Teacher apni class ki past sessions dekhe."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, started_at, ended_at, duration_sec,
               students_present, total_yaw_alerts,
               total_peeking_alerts, total_mobile_alerts, total_paper_exchanges
        FROM sessions
        WHERE class_id=%s
        ORDER BY started_at DESC
        LIMIT %s
    """, (class_id, limit))
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        for k in ('started_at', 'ended_at'):
            if isinstance(r.get(k), datetime):
                r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')
    return rows


# ──────────────────────────────────────────────
# EXISTING FUNCTIONS — SAME RAHENGE
# ──────────────────────────────────────────────

def save_student(student_id: int, name: str, embeddings: list):
    """
    Student register/update karo.
    embeddings: list of numpy arrays → pickle → BLOB
    """
    emb_blob = pickle.dumps(embeddings) if embeddings else None
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO students (id, name, face_embeddings, registered_at, status)
        VALUES (%s, %s, %s, %s, 'active')
        ON DUPLICATE KEY UPDATE
            name             = VALUES(name),
            face_embeddings  = VALUES(face_embeddings)
    """, (student_id, name, emb_blob, datetime.now()))
    conn.close()


def load_all_students() -> dict:
    """
    Saare active students load karo.
    Returns: {id: {'name': str, 'embeddings': [np.array, ...]}}
    """
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, face_embeddings FROM students WHERE is_active=1 AND status='active'")
    rows = cur.fetchall()
    conn.close()

    result = {}
    for row in rows:
        embs = pickle.loads(row['face_embeddings']) if row['face_embeddings'] else []
        result[row['id']] = {'name': row['name'], 'embeddings': embs}
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

    cur.execute("""
        SELECT id, name, email, roll_no, registered_at,
               total_sessions, total_alerts, notes, is_active, status
        FROM students WHERE id=%s
    """, (student_id,))
    student = cur.fetchone()
    if not student:
        conn.close()
        return {}

    cur.execute("""
        SELECT event_type, severity, direction, device_class,
               paper_from, paper_to, occurred_at
        FROM cheating_events WHERE student_id=%s
        ORDER BY occurred_at DESC LIMIT 50
    """, (student_id,))
    events = cur.fetchall()

    cur.execute("""
        SELECT session_id, engagement_score, looking_away_frames,
               total_frames, recorded_at
        FROM engagement_stats WHERE student_id=%s
        ORDER BY recorded_at DESC LIMIT 20
    """, (student_id,))
    engagement = cur.fetchall()

    cur.execute("""
        SELECT alert_type, COUNT(*) as cnt
        FROM alert_history WHERE student_id=%s
        GROUP BY alert_type
    """, (student_id,))
    alert_summary = {r['alert_type']: r['cnt'] for r in cur.fetchall()}

    cur.execute("SELECT COUNT(*) as cnt FROM attendance WHERE student_id=%s", (student_id,))
    attendance_count = cur.fetchone()['cnt']

    conn.close()

    def dt_str(v):
        return v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, datetime) else str(v) if v else None

    student['registered_at'] = dt_str(student['registered_at'])
    for e in events:
        e['occurred_at'] = dt_str(e['occurred_at'])
    for eg in engagement:
        eg['recorded_at'] = dt_str(eg['recorded_at'])

    return {
        'student':         student,
        'cheating_events': events,
        'engagement':      engagement,
        'alert_summary':   alert_summary,
        'sessions_attended': attendance_count,
        'subjects':        get_student_subjects(student_id),
    }


def get_all_students_summary() -> list:
    """Dashboard ke liye saare active students ka summary."""
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.name, s.registered_at, s.total_sessions,
               s.total_alerts, s.is_active,
               COUNT(DISTINCT a.session_id) as sessions_attended,
               ROUND(AVG(e.engagement_score),1) as avg_engagement
        FROM students s
        LEFT JOIN attendance a       ON a.student_id = s.id
        LEFT JOIN engagement_stats e ON e.student_id = s.id
        WHERE s.is_active=1 AND s.status='active'
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

def start_session(class_id: int = None, teacher_id: int = None) -> int:
    """Nayi session shuru karo, session_id return karo."""
    conn = get_conn()
    cur  = conn.cursor()

    # Purani table mein class_id / teacher_id nahi hoga — auto-add karo
    for col, coltype in [('class_id', 'INT DEFAULT NULL'), ('teacher_id', 'INT DEFAULT NULL')]:
        cur.execute(f"SHOW COLUMNS FROM sessions LIKE %s", (col,))
        if not cur.fetchone():
            cur.execute(f"ALTER TABLE sessions ADD COLUMN {col} {coltype}")

    cur.execute("""
        INSERT INTO sessions (started_at, class_id, teacher_id)
        VALUES (%s, %s, %s)
    """, (datetime.now(), class_id, teacher_id))
    sid = cur.lastrowid
    conn.close()
    return sid


def end_session(session_id: int, stats: dict):
    """Session band karo aur final stats save karo."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE sessions SET
            ended_at             = %s,
            duration_sec         = %s,
            total_frames         = %s,
            students_present     = %s,
            total_yaw_alerts     = %s,
            total_peeking_alerts = %s,
            total_mobile_alerts  = %s,
            total_paper_exchanges= %s
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
        session_id
    ))
    conn.close()


def get_sessions_list(limit: int = 20) -> list:
    conn = get_conn()
    cur  = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT s.id, s.started_at, s.ended_at, s.duration_sec,
               s.students_present, s.total_yaw_alerts, s.total_peeking_alerts,
               s.total_mobile_alerts, s.total_paper_exchanges,
               c.name as class_name
        FROM sessions s
        LEFT JOIN classes c ON c.id = s.class_id
        ORDER BY s.started_at DESC LIMIT %s
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
        datetime.now()
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
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT COUNT(*) as cnt FROM students WHERE is_active=1 AND status='active'")
        total_students = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM students WHERE status='pending'")
        pending_students = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM sessions")
        total_sessions = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM cheating_events")
        total_events = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM teachers WHERE is_active=1")
        total_teachers = cur.fetchone()['cnt']

        cur.execute("SELECT COUNT(*) as cnt FROM classes WHERE is_active=1")
        total_classes = cur.fetchone()['cnt']

        cur.execute("""
            SELECT event_type, COUNT(*) as cnt
            FROM cheating_events GROUP BY event_type
        """)
        event_breakdown = {r['event_type']: r['cnt'] for r in cur.fetchall()}

        cur.execute("""
            SELECT s.name, COUNT(ce.id) as alert_count
            FROM students s
            LEFT JOIN cheating_events ce ON ce.student_id = s.id
            WHERE s.is_active=1
            GROUP BY s.id ORDER BY alert_count DESC LIMIT 5
        """)
        top_offenders = cur.fetchall()

        # class_id column check — purani table mein nahi hoga
        cur.execute("SHOW COLUMNS FROM sessions LIKE 'class_id'")
        has_class_id = cur.fetchone() is not None

        if has_class_id:
            cur.execute("""
                SELECT s.id, s.started_at, s.ended_at, s.students_present,
                       COALESCE(s.total_yaw_alerts,0) + COALESCE(s.total_peeking_alerts,0) +
                       COALESCE(s.total_mobile_alerts,0) + COALESCE(s.total_paper_exchanges,0) as total_alerts,
                       c.name as class_name
                FROM sessions s
                LEFT JOIN classes c ON c.id = s.class_id
                ORDER BY s.started_at DESC LIMIT 5
            """)
        else:
            # Purani table — class_id column nahi hai, ALTER karke add karo
            cur.execute("ALTER TABLE sessions ADD COLUMN class_id INT DEFAULT NULL")
            cur.execute("""
                SELECT s.id, s.started_at, s.ended_at, s.students_present,
                       COALESCE(s.total_yaw_alerts,0) + COALESCE(s.total_peeking_alerts,0) +
                       COALESCE(s.total_mobile_alerts,0) + COALESCE(s.total_paper_exchanges,0) as total_alerts,
                       NULL as class_name
                FROM sessions s
                ORDER BY s.started_at DESC LIMIT 5
            """)
        recent_sessions = cur.fetchall()
        for r in recent_sessions:
            for k in ('started_at', 'ended_at'):
                if isinstance(r.get(k), datetime):
                    r[k] = r[k].strftime('%Y-%m-%d %H:%M:%S')

        return {
            'total_students':   total_students,
            'pending_students': pending_students,
            'total_sessions':   total_sessions,
            'total_events':     total_events,
            'total_teachers':   total_teachers,
            'total_classes':    total_classes,
            'event_breakdown':  event_breakdown,
            'top_offenders':    top_offenders,
            'recent_sessions':  recent_sessions,
        }
    except Exception as e:
        raise e
    finally:
        conn.close()   # hamesha close hoga chahe error aaye