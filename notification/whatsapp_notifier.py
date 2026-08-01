"""
whatsapp_notifier.py
====================
Smart Classroom — Twilio WhatsApp Alert System

Features:
- Har cheating event pe turant alert (yaw, peeking, mobile, paper, book)
- Same student same cheating repeat kare toh cooldown (spam nahi hoga)
- Student ki photo (snapshot) attach hoti hai message ke sath
- Student ka naam aur exact cheating type clearly likha hota hai
- Ngrok ke zariye local server se image serve hoti hai
"""

import os
import time
import uuid
import threading
from collections import defaultdict

import cv2
from dotenv import load_dotenv

load_dotenv()

# ── Twilio config ─────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM        = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TEACHER_TO         = os.getenv("TEACHER_WHATSAPP_TO", "")
NGROK_PUBLIC_URL   = os.getenv("NGROK_PUBLIC_URL", "")

# ── Cooldown: kitne seconds baad same student ka SAME event dobara alert ──
COOLDOWN_PER_EVENT = {
    "yaw":            20,    # head turn — har 20 sec mein ek baar
    "peeking":        25,    # peeking
    "mobile":         30,    # phone/watch
    "paper_exchange": 15,    # paper exchange — serious, jaldi alert
    "book":           30,    # book/notes
}

# ── In-memory state ───────────────────────────────────────────────────────
_last_alert_time: dict = defaultdict(lambda: defaultdict(float))
_last_event_time: dict = defaultdict(lambda: defaultdict(int))  # ← YEH ADD KARO
# { student_id: { event_type: timestamp } }

_snapshot_store: dict = {}
# { filename: jpeg_bytes } — Flask /api/snapshot/<filename> serve karega

_store_lock = threading.Lock()

# ── Event labels ──────────────────────────────────────────────────────────
EVENT_LABELS = {
    "yaw":            "Head Turn ↔️  (Side dekh raha tha)",
    "peeking":        "Peeking 👀  (Doosre ki taraf dekha)",
    "mobile":         "Mobile / Watch 📱  (Device pakda gaya)",
    "paper_exchange": "Paper Exchange 📄  (Parchi di ya li)",
    "book":           "Book / Notes 📚  (Kitaab ya notes use ki)",
}


# =============================================================================
# SNAPSHOT STORE  (Flask endpoint ke liye)
# =============================================================================

def get_snapshot_bytes(filename: str) -> bytes | None:
    """Flask route is function se image bytes leta hai."""
    with _store_lock:
        return _snapshot_store.get(filename)


def _store_snapshot(frame_bgr) -> str | None:
    """
    Frame ko memory mein store karo.
    Return: unique filename (Flask URL mein use hoga)
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    try:
        success, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not success:
            return None
        filename = f"snap_{uuid.uuid4().hex}.jpg"
        with _store_lock:
            _snapshot_store[filename] = buf.tobytes()
        # 5 minute baad auto-delete (memory leak nahi hoga)
        threading.Timer(300, lambda: _snapshot_store.pop(filename, None)).start()
        return filename
    except Exception as e:
        print(f"[Notifier] Snapshot store error: {e}")
        return None


# =============================================================================
# TWILIO CLIENT
# =============================================================================

def _get_client():
    try:
        from twilio.rest import Client
        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            print("[Notifier] ⚠️  TWILIO_ACCOUNT_SID ya AUTH_TOKEN .env mein nahi hai")
            return None
        return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    except ImportError:
        print("[Notifier] ⚠️  'pip install twilio' karo pehle")
        return None
    except Exception as e:
        print(f"[Notifier] Twilio client error: {e}")
        return None


# =============================================================================
# MESSAGE BUILDER
# =============================================================================

def _build_message(student_name: str, event_type: str, extra_info: str = "") -> str:
    label     = EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())
    timestamp = time.strftime("%I:%M:%S %p")
    date_str  = time.strftime("%d %b %Y")

    lines = [
        "🚨 *CHEATING ALERT* 🚨",
        "",
        f"👤 *Student:*  {student_name}",
        f"⚠️  *Cheating:*  {label}",
    ]

    if extra_info:
        lines.append(f"📝 *Detail:*  {extra_info}")

    lines += [
        "",
        f"🕐 *Time:*  {timestamp}  —  {date_str}",
        f"📷 _Student ki photo neeche attached hai_",
        "",
        "— Smart Classroom Monitor",
    ]

    return "\n".join(lines)


# =============================================================================
# SENDER  (background thread mein — main loop block nahi hoga)
# =============================================================================

def _send_async(body: str, media_url: str | None):
    def _do():
        client = _get_client()
        if not client or not TEACHER_TO:
            return
        try:
            kwargs = dict(from_=TWILIO_FROM, to=TEACHER_TO, body=body)
            if media_url:
                kwargs["media_url"] = [media_url]
            msg = client.messages.create(**kwargs)
            status = "✅ with image" if media_url else "✅ text only"
            print(f"[Notifier] WhatsApp sent {status} — SID: {msg.sid}")
        except Exception as e:
            print(f"[Notifier] ❌ Send failed: {e}")
    threading.Thread(target=_do, daemon=True).start()


# =============================================================================
# MAIN PUBLIC FUNCTION  — app.py se yeh call karo
# =============================================================================

def notify_cheating(
    student_id,
    student_name: str,
    event_type:   str,
    frame_bgr=None,
    extra_info:   str = "",
):
    """
    Koi bhi cheating detect hote hi is function ko call karo.

    Parameters
    ----------
    student_id   : unique student ID (int ya str)
    student_name : display naam (e.g. "Ali Hassan (10-A)")
    event_type   : "yaw" | "peeking" | "mobile" | "paper_exchange" | "book"
    frame_bgr    : OpenCV BGR frame — student ka cropped ya full frame
    extra_info   : extra detail string (e.g. "Left side, 28 frames")

    Cooldown logic:
    - Same student + same event type ke liye COOLDOWN_PER_EVENT seconds tak
      dobara message nahi jaayega (spam prevent)
    - Alag event type ka alag cooldown hai
    """
    if not TEACHER_TO:
        print("[Notifier] TEACHER_WHATSAPP_TO .env mein set karo")
        return

    # ── Cooldown check ────────────────────────────────────────────────────
    now      = time.time()
    cooldown = COOLDOWN_PER_EVENT.get(event_type, 20)
    last_t   = _last_alert_time[student_id][event_type]

    if now - last_t < cooldown:
        return  # same student, same cheating — abhi mat bhejo

    _last_alert_time[student_id][event_type] = now

    # ── Snapshot store karo ───────────────────────────────────────────────
    media_url = None
    if frame_bgr is not None and NGROK_PUBLIC_URL:
        filename = _store_snapshot(frame_bgr)
        if filename:
            base = NGROK_PUBLIC_URL.rstrip('/')
            media_url = f"{base}/api/snapshot/{filename}"

    # ── Message banao aur bhejo ───────────────────────────────────────────
    body = _build_message(student_name, event_type, extra_info)
    _send_async(body, media_url)

    img_status = "📷 image attached" if media_url else "⚠️  no image (NGROK_PUBLIC_URL set karo)"
    print(f"[Notifier] 📲 {student_name} | {event_type} | {img_status}")


# =============================================================================
# LEGACY WRAPPER  — purane check_and_notify() calls ke liye (backward compat)
# =============================================================================

def check_and_notify(
    student_id,
    student_name: str,
    risk_result:  dict,
    frame_bgr=None,
):
    """
    RiskEngine result se events nikaal ke notify_cheating() call karta hai.
    Yeh function pehle wale code ke saath compatible hai.
    """
    event_counts = risk_result.get("event_counts", {})
    reasons      = risk_result.get("reasons", [])
    reason_text  = "; ".join(reasons[:2]) if reasons else ""

    mapping = {
        "yaw_deviation":    "yaw",
        "peeking":          "peeking",
        "mobile_detection": "mobile",
        "paper_exchange":   "paper_exchange",
        "book_detection":   "book",
    }

    for engine_key, notif_key in mapping.items():
        current_count = event_counts.get(engine_key, 0)
        last_count = _last_event_time[student_id][engine_key]

        if current_count > last_count:  # sirf naya event aane pe
            _last_event_time[student_id][engine_key] = current_count
            notify_cheating(
                student_id   = student_id,
                student_name = student_name,
                event_type   = notif_key,
                frame_bgr    = frame_bgr,
                extra_info   = reason_text,
            )
