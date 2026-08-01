"""
SmartClass — Calculator & Smartwatch Detector
calculator.pt  →  calculator detection
watch.pt       →  smartwatch detection
"""

import cv2
import time
import torch
from collections import deque
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CALC_MODEL_PATH  = "calculator.pt"
WATCH_MODEL_PATH = "watch.pt"

CALC_CONF_THRESHOLD  = 0.60
WATCH_CONF_THRESHOLD = 0.60

# Kitne frames mein se kitne positive ho toh persistent alert fire ho
PERSISTENT_WINDOW    = 30
PERSISTENT_THRESHOLD = 0.7   # 50% frames mein detect ho

# Alert cooldown — same student ke liye dobara log kitne seconds baad
ALERT_COOLDOWN_SEC = 15.0


# ─────────────────────────────────────────────
# DETECTOR CLASS
# ─────────────────────────────────────────────

class DeviceDetector:
    """
    Calculator aur smartwatch dono detect karta hai.
    Alag alag models load karta hai — dono optional hain.
    """

    def __init__(self):
        self.calc_model  = None
        self.watch_model = None

        self.calc_loaded  = False
        self.watch_loaded = False

        # Per-student detection history
        # { student_id: { 'calculator': deque, 'watch': deque } }
        self.detection_history = {}

        # Alert cooldown tracker
        # { (student_id, device_type): last_alert_time }
        self._last_alert: dict = {}

        # Session totals
        self.total_calc_detections  = 0
        self.total_watch_detections = 0

        self._load_models()

    def _load_models(self):
        try:
            self.calc_model  = YOLO(CALC_MODEL_PATH)
            self.calc_loaded = True
            print(f"✅ Calculator model loaded: {CALC_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Calculator model not found ({CALC_MODEL_PATH}): {e}")

        try:
            self.watch_model  = YOLO(WATCH_MODEL_PATH)
            self.watch_loaded = True
            print(f"✅ Smartwatch model loaded: {WATCH_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  Smartwatch model not found ({WATCH_MODEL_PATH}): {e}")

    def is_available(self):
        return self.calc_loaded or self.watch_loaded

    # ── Detection ────────────────────────────────────────────────────────────

    def detect(self, frame):
        """
        Frame mein calculator aur watch detect karo.
        Returns list of dicts:
          { 'box': (x1,y1,x2,y2), 'confidence': float,
            'class': 'calculator'|'watch', 'center': (cx,cy) }
        """
        results = []

        if self.calc_model is not None:
            try:
                preds = self.calc_model(frame, conf=CALC_CONF_THRESHOLD, verbose=False)
                for r in preds:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        results.append({
                            'box':        (x1, y1, x2, y2),
                            'confidence': conf,
                            'class':      'calculator',
                            'center':     (cx, cy)
                        })
                        self.total_calc_detections += 1
            except Exception as e:
                print(f"Calculator detect error: {e}")

        if self.watch_model is not None:
            try:
                preds = self.watch_model(frame, conf=WATCH_CONF_THRESHOLD, verbose=False)
                for r in preds:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        results.append({
                            'box':        (x1, y1, x2, y2),
                            'confidence': conf,
                            'class':      'watch',
                            'center':     (cx, cy)
                        })
                        self.total_watch_detections += 1
            except Exception as e:
                print(f"Watch detect error: {e}")

        return results

    # ── History tracking ─────────────────────────────────────────────────────

    def update_history(self, student_id, device_type: str, detected: bool):
        if student_id not in self.detection_history:
            self.detection_history[student_id] = {
                'calculator': deque(maxlen=PERSISTENT_WINDOW),
                'watch':      deque(maxlen=PERSISTENT_WINDOW),
            }
        if device_type in self.detection_history[student_id]:
            self.detection_history[student_id][device_type].append(detected)

    def is_persistent(self, student_id, device_type: str) -> bool:
        if student_id not in self.detection_history:
            return False
        h = list(self.detection_history[student_id].get(device_type, []))
        if len(h) < 20:        # 10 se 20 karo
            return False
        return (sum(h) / len(h)) >= PERSISTENT_THRESHOLD

    # ── Alert cooldown ───────────────────────────────────────────────────────

    def should_alert(self, student_id, device_type: str) -> bool:
        key  = (student_id, device_type)
        now  = time.time()
        last = self._last_alert.get(key, 0)
        if now - last > ALERT_COOLDOWN_SEC:
            self._last_alert[key] = now
            return True
        return False

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw_detections(self, frame, detections):
        color_map = {
            'calculator': (255, 140, 0),   # orange
            'watch':      (0,   0,   255), # red
        }
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls  = det['class']
            conf = det['confidence']
            cx, cy = det['center']
            color = color_map.get(cls, (200, 200, 0))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            label = f"{cls.upper()} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            warning = ("!! CALCULATOR DETECTED !!" if cls == 'calculator'
                       else "!! SMARTWATCH DETECTED !!")
            cv2.putText(frame, warning, (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 6, color, -1)

        return frame

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self):
        flagged = {}
        for sid, hist in self.detection_history.items():
            flagged_types = [t for t in ('calculator', 'watch')
                             if self.is_persistent(sid, t)]
            if flagged_types:
                flagged[sid] = flagged_types

        return {
            'calc_model_loaded':       self.calc_loaded,
            'watch_model_loaded':      self.watch_loaded,
            'total_calc_detections':   self.total_calc_detections,
            'total_watch_detections':  self.total_watch_detections,
            'students_with_history':   len(self.detection_history),
            'students_flagged':        flagged,
        }

    def reset(self):
        self.detection_history.clear()
        self._last_alert.clear()
        self.total_calc_detections  = 0
        self.total_watch_detections = 0