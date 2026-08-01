"""
Smart Classroom - Flask Backend API
MySQL + Risk Scoring Engine
"""

from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import cv2
import torch
import numpy as np
from collections import deque
import time
import os
import threading
import platform
import csv
from datetime import datetime
from scipy.spatial.distance import cosine
import json
from whatsapp_notifier import check_and_notify
from head_eye_tracker import process_face_eye_head, StudentEyeHeadTracker
from device_detector import DeviceDetector
# MySQL database layer
import database as db

# ============================================
# RISK SCORING ENGINE
# ============================================
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

RISK_WEIGHTS = {
    'mobile_detection': 40.0,
    'paper_exchange':   30.0,
    'book_detection':   18.0,
    'peeking':          22.0,
    'yaw_deviation':    14.0,
    'repeated_offense':  8.0,
}
RISK_HALF_LIFE = {
    'mobile_detection': 15.0,
    'paper_exchange':   60.0,
    'book_detection':   20.0,
    'peeking':          12.0,
    'yaw_deviation':     8.0,
    'repeated_offense': 30.0,
}
RISK_EMA_ALPHA        = 0.25
RISK_THRESHOLD_YELLOW = 35.0
RISK_THRESHOLD_RED    = 65.0
YAW_SUSTAINED_FRAMES  = 12
PEEK_RATIO_THRESHOLD  = 0.50


@dataclass
class _TimedEvent:
    signal_type: str
    raw_weight:  float
    timestamp:   float = field(default_factory=time.time)
    note:        str   = ''

    def decayed_weight(self, now: float) -> float:
        age = max(0.0, now - self.timestamp)
        hl  = RISK_HALF_LIFE.get(self.signal_type, 30.0)
        return self.raw_weight * math.pow(0.5, age / hl)


@dataclass
class _StudentRiskState:
    student_id:        object
    smoothed_score:    float = 0.0
    events:            list  = field(default_factory=list)
    event_type_counts: dict  = field(default_factory=lambda: defaultdict(int))

    def record(self, ev: '_TimedEvent'):
        self.events.append(ev)
        self.event_type_counts[ev.signal_type] += 1
        now = time.time()
        self.events = [e for e in self.events if now - e.timestamp < 120]

    def raw_score(self, now: float) -> float:
        return sum(e.decayed_weight(now) for e in self.events)

    def normalized_score(self, now: float) -> float:
        raw = self.raw_score(now)
        self.smoothed_score = RISK_EMA_ALPHA * raw + (1 - RISK_EMA_ALPHA) * self.smoothed_score
        sig = 1.0 / (1.0 + math.exp(-0.025 * (self.smoothed_score - 80.0)))
        return round(min(100.0, max(0.0, sig * 130.0)), 1)


class RiskEngine:
    def __init__(self):
        self._states: Dict[object, _StudentRiskState] = {}

    def _get(self, sid) -> _StudentRiskState:
        if sid not in self._states:
            self._states[sid] = _StudentRiskState(student_id=sid)
        return self._states[sid]

    def _repeat_bonus(self, signal_type: str, st: _StudentRiskState) -> float:
        n = st.event_type_counts.get(signal_type, 0)
        return RISK_WEIGHTS['repeated_offense'] * min(1.0, math.log1p(n) / math.log1p(5)) if n else 0.0

    def update(self, student_id, signals: dict) -> dict:
        st  = self._get(student_id)
        now = time.time()

        if signals.get('mobile_detected'):
            conf = float(signals.get('mobile_confidence', 0.7))
            w    = RISK_WEIGHTS['mobile_detection'] * min(1.0, conf / 0.6)
            st.record(_TimedEvent('mobile_detection', w + self._repeat_bonus('mobile_detection', st),
                                  now, note=f'conf={conf:.0%}'))

        if signals.get('paper_exchange'):
            w = RISK_WEIGHTS['paper_exchange']
            st.record(_TimedEvent('paper_exchange', w + self._repeat_bonus('paper_exchange', st), now))

        if signals.get('book_detected'):
            conf = float(signals.get('book_confidence', 0.5))
            w    = RISK_WEIGHTS['book_detection'] * min(1.0, conf / 0.4)
            st.record(_TimedEvent('book_detection', w, now, note=f'conf={conf:.0%}'))

        if signals.get('peeking'):
            ratio     = float(signals.get('peeking_ratio', PEEK_RATIO_THRESHOLD))
            intensity = min(1.0, (ratio - PEEK_RATIO_THRESHOLD) / 0.3 + 0.5)
            w         = RISK_WEIGHTS['peeking'] * intensity
            direction = signals.get('peeking_direction', '')
            st.record(_TimedEvent('peeking', w + self._repeat_bonus('peeking', st),
                                  now, note=f'dir={direction} ratio={ratio:.0%}'))

        yaw_dir    = signals.get('yaw_direction', 'Center')
        yaw_frames = int(signals.get('yaw_consecutive_frames', 0))
        if yaw_dir != 'Center' and yaw_frames >= YAW_SUSTAINED_FRAMES and yaw_frames % 15 == 0:
            intensity = min(1.0, yaw_frames / 60.0)
            st.record(_TimedEvent('yaw_deviation',
                                  RISK_WEIGHTS['yaw_deviation'] * intensity, now,
                                  note=f'dir={yaw_dir} frames={yaw_frames}'))

        score = st.normalized_score(now)
        level = 'RED' if score >= RISK_THRESHOLD_RED else ('YELLOW' if score >= RISK_THRESHOLD_YELLOW else 'GREEN')
        reasons = self._reasons(st, signals, now)

        return {
            'student_id':   student_id,
            'risk_score':   score,
            'risk_level':   level,
            'reasons':      reasons,
            'event_counts': dict(st.event_type_counts),
        }

    def _reasons(self, st: _StudentRiskState, signals: dict, now: float) -> List[str]:
        reasons = []
        recent: Dict[str, list] = defaultdict(list)
        for ev in st.events:
            if now - ev.timestamp <= 60.0:
                recent[ev.signal_type].append(ev)

        if recent.get('mobile_detection'):
            n = len(recent['mobile_detection'])
            confs = []
            for e in recent['mobile_detection']:
                if 'conf=' in e.note:
                    try:
                        confs.append(float(e.note.split('conf=')[1].rstrip('%')) / 100)
                    except Exception:
                        pass
            avg_c = sum(confs) / len(confs) if confs else 0.0
            reasons.append(f'Mobile/watch detected {n} time(s) in last 60 s (avg conf {avg_c:.0%})')

        if recent.get('paper_exchange'):
            reasons.append(f'Paper exchange confirmed {len(recent["paper_exchange"])} time(s)')

        if recent.get('peeking'):
            dirs = []
            for e in recent['peeking']:
                if 'dir=' in e.note:
                    dirs.append(e.note.split('dir=')[1].split()[0])
            primary = max(set(dirs), key=dirs.count) if dirs else '?'
            reasons.append(f'Sustained peeking behaviour (primary direction: {primary})')

        if recent.get('yaw_deviation'):
            yaw_frames = signals.get('yaw_consecutive_frames', '?')
            yaw_dir    = signals.get('yaw_direction', '')
            reasons.append(f'Continuous head turn {yaw_dir} ({yaw_frames} consecutive frame(s))')

        if recent.get('book_detection'):
            reasons.append(f'Book/notes visible near student ({len(recent["book_detection"])} detection(s))')

        multi = [t for t, c in st.event_type_counts.items() if c >= 3]
        if multi:
            label_map = {'mobile_detection': 'mobile', 'paper_exchange': 'paper exchange',
                         'peeking': 'peeking', 'yaw_deviation': 'head-turning', 'book_detection': 'book'}
            labels = [label_map.get(t, t) for t in multi]
            reasons.append(f'Repeat offender: {", ".join(labels)} flagged 3+ times this session')

        if not reasons:
            reasons.append('No significant cheating signals detected')
        return reasons

    def reset_all(self):
        self._states.clear()

    def snapshot(self) -> list:
        now = time.time()
        out = []
        for sid, st in self._states.items():
            score = st.normalized_score(now)
            level = 'RED' if score >= RISK_THRESHOLD_RED else ('YELLOW' if score >= RISK_THRESHOLD_YELLOW else 'GREEN')
            out.append({'student_id': sid, 'risk_score': score, 'risk_level': level,
                        'event_counts': dict(st.event_type_counts)})
        return sorted(out, key=lambda x: x['risk_score'], reverse=True)


# ============================================
# DETECTION CLASSES
# ============================================

class MobileWatchDetector:
    def __init__(self, model):
        self.model = model
        self.detection_history = {}
        self.active_alerts = {}
        self.total_detections = 0

    def detect(self, frame, confidence_threshold=0.45):
        if self.model is None:
            return []
        try:
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls[0])]
                    if cls_name != 'cell phone':   # ← YE LINE ADD KAR
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls_name,
                        'center': (cx, cy)
                    })
                    self.total_detections += 1
            return detections
        except Exception as e:
            print(f"MobileWatchDetector error: {e}")
            return []

    def update_history(self, student_id, detected: bool):
        if student_id not in self.detection_history:
            self.detection_history[student_id] = deque(maxlen=30)
        self.detection_history[student_id].append(detected)

    def is_persistent(self, student_id, threshold=0.6):
        if student_id not in self.detection_history:
            return False
        h = list(self.detection_history[student_id])
        if len(h) < 10:
            return False
        return (sum(h) / len(h)) >= threshold

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls  = det['class']
            conf = det['confidence']
            cx, cy = det['center']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"{cls.upper()} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 8, y1), (0, 0, 200), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "!! PROHIBITED DEVICE !!",
                        (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        return frame

    def get_stats(self):
        return {
            'total_detections':       self.total_detections,
            'students_with_history':  len(self.detection_history),
            'active_alerts':          len(self.active_alerts),
        }


class PaperExchangeDetector:
    def __init__(self, time_threshold=0.5):
        self.time_threshold    = time_threshold
        self.paper_owner       = {}
        self.owner_change_time = {}
        self.event_logged      = {}
        self.alert_time_tracker = {}
        self.total_exchanges   = 0
        self.exchange_events   = []

    def process_frame(self, paper_results, person_results, current_time_sec):
        exchanges = []
        if paper_results.boxes.id is None or person_results.boxes.id is None:
            return exchanges

        paper_boxes  = paper_results.boxes.xyxy.cpu().numpy()
        paper_ids    = paper_results.boxes.id.cpu().numpy()
        person_boxes = person_results.boxes.xyxy.cpu().numpy()
        person_ids   = person_results.boxes.id.cpu().numpy()
        person_cls   = person_results.boxes.cls.cpu().numpy()

        persons = [
            (person_boxes[i], int(person_ids[i]))
            for i in range(len(person_cls)) if person_cls[i] == 0
        ]

        for i, p_box in enumerate(paper_boxes):
            p_id = int(paper_ids[i])
            x1, y1, x2, y2 = map(int, p_box)
            px_c = (x1 + x2) / 2
            py_c = (y1 + y2) / 2
            current_owner = None

            for pb, pid in persons:
                if pb[0] < px_c < pb[2] and pb[1] < py_c < pb[3]:
                    current_owner = pid
                    break

            if p_id not in self.paper_owner:
                self.paper_owner[p_id]          = current_owner
                self.event_logged[p_id]         = False
                self.alert_time_tracker[p_id]   = None
            else:
                old = self.paper_owner[p_id]
                if old != current_owner and current_owner is not None:
                    if p_id not in self.owner_change_time or self.owner_change_time[p_id] is None:
                        self.owner_change_time[p_id] = time.time()
                    elif time.time() - self.owner_change_time[p_id] > self.time_threshold:
                        if not self.event_logged[p_id]:
                            self.total_exchanges += 1
                            self.event_logged[p_id]       = True
                            self.alert_time_tracker[p_id] = current_time_sec
                            ev = {
                                'timestamp': current_time_sec,
                                'paper_id':  p_id,
                                'old_owner': old,
                                'new_owner': current_owner,
                                'position':  (int(px_c), int(py_c))
                            }
                            self.exchange_events.append(ev)
                            exchanges.append(ev)
                            self.paper_owner[p_id]       = current_owner
                            self.owner_change_time[p_id] = None
                else:
                    self.owner_change_time[p_id] = None
                    self.event_logged[p_id]      = False

        return exchanges

    def draw_exchange(self, frame, event):
        ex, ey = event['position']
        cv2.circle(frame, (ex, ey), 50, (0, 0, 255), 4)
        cv2.circle(frame, (ex, ey), 8, (0, 0, 255), -1)
        cv2.putText(frame, "PAPER EXCHANGE!", (ex - 120, ey - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        cv2.putText(frame, f"P{event['old_owner']} -> P{event['new_owner']}",
                    (ex - 100, ey - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return frame

    def get_active_alerts(self, current_time_sec, alert_duration=2.0):
        return [pid for pid, t in self.alert_time_tracker.items()
                if t is not None and current_time_sec - t <= alert_duration]


class BookDetector:
    def __init__(self, model, book_class_names=None):
        self.model            = model
        self.book_class_names = book_class_names or ['book', 'notebook', 'textbook']
        self.alert_cooldown   = {}
        self.total_detections = 0
        self.detection_events = []

    def detect(self, frame, confidence_threshold=0.30):
        if self.model is None:
            return []
        try:
            results    = self.model(frame, conf=confidence_threshold, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_name = self.model.names[int(box.cls[0])]
                    if cls_name.lower() not in self.book_class_names:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cx   = (x1 + x2) // 2
                    cy   = (y1 + y2) // 2
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls_name,
                        'center': (cx, cy)
                    })
                    self.total_detections += 1
            return detections
        except Exception as e:
            print(f"BookDetector error: {e}")
            return []

    def draw_detections(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls  = det['class']
            conf = det['confidence']
            cx, cy = det['center']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 3)
            label = f"{cls.upper()} {conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 14), (x1 + lw + 8, y1), (200, 80, 0), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "!! BOOK/NOTES DETECTED !!",
                        (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (255, 120, 0), -1)
        return frame

    def should_alert(self, track_id, cooldown_sec=5.0):
        now  = time.time()
        last = self.alert_cooldown.get(track_id, 0)
        if now - last > cooldown_sec:
            self.alert_cooldown[track_id] = now
            return True
        return False


# ============================================
# GLOBAL STATE
# ============================================
state = {
    'student_database_facenet': {},
    'next_id':                  1,
    'student_data':             {},
    'activity_log':             deque(maxlen=100),
    'frame_count':              0,
    'start_time':               None,
    'current_session_id':       None,
    'unknown_student_counter':  1000,
    'unknown_students':         {},
    'yaw_cheating_counter':     {},
    'yaw_alert_active':         {},
    'yaw_alert_time':           {},
    'eye_head_trackers': {},
    'alarm_playing':            {},
    'monitoring_active':        False,
    'latest_frame':             None,
    'mobile_detector':          None,
    'paper_detector':           None,
    'book_detector':            None,
    'device_detector': None,   # ← ye add karo
    'current_exam_name':        None,
    'current_exam_type':        'quiz',
    'risk_engine':              RiskEngine(),
    'risk_scores':              {},
    'latest_stats': {
        'person_count':           0,
        'recognized_count':       0,
        'peeking_count':          0,
        'yaw_alert_count':        0,
        'paper_exchange_count':   0,
        'mobile_count':           0,
        'hands_detected':         0,
        'session_time':           0,
        'total_mobile_detections':0,
        'total_paper_exchanges':  0,
        'total_yaw_alerts':       0,
        'total_peeking_alerts':   0,
        'book_count':             0,
        'total_book_detections':  0,
    },
}

app = Flask(__name__)
CORS(app)

# ============================================
# DATABASE INIT + STARTUP LOAD
# ============================================
try:
    db.init_db()
    state['student_database_facenet'] = db.load_all_students()
    if state['student_database_facenet']:
        state['next_id'] = max(state['student_database_facenet'].keys()) + 1
    print(f"✅ MySQL connected — {len(state['student_database_facenet'])} students loaded")
except Exception as e:
    print(f"⚠️  MySQL connection failed: {e}")
    print("    Check database.py config (MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD)")

# ============================================
# LAZY MODEL LOADING
# ============================================
_models = {}
_face_det_lock = threading.Lock()
_reg_cap = None
_reg_cap_lock = threading.Lock()

def get_register_cap():
    global _reg_cap
    if _reg_cap is None or not _reg_cap.isOpened():
        _reg_cap = cv2.VideoCapture(0)
    return _reg_cap

def get_models():
    if _models:
        return _models

    try:
        import ultralytics.nn.tasks as tasks
        _orig = tasks.torch_safe_load
        def patched(file, *args, **kwargs):
            try:
                return torch.load(file, map_location='cpu', weights_only=False), file
            except:
                return _orig(file, *args, **kwargs)
        tasks.torch_safe_load = patched
    except:
        pass

    from ultralytics import YOLO
    import mediapipe as mp

    try:
        _models['pose']   = YOLO("yolov8s-pose.pt")
        _models['person'] = YOLO("yolov8n.pt")
    except Exception as e:
        print(f"YOLO load error: {e}")

    try:
        _models['paper']        = YOLO("best.pt")
        _models['paper_loaded'] = True
    except:
        _models['paper']        = None
        _models['paper_loaded'] = False

    # try:
    #     _models['custom']        = YOLO("best1.pt")
    #     _models['custom_loaded'] = True
    # except:
    #     _models['custom']        = None
    #     _models['custom_loaded'] = False

    try:
        from facenet_pytorch import InceptionResnetV1
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _models['facenet'] = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
        _models['device']  = dev
    except Exception as e:
        _models['facenet'] = None
        _models['device']  = 'cpu'

    try:
        prototxt    = "deploy.prototxt"
        caffemodel  = "res10_300x300_ssd_iter_140000.caffemodel"
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            _models['face_det'] = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        else:
            _models['face_det'] = None
    except:
        _models['face_det'] = None

    mp_fm = mp.solutions.face_mesh
    mp_h  = mp.solutions.hands
    _models['face_mesh'] = mp_fm.FaceMesh(
        max_num_faces=10, refine_landmarks=True,
        min_detection_confidence=0.4, min_tracking_confidence=0.4)
    _models['hands'] = mp_h.Hands(
        max_num_hands=20, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    _models['mp_draw']          = mp.solutions.drawing_utils
    _models['mp_draw_styles']   = mp.solutions.drawing_styles
    _models['mp_hands_mod']     = mp_h
    _models['mp_face_mesh_mod'] = mp_fm

    # if state['mobile_detector'] is None and _models.get('custom_loaded'):
    #     state['mobile_detector'] = MobileWatchDetector(_models['custom'])

    ## NEW — paste this after book_detector init:
    

    if state['book_detector'] is None:
        person_model = _models.get('person')
        if person_model:
            state['book_detector'] = BookDetector(person_model, book_class_names=['book'])
            print("📚 BookDetector ready — yolov8n.pt (COCO book class 73)")
    # Mobile detector — BOOK KE BAAD
    if state['mobile_detector'] is None:
        person_model = _models.get('person')
        if person_model:
            state['mobile_detector'] = MobileWatchDetector(person_model)
            print("📱 MobileWatchDetector ready — yolov8n COCO (cell phone class)")

    if state['paper_detector'] is None:
        state['paper_detector'] = PaperExchangeDetector(time_threshold=0.5)
    if state['device_detector'] is None:
        state['device_detector'] = DeviceDetector()
    return _models

# ============================================
# DETECTION HELPERS
# ============================================

def detect_faces(frame):
    m = get_models()
    if not m.get('face_det'):
        return []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))
    with _face_det_lock:
        m['face_det'].setInput(blob)
        dets = m['face_det'].forward()
    faces = []
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf > 0.5:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                faces.append({'box': (x1, y1, x2, y2), 'confidence': float(conf)})
    return faces


def get_face_embedding(face_tensor):
    m = get_models()
    if not m.get('facenet'):
        return None
    try:
        with torch.no_grad():
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            emb = m['facenet'](face_tensor.to(m['device']))
            return emb.cpu().numpy().flatten()
    except:
        return None


def register_face(frame, student_id, student_name, roll_number='', department=''):
    faces = detect_faces(frame)
    if not faces:
        return False, "No face detected"
    best = max(faces, key=lambda x: x['confidence'])
    if best['confidence'] < 0.9:
        return False, f"Low confidence: {best['confidence']*100:.1f}%"
    x1, y1, x2, y2 = best['box']
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False, "Invalid face crop"
    face_rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    t = torch.from_numpy(face_resized).float()
    t = (t - 127.5) / 128.0
    t = t.permute(2, 0, 1)
    emb = get_face_embedding(t)
    if emb is not None:
        if student_id not in state['student_database_facenet']:
            state['student_database_facenet'][student_id] = {'name': student_name, 'embeddings': []}
        state['student_database_facenet'][student_id]['embeddings'].append(emb)
        try:
            db.save_student(student_id, student_name,
                            state['student_database_facenet'][student_id]['embeddings'],
                            roll_number=roll_number,
                            department=department)
            print(f"✅ Student saved: {student_name} (ID:{student_id})")
        except Exception as e:
            print(f"❌ DB save error: {e}")
        return True, f"Captured (conf: {best['confidence']*100:.1f}%)"
    return False, "Embedding failed"


def recognize_face(frame, box):
    if not state['student_database_facenet']:
        return None, 0.0
    m = get_models()
    if not m.get('facenet'):
        return None, 0.0
    try:
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, 0.0
        face_rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        t = torch.from_numpy(face_resized).float()
        t = (t - 127.5) / 128.0
        t = t.permute(2, 0, 1)
        emb = get_face_embedding(t)
        if emb is not None:
            best_id, best_sim = None, 0.0
            for sid, data in state['student_database_facenet'].items():
                for stored in data['embeddings']:
                    sim = 1 - cosine(emb, stored)
                    if sim > best_sim and sim > 0.6:
                        best_sim = sim
                        best_id  = sid
            if best_id:
                return best_id, best_sim * 100
    except:
        pass
    return None, 0.0


def get_head_yaw(landmarks, image_width):
    eye_cx = (landmarks[33].x + landmarks[263].x) / 2
    return (landmarks[1].x - eye_cx) * image_width


def detect_head_direction(face_landmarks, image_width=None):
    lm = face_landmarks.landmark
    if image_width:
        yaw = get_head_yaw(lm, image_width)
        if yaw < -20: return 'LEFT'
        if yaw >  20: return 'RIGHT'
    nose      = lm[1]
    left_eye  = lm[33]
    right_eye = lm[263]
    eye_cx = (left_eye.x + right_eye.x) / 2
    eye_cy = (left_eye.y + right_eye.y) / 2
    h_dev  = nose.x - eye_cx
    v_dev  = nose.y - eye_cy
    if abs(h_dev) > 0.04:
        return 'LEFT' if h_dev < 0 else 'RIGHT'
    if abs(v_dev) > 0.03:
        return 'DOWN' if v_dev > 0 else 'UP'
    return 'CENTER'


def play_alarm():
    try:
        if platform.system() == "Windows":
            import winsound
            for _ in range(3):
                winsound.Beep(1200, 300)
                winsound.Beep(800, 300)
        else:
            for _ in range(3):
                print('\a', end='', flush=True)
                time.sleep(0.3)
    except:
        pass


# ============================================
# LIVE MONITORING THREAD
# ============================================

peeking_history = {}


def monitoring_thread():
    m   = get_models()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        state['monitoring_active'] = False
        return

    try:
        session_id = db.start_session(
            exam_name    = state.get('current_exam_name'),
            exam_type    = state.get('current_exam_type', 'quiz'),
            classroom_id = state.get('current_classroom_id'),
        )
        state['current_session_id'] = session_id
        print(f"📊 DB Session started: #{session_id}")
    except Exception as e:
        print(f"⚠️  Could not start DB session: {e}")
        session_id = None

    state['start_time']   = time.time()
    YAW_THRESH            = 15
    CHEAT_FRAMES          = 20
    ALERT_DUR             = 2
    session_mobile_total  = 0
    session_paper_total   = 0

    student_yaw_frames: Dict[object, int] = defaultdict(int)
    student_yaw_dir:    Dict[object, str] = defaultdict(lambda: 'Center')

    while state['monitoring_active']:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        state['frame_count'] += 1
        fh, fw           = frame.shape[:2]
        rgb              = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time_sec = state['frame_count'] / 15.0

        t = fw // 3
        cv2.line(frame, (t, 0),    (t, fh),    (255, 255, 0), 2)
        cv2.line(frame, (t*2, 0),  (t*2, fh),  (255, 255, 0), 2)
        for txt, pos in [("LEFT",  (10, fh-10)),
                         ("CENTER",(t+10, fh-10)),
                         ("RIGHT", (t*2+10, fh-10))]:
            cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        stats = state['latest_stats']
        stats['person_count']         = 0
        stats['recognized_count']     = 0
        stats['peeking_count']        = 0
        stats['yaw_alert_count']      = 0
        stats['hands_detected']       = 0
        stats['mobile_count']         = 0
        stats['paper_exchange_count'] = 0
        stats['book_count']           = 0
        stats['session_time']         = int(time.time() - state['start_time'])

        # ── 1. FACE DETECTION + RECOGNITION ──────────────────────────────────
        faces            = detect_faces(frame)
        recognized_faces = []
        for face in faces:
            x1, y1, x2, y2 = face['box']
            sid, conf = recognize_face(frame, face['box'])
            cx, cy    = (x1+x2)/2, (y1+y2)/2
            stats['person_count'] += 1

            if sid:
                stats['recognized_count'] += 1
                sname  = state['student_database_facenet'][sid]['name']
                is_unk = False
                color  = (0, 255, 0) if conf > 85 else (0, 255, 255)
            else:
                sid    = f"Unk_{int(cx)}"
                sname  = sid
                conf   = 0
                is_unk = True
                color  = (128, 128, 128)

            recognized_faces.append({'box':(x1,y1,x2,y2), 'student_id':sid,
                                      'student_name':sname, 'confidence':conf,
                                      'is_unknown':is_unk})
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
            cv2.putText(frame, sname, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if not is_unk:
                cv2.putText(frame, f"{conf:.1f}%", (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── 2. FACE MESH + HEAD DIRECTION + YAW ──────────────────────────────
        # ── 2. FACE MESH + HEAD DIRECTION + YAW ──────────────────────────────
        face_results = m['face_mesh'].process(rgb)
        if face_results.multi_face_landmarks:
            for fidx, face_lm in enumerate(face_results.multi_face_landmarks):
                xc   = [lm.x*fw for lm in face_lm.landmark]
                yc   = [lm.y*fh for lm in face_lm.landmark]
                mx1, my1 = int(min(xc)), int(min(yc))
                mx2, my2 = int(max(xc)), int(max(yc))

                matched_id = None
                for fi in recognized_faces:
                    fx1,fy1,fx2,fy2 = fi['box']
                    ox = max(0, min(mx2,fx2) - max(mx1,fx1))
                    oy = max(0, min(my2,fy2) - max(my1,fy1))
                    if ox > 50 and oy > 50:
                        matched_id = fi['student_id']
                        break

                hdir = detect_head_direction(face_lm, fw)

                if matched_id:
                    # ── peeking history update ─────────────────────────
                    if matched_id not in peeking_history:
                        peeking_history[matched_id] = deque(maxlen=90)
                    peeking_history[matched_id].append(hdir)

                    # ── eye + head tracker ─────────────────────────────
                    if not (isinstance(matched_id, str) and matched_id.startswith('Unk_')):
                        try:
                            face_data = process_face_eye_head(
                                frame          = frame,
                                face_landmarks = face_lm,
                                face_bbox      = (mx1, my1, mx2, my2),
                                student_id     = matched_id,
                                tracker_dict   = state['eye_head_trackers'],
                                img_w          = fw,
                                img_h          = fh,
                                draw_iris      = True,
                                draw_pose      = True,
                                draw_blink     = True,
                                yaw_cheating_threshold = 20.0,
                                pitch_down_threshold   = 25.0,
                                drowsy_duration_alert  = 3.0
                            )
                            for alert in face_data['alerts']:
                                sname = state['student_database_facenet'].get(
                                    matched_id, {}).get('name', str(matched_id))
                                state['activity_log'].append(
                                    f"{alert} — {sname} at {time.strftime('%H:%M:%S')}"
                                )
                        except Exception as _eye_err:
                            pass

                    # ── peeking detection ──────────────────────────────
                    hist = list(peeking_history[matched_id])
                    if len(hist) >= 30:
                        lr = sum(1 for h in hist if h == 'LEFT')  / len(hist)
                        rr = sum(1 for h in hist if h == 'RIGHT') / len(hist)
                        if lr >= 0.5 or rr >= 0.5:
                            stats['peeking_count'] += 1
                            stats['total_peeking_alerts'] = stats.get('total_peeking_alerts', 0) + 1
                            peek_dir = 'LEFT' if lr > rr else 'RIGHT'
                            cv2.putText(frame, f"PEEKING {peek_dir}!", (mx1, my2+30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                            cv2.rectangle(frame, (mx1,my1), (mx2,my2), (0,0,255), 3)
                            try:
                                sname = state['student_database_facenet'].get(
                                    matched_id, {}).get('name', str(matched_id))
                                db.log_cheating_event(
                                    session_id, matched_id, sname, 'peeking', severity='MEDIUM',
                                    direction=peek_dir, frame_number=state['frame_count'],
                                    timestamp_sec=current_time_sec)
                                db.log_alert(session_id, matched_id, sname,
                                    'peeking', f"Peeking {peek_dir} ({int(max(lr,rr)*100)}%)")
                                db.mark_attendance(session_id, matched_id, sname)
                            except Exception as _e:
                                pass

                    dir_color = (0,165,255) if hdir in ['LEFT','RIGHT'] else (0,255,0)
                    cv2.putText(frame, f"Head:{hdir}", (mx1, my2+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 2)

                # ── yaw counter (fidx based) ───────────────────────────
                yaw = get_head_yaw(face_lm.landmark, fw)
                if fidx not in state['yaw_cheating_counter']:
                    state['yaw_cheating_counter'][fidx] = 0
                    state['yaw_alert_active'][fidx]     = False
                    state['yaw_alert_time'][fidx]       = 0
                    state['alarm_playing'][fidx]        = False

                direction = ("Looking Left"  if yaw < -YAW_THRESH else
                             "Looking Right" if yaw >  YAW_THRESH else "Center")

                if direction != "Center":
                    state['yaw_cheating_counter'][fidx] += 1
                else:
                    state['yaw_cheating_counter'][fidx] = 0
                    state['alarm_playing'][fidx] = False

                if matched_id and not (isinstance(matched_id, str) and matched_id.startswith('Unk_')):
                    if direction != 'Center':
                        student_yaw_frames[matched_id] += 1
                        student_yaw_dir[matched_id]     = direction
                    else:
                        student_yaw_frames[matched_id] = 0
                        student_yaw_dir[matched_id]    = 'Center'

                if (state['yaw_cheating_counter'][fidx] > CHEAT_FRAMES
                        and not state['yaw_alert_active'][fidx]):
                    state['yaw_alert_active'][fidx] = True
                    state['yaw_alert_time'][fidx]   = time.time()
                    if not state['alarm_playing'][fidx]:
                        state['alarm_playing'][fidx] = True
                        threading.Thread(target=play_alarm, daemon=True).start()
                    log = (f"🚨 YAW ALERT: Face {fidx} cheating ({direction}) "
                           f"at {time.strftime('%H:%M:%S')}")
                    state['activity_log'].append(log)
                    stats['total_yaw_alerts'] = stats.get('total_yaw_alerts', 0) + 1
                    try:
                        if matched_id:
                            sname = state['student_database_facenet'].get(
                                matched_id, {}).get('name', str(matched_id))
                            db.log_cheating_event(
                                session_id, matched_id, sname, 'yaw', severity='HIGH',
                                direction=direction, frame_number=state['frame_count'],
                                timestamp_sec=current_time_sec)
                            db.log_alert(session_id, matched_id, sname,
                                'yaw', f"Yaw cheating: {direction}")
                            db.mark_attendance(session_id, matched_id, sname)
                    except Exception as _e:
                        pass

                nose = face_lm.landmark[1]
                nx, ny  = int(nose.x*fw), int(nose.y*fh)
                dir_col = (0,255,0) if direction == "Center" else (0,0,255)
                cv2.putText(frame, f"Face {fidx}: {direction}", (nx-60, ny-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_col, 2)

                if state['yaw_alert_active'].get(fidx, False):
                    stats['yaw_alert_count'] += 1
                    if time.time() - state['yaw_alert_time'][fidx] < ALERT_DUR:
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (nx-140, ny-80), (nx+140, ny-15), (0,0,180), -1)
                        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                        cv2.putText(frame, f"CHEATING! Face {fidx}", (nx-130, ny-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
                    else:
                        state['yaw_alert_active'][fidx]     = False
                        state['yaw_cheating_counter'][fidx] = 0


        # ── 3. HAND LANDMARKS ─────────────────────────────────────────────────
        hand_results = m['hands'].process(rgb)
        if hand_results.multi_hand_landmarks:
            stats['hands_detected'] = len(hand_results.multi_hand_landmarks)
            for hand_lm in hand_results.multi_hand_landmarks:
                m['mp_draw'].draw_landmarks(
                    frame, hand_lm, m['mp_hands_mod'].HAND_CONNECTIONS,
                    m['mp_draw_styles'].get_default_hand_landmarks_style(),
                    m['mp_draw_styles'].get_default_hand_connections_style())

        
                
        # ── 4b. CALCULATOR + WATCH DETECTION ──────────────────────────────────
        dev_det = state.get('device_detector')
        if dev_det is not None and dev_det.is_available():
            dev_detections = dev_det.detect(frame)
            if dev_detections:
                frame = dev_det.draw_detections(frame, dev_detections)
                for det in dev_detections:
                    cls    = det['class']
                    dx, dy = det['center']
                    min_dist, closest_sid = float('inf'), None
                    for fi in recognized_faces:
                        fx1, fy1, fx2, fy2 = fi['box']
                        fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
                        d = np.sqrt((dx-fcx)**2 + (dy-fcy)**2)
                        if d < min_dist:
                            min_dist    = d
                            closest_sid = fi['student_id']
                    if closest_sid and min_dist < fw * 0.4:
                        dev_det.update_history(closest_sid, cls, True)
                        if dev_det.is_persistent(closest_sid, cls):
                            if dev_det.should_alert(closest_sid, cls):
                                sname = state['student_database_facenet'].get(
                                    closest_sid, {}).get('name', str(closest_sid))
                                state['activity_log'].append(
                                    f"🚨 {cls.upper()} DETECTED: {sname} "
                                    f"at {time.strftime('%H:%M:%S')}")
                                try:
                                    db.log_cheating_event(
                                        session_id, closest_sid, sname,
                                        cls, severity='HIGH',
                                        device_class=cls,
                                        confidence=det['confidence'],
                                        frame_number=state['frame_count'],
                                        timestamp_sec=current_time_sec)
                                    db.log_alert(session_id, closest_sid, sname,
                                        cls, f"{cls} detected ({det['confidence']:.0%})")
                                    db.mark_attendance(session_id, closest_sid, sname)
                                except Exception as _e:
                                    pass
                    else:
                        if closest_sid:
                            dev_det.update_history(closest_sid, cls, False)


        # ── 5. PAPER EXCHANGE DETECTION ───────────────────────────────────────
        paper_det    = state.get('paper_detector')
        paper_model  = m.get('paper')
        person_model = m.get('person')

        if paper_det is not None and paper_model is not None and person_model is not None:
            try:
                per_res   = person_model.track(frame, persist=True, verbose=False, conf=0.5)
                paper_res = paper_model.track(frame, persist=True, verbose=False, conf=0.5)

                if per_res and paper_res:
                    for box in paper_res[0].boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
                        cv2.putText(frame, "PAPER", (px1, py1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    exchanges = paper_det.process_frame(paper_res[0], per_res[0], current_time_sec)
                    for ev in exchanges:
                        stats['paper_exchange_count'] += 1
                        session_paper_total           += 1
                        stats['total_paper_exchanges'] = session_paper_total
                        frame = paper_det.draw_exchange(frame, ev)
                        state['activity_log'].append(
                            f"📄 PAPER EXCHANGE: Person {ev['old_owner']} → {ev['new_owner']} "
                            f"at {int(current_time_sec//60)}:{int(current_time_sec%60):02d}")
                        try:
                            db.log_cheating_event(
                                session_id, ev['old_owner'], f"Person {ev['old_owner']}",
                                'paper_exchange', severity='HIGH',
                                paper_from=ev['old_owner'], paper_to=ev['new_owner'],
                                frame_number=state['frame_count'],
                                timestamp_sec=current_time_sec)
                            db.log_alert(session_id, ev['old_owner'], f"Person {ev['old_owner']}",
                                'paper_exchange', f"Paper passed to Person {ev['new_owner']}")
                        except Exception as _e:
                            pass
            except Exception:
                pass

        # ── 6. BOOK DETECTION ─────────────────────────────────────────────────
        book_det = state.get('book_detector')
        if book_det is not None:
            book_detections = book_det.detect(frame, confidence_threshold=0.30)
            if book_detections:
                stats['book_count'] = len(book_detections)
                frame = book_det.draw_detections(frame, book_detections)

                for det in book_detections:
                    dx, dy = det['center']
                    tid    = (dx // 20) * 1000 + (dy // 20)

                    if book_det.should_alert(tid, cooldown_sec=5.0):
                        stats['total_book_detections'] = stats.get('total_book_detections', 0) + 1
                        state['activity_log'].append(
                            f"📚 BOOK DETECTED (conf:{det['confidence']:.0%}) "
                            f"at {time.strftime('%H:%M:%S')}")

                        min_d, closest_sid = float('inf'), None
                        for fi in recognized_faces:
                            fx1, fy1, fx2, fy2 = fi['box']
                            d = np.sqrt((dx-(fx1+fx2)/2)**2 + (dy-(fy1+fy2)/2)**2)
                            if d < min_d:
                                min_d       = d
                                closest_sid = fi['student_id']

                        try:
                            if closest_sid and session_id:
                                sname = state['student_database_facenet'].get(
                                    closest_sid, {}).get('name', str(closest_sid))
                                db.log_cheating_event(
                                    session_id, closest_sid, sname,
                                    'book', severity='HIGH', device_class='book',
                                    confidence=det['confidence'],
                                    frame_number=state['frame_count'],
                                    timestamp_sec=current_time_sec)
                                db.log_alert(session_id, closest_sid, sname,
                                    'book', f"Book detected ({det['confidence']:.0%})")
                                db.mark_attendance(session_id, closest_sid, sname)
                        except Exception as _e:
                            pass

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 46), (fw, 88), (120, 60, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                cv2.putText(frame, f"!! {len(book_detections)} BOOK/NOTES DETECTED !!",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 180, 0), 2)
            else:
                stats['book_count'] = 0

        # ── 4c. MOBILE PHONE DETECTION (frame-level, loop se bahar) ───────────────
        mob_det  = state.get('mobile_detector')
        mob_dets_frame = mob_det.detect(frame) if mob_det else []
        if mob_dets_frame and mob_det:
            frame = mob_det.draw_detections(frame, mob_dets_frame)
            stats['mobile_count'] = len(mob_dets_frame)
            stats['total_mobile_detections'] = stats.get('total_mobile_detections', 0) + len(mob_dets_frame)
            state['activity_log'].append(
                f"📱 MOBILE DETECTED ({len(mob_dets_frame)}) at {time.strftime('%H:%M:%S')}")

        # ── 7. RISK SCORING ENGINE ─────────────────────────────────────────────
        risk_engine = state.get('risk_engine')
        if risk_engine and recognized_faces:
            for fi in recognized_faces:
                sid = fi['student_id']
                if fi['is_unknown']:
                    continue
                # IS KE BAAD YE NAYA BLOCK ADD KARO:
                # ── Classroom check ───────────────────────────────
                classroom_id = state.get('current_classroom_id')
                if classroom_id:
                    in_class = db.check_student_in_classroom(sid, classroom_id)
                    if not in_class:
                        x1, y1, x2, y2 = fi['box']
                        sname_wr = state['student_database_facenet'].get(
                            sid, {}).get('name', str(sid))
                        assigned = db.get_student_classroom(sid)
                        assigned_name = assigned.get('name', 'unassigned')
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 4)
                        cv2.putText(frame, f"WRONG ROOM: {sname_wr}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 165, 255), 2)
                        cv2.putText(frame, f"Assigned: {assigned_name}",
                                    (x1, y1 - 32), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (0, 165, 255), 2)
                        state['activity_log'].append(
                            f"⚠️ WRONG ROOM: {sname_wr} — "
                            f"should be in '{assigned_name}' "
                            f"at {time.strftime('%H:%M:%S')}")
                        try:
                            db.log_alert(session_id, sid, sname_wr,
                                         'wrong_room',
                                         f"Should be in: {assigned_name}")
                        except Exception:
                            pass
                        continue  # risk scoring skip — ye student yahan ka nahi
                # ── End classroom check ───────────────────────────
                # NAYA — actual frame pe detect karo:
                # Mobile — frame-level result use karo (detect() dobara mat chalao)
                mob_detected = len(mob_dets_frame) > 0
                mob_conf     = mob_dets_frame[0]['confidence'] if mob_dets_frame else 0.0
                if mob_det:
                    mob_det.update_history(sid, mob_detected)

                paper_exchange_fired = stats.get('paper_exchange_count', 0) > 0
                book_detected        = stats.get('book_count', 0) > 0
                book_conf            = 0.50

                peeking_active    = False
                peeking_direction = ''
                peeking_ratio     = 0.0
                if sid in peeking_history:
                    hist = list(peeking_history[sid])
                    if len(hist) >= 30:
                        lr = sum(1 for h in hist if h == 'LEFT')  / len(hist)
                        rr = sum(1 for h in hist if h == 'RIGHT') / len(hist)
                        if lr >= PEEK_RATIO_THRESHOLD or rr >= PEEK_RATIO_THRESHOLD:
                            peeking_active    = True
                            peeking_ratio     = max(lr, rr)
                            peeking_direction = 'LEFT' if lr > rr else 'RIGHT'

                yaw_dir    = student_yaw_dir.get(sid, 'Center')
                yaw_frames = student_yaw_frames.get(sid, 0)

                signals = {
                    'mobile_detected':        mob_detected,
                    'mobile_confidence':      mob_conf,
                    'paper_exchange':         paper_exchange_fired,
                    'book_detected':          book_detected,
                    'book_confidence':        book_conf,
                    'peeking':                peeking_active,
                    'peeking_direction':      peeking_direction,
                    'peeking_ratio':          peeking_ratio,
                    'yaw_direction':          yaw_dir,
                    'yaw_consecutive_frames': yaw_frames,
                }

                try:
                    result = risk_engine.update(sid, signals)
                    state['risk_scores'][sid] = result
                    # ── WhatsApp Notification ─────────────────────────
                    sname_for_notify = state['student_database_facenet'].get(
                        sid, {}).get('name', str(sid))
 
                    # Frame snapshot crop karo (student ka face area)
                    try:
                        x1n, y1n, x2n, y2n = fi['box']
                        pad = 60
                        fh_n, fw_n = frame.shape[:2]
                        snap_x1 = max(0, x1n - pad)
                        snap_y1 = max(0, y1n - pad)
                        snap_x2 = min(fw_n, x2n + pad)
                        snap_y2 = min(fh_n, y2n + pad)
                        student_snapshot = frame[snap_y1:snap_y2, snap_x1:snap_x2].copy()
                        if student_snapshot.size == 0:
                            student_snapshot = frame.copy()
                    except Exception:
                        student_snapshot = frame.copy()
 
                    check_and_notify(
                        student_id   = sid,
                        student_name = sname_for_notify,
                        risk_result  = result,
                        frame_bgr    = student_snapshot,
                    )
                    # ── End WhatsApp Notification ─────────────────────
                    x1, y1, x2, y2 = fi['box']
                    lvl   = result['risk_level']
                    score = result['risk_score']
                    color_map = {'GREEN':(0,200,0), 'YELLOW':(0,165,255), 'RED':(0,0,255)}
                    rc = color_map.get(lvl, (200,200,200))
                    cv2.putText(frame, f"RISK:{score:.0f} {lvl}",
                                (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc, 2)
                except Exception as _re:
                    pass

        # ── 8. STATS OVERLAY ──────────────────────────────────────────────────
        cv2.putText(frame,
                    f"People:{stats['person_count']}  Reg:{stats['recognized_count']}  "
                    f"Hands:{stats['hands_detected']}",
                    (10, 30 if stats['mobile_count'] == 0 else 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        if stats['yaw_alert_count'] > 0:
            cv2.putText(frame, f"YAW CHEAT: {stats['yaw_alert_count']}",
                        (10, fh-60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 3)
        if stats['peeking_count'] > 0:
            cv2.putText(frame, f"PEEKING: {stats['peeking_count']}",
                        (10, fh-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        if stats['paper_exchange_count'] > 0:
            cv2.putText(frame, f"PAPER EXCHANGE: {stats['paper_exchange_count']}",
                        (10, fh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        state['latest_frame'] = buffer.tobytes()

    cap.release()
    state['monitoring_active'] = False

    try:
        if session_id:
            final_stats = dict(state['latest_stats'])
            final_stats['total_frames'] = state['frame_count']
            db.end_session(session_id, final_stats)
            for sid, s_data in state['student_database_facenet'].items():
                sname = s_data.get('name', str(sid))
                if sid in peeking_history:
                    hist    = list(peeking_history[sid])
                    total_f = len(hist)
                    away_f  = sum(1 for h in hist if h in ['LEFT', 'RIGHT'])
                    if total_f > 0:
                        eng = max(0, 100 - (away_f / total_f) * 100)
                        try:
                            db.save_engagement(session_id, sid, sname, eng, away_f, total_f)
                            db.update_student_alert_count(sid, 0)
                        except Exception as e:
                            print(f"⚠️ Engagement save error: {e}")
            print(f"📊 DB Session #{session_id} closed")
    except Exception as e:
        print(f"⚠️  Error closing DB session: {e}")

    state['current_session_id'] = None
    state['start_time']         = None
    state['student_data']       = {}


# ============================================
# FLASK PAGE ROUTES
# ============================================

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/students')
def students_page():
    return render_template('students.html')


# ============================================
# API ROUTES
# ============================================

@app.route('/api/status')
def api_status():
    m = get_models()
    return jsonify({
        'facenet':            m.get('facenet') is not None,
        'face_det':           m.get('face_det') is not None,
        'paper_model':        m.get('paper_loaded', False),
        'device_model': state.get('device_detector').is_available(),
        'gpu':                torch.cuda.is_available(),
        'monitoring':         state['monitoring_active'],
        'students_registered':len(state['student_database_facenet']),
    })

@app.route('/api/students')
def get_students():
    try:
        students = db.get_all_students_summary()
        return jsonify({'students': students, 'source': 'mysql'})
    except Exception as e:
        return jsonify({'students': [
            {'id': sid, 'name': data['name'], 'samples': len(data['embeddings'])}
            for sid, data in state['student_database_facenet'].items()
        ], 'source': 'memory'})

@app.route('/api/students/<int:sid>/detail')
def student_detail(sid):
    try:
        detail = db.get_student_full_detail(sid)
        if not detail:
            return jsonify({'error': 'Student not found'}), 404
        return jsonify(detail)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
def get_sessions():
    try:
        sessions = db.get_sessions_list(limit=20)
        return jsonify({'sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/attendance')
def session_attendance(session_id):
    try:
        return jsonify({'attendance': db.get_session_attendance(session_id)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/events')
def session_events(session_id):
    try:
        event_type = request.args.get('type')
        events = db.get_cheating_events(session_id=session_id,
                                         event_type=event_type, limit=200)
        return jsonify({'events': events})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard_summary')
def dashboard_summary():
    try:
        return jsonify(db.get_dashboard_summary())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:sid>', methods=['DELETE'])
def delete_student(sid):
    if sid in state['student_database_facenet']:
        del state['student_database_facenet'][sid]
    try:
        db.delete_student(sid)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_student():
    data = request.json
    name        = data.get('name', '').strip()
    roll_number = data.get('roll_number', '').strip()
    department  = data.get('department', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name required'})
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({'success': False, 'error': 'Camera not accessible'})
    captured = 0
    max_cap  = 10
    sid      = state['next_id']
    for _ in range(50):
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        success, msg = register_face(frame, sid, name, roll_number, department)
        if success:
            captured += 1
        if captured >= max_cap:
            break
        time.sleep(0.2)
    cap.release()
    if captured > 0:
        state['next_id'] += 1
        state['activity_log'].append(f"✅ {name} registered ({captured} samples)")
        return jsonify({'success': True, 'captured': captured, 'student_id': sid})
    return jsonify({'success': False, 'error': 'Could not capture face'})

@app.route('/api/register/frame')
def register_frame():
    with _reg_cap_lock:
        cap = get_register_cap()
        ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        try:
            faces = detect_faces(frame)
            for f in faces:
                x1, y1, x2, y2 = f['box']
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        except Exception:
            pass
        _, buf = cv2.imencode('.jpg', frame)
        return Response(buf.tobytes(), mimetype='image/jpeg')
    return '', 404

@app.route('/api/monitoring/start', methods=['POST'])
def start_monitoring():
    if state['monitoring_active']:
        return jsonify({'success': False, 'error': 'Already running'})
    data = request.json or {}
    state['current_classroom_id'] = data.get('classroom_id')
    state['current_exam_name'] = (data.get('exam_name') or '').strip() or None
    state['current_exam_type']    = data.get('exam_type', 'quiz')
    state['monitoring_active'] = True
    state['frame_count']       = 0
    threading.Thread(target=monitoring_thread, daemon=True).start()
    return jsonify({'success': True})

@app.route('/api/monitoring/stop', methods=['POST'])
def stop_monitoring():
    state['monitoring_active'] = False
    state['yaw_cheating_counter'].clear()
    state['yaw_alert_active'].clear()
    state['eye_head_trackers'] = {}   # ← ye add karo
    peeking_history.clear()
    state['risk_scores'] = {}
    if state.get('risk_engine'):
        state['risk_engine'].reset_all()
    
    if state.get('paper_detector'):
        state['paper_detector'] = PaperExchangeDetector(time_threshold=0.5)
    if state.get('book_detector'):
        state['book_detector'].alert_cooldown.clear()
    if state.get('device_detector'):
        state['device_detector'].reset()
    # ── WhatsApp cooldown reset (naya session = fresh alerts) ──
    from whatsapp_notifier import _last_alert_time, _last_event_time
    _last_alert_time.clear()
    _last_event_time.clear()
 
    return jsonify({'success': True})

def generate_frames():
    while state['monitoring_active']:
        if state['latest_frame']:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   state['latest_frame'] + b'\r\n')
        time.sleep(0.033)

@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    logs        = list(state['activity_log'])[-15:]
    student_info = {}
    for sid, s_data in state['student_database_facenet'].items():
        mob_det       = state.get('mobile_detector')
        is_persistent = mob_det.is_persistent(sid) if mob_det else False
        risk_data     = state['risk_scores'].get(sid, {})
        student_info[sid] = {
            'name':         s_data['name'],
            'peeking':      sid in peeking_history and len(peeking_history[sid]) > 30,
            'yaw_alert':    state['yaw_alert_active'].get(sid, False),
            'device_alert': is_persistent,
            'risk_score':   risk_data.get('risk_score', 0),
            'risk_level':   risk_data.get('risk_level', 'GREEN'),
            'risk_reasons': risk_data.get('reasons', []),
        }
    mob_stats   = state['mobile_detector'].get_stats() if state.get('mobile_detector') else {}
    paper_total = state['paper_detector'].total_exchanges if state.get('paper_detector') else 0

    return jsonify({
        'stats':        state['latest_stats'],
        'logs':         logs,
        'students':     student_info,
        'monitoring':   state['monitoring_active'],
        'paper_total':  paper_total,
        'book_total':   state['book_detector'].total_detections if state.get('book_detector') else 0,
        'risk_scores':  state.get('risk_scores', {}),
        'current_classroom_id': state.get('current_classroom_id'),  # ✅
        'risk_summary': state['risk_engine'].snapshot() if state.get('risk_engine') else [],
    })

@app.route('/api/risk_scores')
def get_risk_scores():
    engine = state.get('risk_engine')
    if not engine:
        return jsonify({'risk_scores': [], 'error': 'Risk engine not initialised'})
    return jsonify({
        'risk_scores': engine.snapshot(),
        'detail':      state.get('risk_scores', {}),
    })

@app.route('/api/mobile_stats')
def mobile_stats():
    mob_det = state.get('mobile_detector')
    if mob_det is None:
        return jsonify({'available': False, 'error': 'best1.pt model not loaded'})
    return jsonify({
        'available':        True,
        'stats':            mob_det.get_stats(),
        'students_flagged': [sid for sid in mob_det.detection_history
                             if mob_det.is_persistent(sid)]
    })

@app.route('/api/activity_log')
def activity_log():
    return jsonify({'logs': list(state['activity_log'])[-20:]})

@app.route('/api/export_report')
def export_report():
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"/tmp/cheating_report_{ts}.csv"
    try:
        session_id = state.get('current_session_id')
        if not session_id:
            sessions   = db.get_sessions_list(limit=1)
            session_id = sessions[0]['id'] if sessions else None
        if session_id:
            db.export_session_report_csv(session_id, path)
        else:
            with open(path, 'w', newline='') as f:
                import csv as _csv
                w = _csv.writer(f)
                w.writerow(['No data available'])
        return send_file(path, as_attachment=True,
                         download_name=f"classroom_report_{ts}.csv")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_report/<int:session_id>')
def export_session_report(session_id):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"/tmp/session_{session_id}_report_{ts}.csv"
    try:
        db.export_session_report_csv(session_id, path)
        return send_file(path, as_attachment=True,
                         download_name=f"session_{session_id}_report.csv")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# CLASSROOM ROUTES
# ============================================

@app.route('/api/classrooms')
def get_classrooms():
    try:
        return jsonify({'classrooms': db.get_all_classrooms()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classrooms', methods=['POST'])
def create_classroom():
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Name required'})
    cid = db.create_classroom(name, data.get('exam_date'), data.get('notes', ''))
    return jsonify({'success': True, 'id': cid})

@app.route('/api/classrooms/<int:cid>', methods=['DELETE'])
def delete_classroom_route(cid):
    db.delete_classroom(cid)
    return jsonify({'success': True})

@app.route('/api/classrooms/<int:cid>/students')
def classroom_students(cid):
    return jsonify({'students': db.get_classroom_students(cid)})

@app.route('/api/classrooms/<int:cid>/assign', methods=['POST'])
def assign_student(cid):
    sid = (request.json or {}).get('student_id')
    if not sid:
        return jsonify({'success': False, 'error': 'student_id required'})
    db.assign_student_to_classroom(int(sid), cid)
    return jsonify({'success': True})

@app.route('/api/classrooms/<int:cid>/remove', methods=['POST'])
def remove_student_from_class(cid):
    sid = (request.json or {}).get('student_id')
    if not sid:
        return jsonify({'success': False, 'error': 'student_id required'})
    db.remove_student_from_classroom(int(sid), cid)
    return jsonify({'success': True})

@app.route('/classrooms')
def classrooms_page():
    return render_template('classrooms.html')
# ============================================
# ENTRY POINT
# ============================================

if __name__ == '__main__':
    print("🎓 Smart Classroom Flask API starting...")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)