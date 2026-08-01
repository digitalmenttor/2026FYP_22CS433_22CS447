"""
liveness_detector.py  (v3)
==========================
IntelliExam — Real-Time Liveness Detection Module

Model:
  MiniFASNetV2-SE  (quantized INT8, ~600 KB)
  Source: github.com/facenox/face-antispoof-onnx
  Binary classification: Real (1) vs Spoof (0)
  Input: 128×128 BGR face crop, pixel/255, NCHW float32
  Output: [spoof_logit, real_logit]  → softmax → real_prob

Signals (hybrid):
  1. ML model score   (weight 0.65)  — primary
  2. Eye blink        (weight 0.20)  — backup
  3. Head motion      (weight 0.15)  — backup

Returns: LIVE | SPOOF | UNCERTAIN | NO_FACE
         + confidence (0.0–1.0) + reason string

pip install:
  pip install onnxruntime opencv-python mediapipe numpy
"""

import cv2
import os
import math
import time
import threading
import numpy as np
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

# ─────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────

MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_FNAME = "best_model_quantized.onnx"
MODEL_URL   = (
    "https://raw.githubusercontent.com/facenox/"
    "face-antispoof-onnx/main/models/best_model_quantized.onnx"
)
MODEL_INPUT_SIZE = (128, 128)   # this model expects 128×128
CROP_SCALE       = 1.5          # expand bbox by this factor

# Rolling window
WINDOW_FRAMES        = 45
MIN_FRAMES_TO_DECIDE = 15

# Model threshold
MODEL_LIVE_MIN   = 0.55   # real_prob above this → live

# Blink
EAR_THRESHOLD    = 0.21
EAR_CONSEC       = 2
MIN_BLINKS       = 1

# Motion (pixels) — very low for exam context
HEAD_MOTION_MIN  = 0.4

# Weights
W_MODEL  = 0.65
W_BLINK  = 0.20
W_MOTION = 0.15

# Verdict thresholds
LIVE_THRESH  = 0.52
SPOOF_THRESH = 0.30


# ─────────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────

@dataclass
class LivenessResult:
    status:     str    # LIVE | SPOOF | UNCERTAIN | NO_FACE
    confidence: float
    reason:     str

    def is_live(self) -> bool:
        return self.status == "LIVE"

    def __repr__(self):
        return f"[{self.status}] conf={self.confidence:.2f} — {self.reason}"


# ─────────────────────────────────────────────────────────────────
#  MODEL DOWNLOAD + LOAD
# ─────────────────────────────────────────────────────────────────

def _ensure_model() -> str:
    """Download model if not present. Returns local path."""
    import urllib.request
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, MODEL_FNAME)
    if not os.path.exists(dest):
        print(f"[Liveness] Downloading anti-spoof model (~600 KB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, dest)
            print(f"[Liveness] ✅ Model saved → {dest}")
        except Exception as e:
            print(f"[Liveness] ❌ Download failed: {e}")
            print(f"[Liveness]    Manual download: {MODEL_URL}")
            print(f"[Liveness]    Save to: {dest}")
            return ""
    return dest


class _AntiSpoofModel:
    """Thin ONNX wrapper for MiniFASNetV2-SE."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.log_severity_level   = 3
        self._sess  = ort.InferenceSession(
            model_path,
            sess_options = opts,
            providers    = ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._iname = self._sess.get_inputs()[0].name
        inp_shape   = self._sess.get_inputs()[0].shape
        print(f"[Liveness] Model input shape: {inp_shape}")

    def predict(self, face_bgr: np.ndarray) -> float:
        """
        Returns real_probability (0.0 – 1.0).
        Higher = more likely a live person.
        """
        img = cv2.resize(face_bgr, MODEL_INPUT_SIZE).astype(np.float32) / 255.0
        # Normalize with ImageNet stats (standard for this model family)
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) \
              / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = np.transpose(img, (2, 0, 1))[np.newaxis]  # HWC→NCHW
        out = self._sess.run(None, {self._iname: inp})[0][0]  # shape (2,)
        # Softmax over 2 classes [spoof, real]
        e = np.exp(out - out.max())
        s = e / e.sum()
        return float(s[1])   # real_prob


# ─────────────────────────────────────────────────────────────────
#  FACE CROP HELPER
# ─────────────────────────────────────────────────────────────────

def _crop_face(frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
    """Expand bbox by CROP_SCALE and crop. Returns None on failure."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    fw = (x2 - x1) * CROP_SCALE
    fh = (y2 - y1) * CROP_SCALE
    nx1 = max(0, int(cx - fw / 2))
    ny1 = max(0, int(cy - fh / 2))
    nx2 = min(w, int(cx + fw / 2))
    ny2 = min(h, int(cy + fh / 2))
    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size > 0 else None


# ─────────────────────────────────────────────────────────────────
#  MAIN DETECTOR
# ─────────────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Hybrid liveness detector.

    Primary : MiniFASNetV2-SE ONNX (anti-spoof ML model)
    Backup  : EAR blink + nose motion via MediaPipe

    Usage:
        det = LivenessDetector()

        # with bbox from your existing face detector:
        result = det.process_frame(frame, bbox=(x1,y1,x2,y2), face_count=1)

        # without bbox (auto-detects internally):
        result = det.process_frame(frame)
    """

    def __init__(self, auto_download: bool = True):
        # ML model
        self._model: Optional[_AntiSpoofModel] = None
        path = _ensure_model() if auto_download else \
               os.path.join(MODEL_DIR, MODEL_FNAME)
        if os.path.exists(path):
            try:
                self._model = _AntiSpoofModel(path)
            except Exception as e:
                print(f"[Liveness] ⚠️  Model load error: {e} — signal-only mode")

        # MediaPipe Face Mesh (blink + motion)
        _mp = mp.solutions.face_mesh
        self._fm = _mp.FaceMesh(
            max_num_faces            = 1,
            refine_landmarks         = True,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5,
        )

        # MediaPipe Face Detection (when no bbox given)
        _mpd = mp.solutions.face_detection
        self._fd = _mpd.FaceDetection(
            model_selection          = 0,
            min_detection_confidence = 0.6,
        )

        # Rolling windows
        self._model_scores: Deque[float] = deque(maxlen=WINDOW_FRAMES)
        self._motion_vals:  Deque[float] = deque(maxlen=WINDOW_FRAMES)

        # State
        self._ear_below  = 0
        self._blink_cnt  = 0
        self._frame_cnt  = 0
        self._prev_nose: Optional[tuple] = None

    # ──────────────────────────────────────────────────────────────

    def reset(self):
        self._model_scores.clear()
        self._motion_vals.clear()
        self._ear_below = 0
        self._blink_cnt = 0
        self._frame_cnt = 0
        self._prev_nose = None

    # ──────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr:  np.ndarray,
        bbox:       Optional[tuple] = None,   # (x1,y1,x2,y2)
        face_count: int             = -1,
    ) -> LivenessResult:

        if frame_bgr is None or frame_bgr.size == 0:
            return LivenessResult("NO_FACE", 0.0, "Empty frame")

        self._frame_cnt += 1
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Face presence / multi-face check ─────────────────────
        if face_count == -1 or bbox is None:
            fd_res  = self._fd.process(rgb)
            n_faces = len(fd_res.detections) if fd_res.detections else 0
            if face_count == -1:
                face_count = n_faces
            if n_faces == 0:
                return LivenessResult("NO_FACE", 0.0, "No face detected")
            if max(face_count, n_faces) > 1:
                return LivenessResult(
                    "SPOOF", 0.90,
                    f"Multiple faces ({max(face_count, n_faces)}) — substitute attempt"
                )
            # Build bbox from detection
            if bbox is None and fd_res.detections:
                rel  = fd_res.detections[0].location_data.relative_bounding_box
                bbox = (
                    max(0, int(rel.xmin * w)),
                    max(0, int(rel.ymin * h)),
                    min(w, int((rel.xmin + rel.width)  * w)),
                    min(h, int((rel.ymin + rel.height) * h)),
                )
        else:
            if face_count > 1:
                return LivenessResult(
                    "SPOOF", 0.90,
                    f"Multiple faces ({face_count}) — substitute attempt"
                )

        # ── ML model inference ────────────────────────────────────
        if self._model and bbox:
            crop = _crop_face(frame_bgr, bbox)
            if crop is not None:
                try:
                    score = self._model.predict(crop)
                    self._model_scores.append(score)
                except Exception as e:
                    print(f"[Liveness] inference error: {e}")

        # ── MediaPipe: blink + motion ─────────────────────────────
        fm_res = self._fm.process(rgb)
        if fm_res.multi_face_landmarks:
            lm  = fm_res.multi_face_landmarks[0].landmark
            ear = self._avg_ear(lm, w, h)

            if ear < EAR_THRESHOLD:
                self._ear_below += 1
            else:
                if self._ear_below >= EAR_CONSEC:
                    self._blink_cnt += 1
                self._ear_below = 0

            nx, ny = lm[1].x * w, lm[1].y * h
            if self._prev_nose:
                mv = math.hypot(nx - self._prev_nose[0], ny - self._prev_nose[1])
                self._motion_vals.append(mv)
            self._prev_nose = (nx, ny)

        # ── Warmup guard ──────────────────────────────────────────
        if self._frame_cnt < MIN_FRAMES_TO_DECIDE:
            left = MIN_FRAMES_TO_DECIDE - self._frame_cnt
            return LivenessResult("UNCERTAIN", 0.5, f"Collecting... ({left} frames)")

        return self._verdict()

    # ──────────────────────────────────────────────────────────────

    def _verdict(self) -> LivenessResult:
        reasons = []

        # Model score
        if self._model_scores:
            avg_m   = float(np.mean(list(self._model_scores)))
            model_s = avg_m
            tag     = "LIVE" if avg_m >= MODEL_LIVE_MIN else "SPOOF"
            reasons.append(f"model:{tag}({avg_m:.2f})")
        else:
            model_s = 0.5
            reasons.append("model:unavailable")

        # Blink score
        blink_s = 1.0 if self._blink_cnt >= MIN_BLINKS else 0.0
        reasons.append(f"blink:{'yes' if blink_s else 'no'}({self._blink_cnt}x)")

        # Motion score
        if len(self._motion_vals) >= 5:
            avg_mv   = float(np.mean(list(self._motion_vals)))
            motion_s = 1.0 if avg_mv >= HEAD_MOTION_MIN else 0.0
            reasons.append(f"motion:{avg_mv:.2f}px")
        else:
            motion_s = 0.5
            reasons.append("motion:warmup")

        conf = round(
            min(1.0, max(0.0,
                W_MODEL  * model_s  +
                W_BLINK  * blink_s  +
                W_MOTION * motion_s
            )), 3
        )

        if conf >= LIVE_THRESH:
            status = "LIVE"
        elif conf <= SPOOF_THRESH:
            status = "SPOOF"
        else:
            status = "UNCERTAIN"

        return LivenessResult(status, conf, " | ".join(reasons))

    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _avg_ear(lm, w, h) -> float:
        def ear(idx):
            pts = np.array([[lm[i].x*w, lm[i].y*h] for i in idx])
            A   = np.linalg.norm(pts[1]-pts[5])
            B   = np.linalg.norm(pts[2]-pts[4])
            C   = np.linalg.norm(pts[0]-pts[3]) + 1e-6
            return (A+B)/(2*C)
        return (ear([362,385,387,263,373,380]) + ear([33,160,158,133,153,144])) / 2


# ─────────────────────────────────────────────────────────────────
#  STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────

def _draw_overlay(frame, result: LivenessResult, fps: float):
    h, w = frame.shape[:2]
    colors = {"LIVE":(0,220,0),"SPOOF":(0,0,220),"UNCERTAIN":(0,165,255),"NO_FACE":(128,128,128)}
    c = colors.get(result.status, (200,200,200))
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,100), (15,15,15), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"{result.status}  ({result.confidence:.2f})",
                (12,38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3)
    for i, part in enumerate(result.reason.split(" | ")[:3]):
        cv2.putText(frame, part, (12, 60+i*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w-90,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)


def run_standalone_demo():
    cap = cv2.VideoCapture(0)
    det = LivenessDetector(auto_download=True)
    prev_t = time.time()
    fps    = 0.0
    print("[Liveness] Q=quit  R=reset")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame  = cv2.flip(frame, 1)
        result = det.process_frame(frame)
        now    = time.time()
        fps    = 0.9*fps + 0.1/max(now-prev_t, 1e-6)
        prev_t = now
        _draw_overlay(frame, result, fps)
        cv2.imshow("IntelliExam Liveness v3", frame)
        if result.status == "SPOOF":
            print(f"[ALERT] {result}")
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('r'):
            det.reset()
            print("[Liveness] Reset")
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────
#  EXAM GUARD  (app.py integration)
# ─────────────────────────────────────────────────────────────────

class ExamLivenessGuard:
    """
    Per-student liveness guard for app.py monitoring_thread().

    HOW TO ADD TO app.py
    ────────────────────
    1) Top of file:
         from liveness_detector import ExamLivenessGuard

    2) In state dict:
         'liveness_guard': None,

    3) In get_models() after other inits:
         from whatsapp_notifier import check_and_notify as _cna
         state['liveness_guard'] = ExamLivenessGuard(
             alert_callback=lambda r, sid, frm: _cna(
                 student_id   = sid,
                 student_name = state['student_database_facenet'].get(sid, {}).get('name', str(sid)),
                 risk_result  = {'event_counts': {'liveness_fail': 1}, 'reasons': [r.reason]},
                 frame_bgr    = frm,
             )
         )

    4) Inside monitoring_thread() frame loop, AFTER face recognition block:
         guard = state.get('liveness_guard')
         if guard:
             for fi in recognized_faces:
                 sid_lv = fi['student_id']
                 if fi['is_unknown']:
                     continue
                 lv = guard.update(
                     frame_bgr  = frame,
                     student_id = sid_lv,
                     bbox       = fi['box'],
                     face_count = len(recognized_faces),
                 )
                 if lv and lv.status == "SPOOF":
                     x1, y1, x2, y2 = fi['box']
                     cv2.putText(frame, "⚠ SPOOF", (x1, y1-50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                     cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

    5) In /api/monitoring/stop:
         if state.get('liveness_guard'):
             state['liveness_guard'].reset_all()
    """

    def __init__(
        self,
        alert_callback=None,
        spoof_cooldown_sec: float = 30.0,
        auto_download:      bool  = True,
    ):
        # Pre-load model once; share across per-student detectors
        path = _ensure_model() if auto_download else os.path.join(MODEL_DIR, MODEL_FNAME)
        self._model_path    = path if os.path.exists(path) else ""
        self._per_student:  dict = {}
        self._last_alert:   dict = {}
        self._callback          = alert_callback
        self._cooldown          = spoof_cooldown_sec

    def update(
        self,
        frame_bgr:  np.ndarray,
        student_id,
        bbox:       Optional[tuple] = None,
        face_count: int             = 1,
    ) -> Optional[LivenessResult]:

        if student_id not in self._per_student:
            d = LivenessDetector(auto_download=False)
            # Reuse already-loaded model
            if self._model_path and d._model is None:
                try:
                    d._model = _AntiSpoofModel(self._model_path)
                except Exception:
                    pass
            self._per_student[student_id] = d

        result = self._per_student[student_id].process_frame(
            frame_bgr, bbox=bbox, face_count=face_count
        )

        if result.status == "UNCERTAIN":
            return None

        if result.status in ("SPOOF", "NO_FACE") and self._callback:
            now  = time.time()
            last = self._last_alert.get(student_id, 0)
            if now - last >= self._cooldown:
                self._last_alert[student_id] = now
                try:
                    self._callback(result, student_id, frame_bgr)
                except Exception as e:
                    print(f"[LivenessGuard] callback error: {e}")

        return result

    def reset_student(self, sid):
        if sid in self._per_student:
            self._per_student[sid].reset()

    def reset_all(self):
        for d in self._per_student.values():
            d.reset()
        self._per_student.clear()
        self._last_alert.clear()


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_standalone_demo()