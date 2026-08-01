"""
╔══════════════════════════════════════════════════════════════════╗
║   SmartClass — Head Movement & Eye Tracking Module               ║
║   Add this to your main app.py                                   ║
╚══════════════════════════════════════════════════════════════════╝

WHAT THIS ADDS:
  ✅ Full Head Pose (Pitch / Yaw / Roll in degrees) via solvePnP
  ✅ Eye Gaze Direction (LEFT / RIGHT / UP / DOWN / CENTER) via iris
  ✅ Blink Detection via Eye Aspect Ratio (EAR)
  ✅ Eye Contact Score — is student looking forward?
  ✅ Drowsiness Alert — eyes half-closed too long
  ✅ Per-student eye/head stats tracked over time
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import math

# ─────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDICES (Face Mesh)
# ─────────────────────────────────────────────

# Iris landmarks (only available when refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corners & outline for EAR
LEFT_EYE   = [362, 385, 387, 263, 373, 380]   # top-bottom pairs: 385-380, 387-373, corner 362-263
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]   # top-bottom pairs: 160-144, 158-153, corner 33-133

# Nose tip, chin, eye corners, mouth corners for solvePnP
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip       (landmark 1)
    [0.0,   -330.0, -65.0  ],   # Chin            (152)
    [-225.0, 170.0, -135.0 ],   # Left eye corner (33)
    [225.0,  170.0, -135.0 ],   # Right eye corner(263)
    [-150.0,-150.0, -125.0 ],   # Left mouth      (61)
    [150.0, -150.0, -125.0 ],   # Right mouth     (291)
], dtype=np.float64)

FACE_2D_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


# ─────────────────────────────────────────────
# LOAD MEDIAPIPE WITH IRIS (refine_landmarks=True)
# ─────────────────────────────────────────────

def load_mediapipe_with_iris():
    """
    IMPORTANT: Use this loader instead of the original.
    refine_landmarks=True enables iris landmarks 468–477.
    """
    mp_fm = mp.solutions.face_mesh
    mp_h  = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    face_mesh_iris = mp_fm.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,          # ← KEY: enables iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands_detector = mp_h.Hands(
        max_num_hands=20,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return face_mesh_iris, hands_detector, mp_draw, mp_draw_styles, mp_h, mp_fm


# ─────────────────────────────────────────────
# EYE ASPECT RATIO (blink / drowsiness)
# ─────────────────────────────────────────────

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    < 0.20 → blink / closed
    < 0.25 → drowsy
    """
    pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in eye_indices]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.3
    return (A + B) / (2.0 * C)


# ─────────────────────────────────────────────
# IRIS GAZE DIRECTION
# ─────────────────────────────────────────────

def get_iris_gaze(landmarks, img_w, img_h):
    """
    Returns gaze direction string + normalized offsets.
    Uses iris center relative to eye corner bounding box.
    """
    def iris_center(iris_ids):
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in iris_ids]
        return np.mean(pts, axis=0)

    def eye_bbox(eye_ids):
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_ids]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)

    # Left eye
    l_iris = iris_center(LEFT_IRIS)
    lx_min, lx_max, ly_min, ly_max = eye_bbox(LEFT_EYE)
    l_w = lx_max - lx_min or 1
    l_h = ly_max - ly_min or 1
    l_gaze_x = (l_iris[0] - lx_min) / l_w - 0.5   # -0.5..+0.5 (neg=left, pos=right)
    l_gaze_y = (l_iris[1] - ly_min) / l_h - 0.5   # neg=up, pos=down

    # Right eye
    r_iris = iris_center(RIGHT_IRIS)
    rx_min, rx_max, ry_min, ry_max = eye_bbox(RIGHT_EYE)
    r_w = rx_max - rx_min or 1
    r_h = ry_max - ry_min or 1
    r_gaze_x = (r_iris[0] - rx_min) / r_w - 0.5
    r_gaze_y = (r_iris[1] - ry_min) / r_h - 0.5

    # Average both eyes
    gaze_x = (l_gaze_x + r_gaze_x) / 2
    gaze_y = (l_gaze_y + r_gaze_y) / 2

    # Classify
    H_THRESH = 0.08
    V_THRESH = 0.08

    if   gaze_x < -H_THRESH:                  direction = "GAZE_LEFT"
    elif gaze_x >  H_THRESH:                  direction = "GAZE_RIGHT"
    elif gaze_y < -V_THRESH:                  direction = "GAZE_UP"
    elif gaze_y >  V_THRESH:                  direction = "GAZE_DOWN"
    else:                                      direction = "GAZE_CENTER"

    return {
        'direction': direction,
        'gaze_x': gaze_x,
        'gaze_y': gaze_y,
        'left_iris':  (int(l_iris[0]), int(l_iris[1])),
        'right_iris': (int(r_iris[0]), int(r_iris[1]))
    }


# ─────────────────────────────────────────────
# FULL HEAD POSE (solvePnP)
# ─────────────────────────────────────────────

def get_head_pose_angles(landmarks, img_w, img_h):
    face_2d = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in FACE_2D_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = img_w * 0.8
    cam_matrix = np.array([
        [focal_length, 0,            img_w / 2],
        [0,            focal_length, img_h / 2],
        [0,            0,            1        ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_POINTS, face_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)

    pitch = math.degrees(math.atan2(rmat[2][1], rmat[2][2]))
    yaw   = math.degrees(math.atan2(-rmat[2][0],
                math.sqrt(rmat[2][1]**2 + rmat[2][2]**2)))
    roll  = math.degrees(math.atan2(rmat[1][0], rmat[0][0]))

    return pitch, yaw, roll


# ─────────────────────────────────────────────
# PER-STUDENT EYE+HEAD TRACKER
# ─────────────────────────────────────────────

class StudentEyeHeadTracker:
    """
    Maintains rolling stats per student:
      - blink rate
      - average gaze direction
      - head pose history
      - drowsiness score
      - eye contact score (frames gaze=CENTER and yaw≈0)
    """
    def __init__(self, student_id, name, window=90):
        self.id   = student_id
        self.name = name
        self.window = window

        self.ear_history      = deque(maxlen=window)
        self.gaze_history     = deque(maxlen=window)
        self.yaw_history      = deque(maxlen=window)
        self.pitch_history    = deque(maxlen=window)

        self.blink_count      = 0
        self.last_ear         = 0.3
        self._was_blinking    = False

        self.drowsy_alert     = False
        self.drowsy_start     = None
        self.last_alert_time  = 0

        # Cumulative stats
        self.total_frames     = 0
        self.eye_contact_frames = 0
        self.gaze_away_frames = 0

    def update(self, ear, gaze_info, pitch, yaw):
        self.total_frames += 1
        self.ear_history.append(ear)
        self.gaze_history.append(gaze_info['direction'])
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        # Blink detection (EAR drops below threshold then recovers)
        EAR_BLINK = 0.20
        if ear < EAR_BLINK and not self._was_blinking:
            self._was_blinking = True
        elif ear >= EAR_BLINK and self._was_blinking:
            self.blink_count += 1
            self._was_blinking = False
        self.last_ear = ear

        # Eye contact: gaze center AND yaw small
        if gaze_info['direction'] == 'GAZE_CENTER' and abs(yaw) < 15:
            self.eye_contact_frames += 1
        else:
            self.gaze_away_frames += 1

    def get_drowsiness(self):
        """Returns (is_drowsy, avg_ear, duration_sec)"""
        if len(self.ear_history) < 20:
            return False, 0.3, 0
        avg = np.mean(list(self.ear_history)[-20:])
        DROWSY_THRESH = 0.25
        if avg < DROWSY_THRESH:
            if self.drowsy_start is None:
                self.drowsy_start = time.time()
            duration = time.time() - self.drowsy_start
            return True, avg, duration
        else:
            self.drowsy_start = None
            return False, avg, 0

    def get_eye_contact_score(self):
        if self.total_frames == 0:
            return 100.0
        return min(100.0, (self.eye_contact_frames / self.total_frames) * 100)

    def get_dominant_gaze(self):
        if not self.gaze_history:
            return 'GAZE_CENTER'
        from collections import Counter
        return Counter(self.gaze_history).most_common(1)[0][0]

    def get_avg_yaw(self):
        if not self.yaw_history:
            return 0.0
        return np.mean(list(self.yaw_history))

    def get_avg_pitch(self):
        if not self.pitch_history:
            return 0.0
        return np.mean(list(self.pitch_history))


# ─────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────

def draw_iris_gaze(frame, gaze_info, face_bbox):
    """Draw iris circles + gaze arrow on frame."""
    x1, y1, x2, y2 = face_bbox

    # Draw iris dots
    cv2.circle(frame, gaze_info['left_iris'],  4, (0, 255, 255), -1)
    cv2.circle(frame, gaze_info['right_iris'], 4, (0, 255, 255), -1)
    cv2.circle(frame, gaze_info['left_iris'],  6, (255, 165, 0),  1)
    cv2.circle(frame, gaze_info['right_iris'], 6, (255, 165, 0),  1)

    # Gaze label
    direction = gaze_info['direction']
    color = (0, 255, 0) if direction == 'GAZE_CENTER' else (0, 140, 255)
    cv2.putText(frame, direction, (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_head_pose(frame, pitch, yaw, roll, face_bbox, landmarks, img_w, img_h):
    """Draw 3D pose axes on the nose + angle readouts."""
    x1, y1, x2, y2 = face_bbox
    nose_x = int(landmarks[1].x * img_w)
    nose_y = int(landmarks[1].y * img_h)

    # Color based on severity
    yaw_abs = abs(yaw)
    if yaw_abs < 10:
        color = (0, 255, 0)       # green = forward
    elif yaw_abs < 25:
        color = (0, 165, 255)     # orange = mild
    else:
        color = (0, 0, 255)       # red = cheating

    # Draw angle text near face
    cv2.putText(frame, f"P:{pitch:+.0f} Y:{yaw:+.0f} R:{roll:+.0f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw a small directional cross-hair at nose
    length = 40
    # Yaw → horizontal arrow
    yaw_rad = math.radians(yaw)
    end_x = int(nose_x + length * math.sin(yaw_rad))
    end_y = int(nose_y - length * math.sin(math.radians(pitch)))
    cv2.arrowedLine(frame, (nose_x, nose_y), (end_x, end_y), color, 2, tipLength=0.3)


def draw_blink_drowsy(frame, ear, is_drowsy, blink_count, face_bbox, eye_contact_score):
    x1, y1, x2, y2 = face_bbox

    ear_color = (0, 0, 255) if ear < 0.20 else (0, 165, 255) if ear < 0.25 else (0, 255, 0)
    cv2.putText(frame, f"EAR:{ear:.2f} Blinks:{blink_count}",
                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_color, 1)

    ec_color = (0, 255, 0) if eye_contact_score > 60 else (0, 165, 255) if eye_contact_score > 30 else (0, 0, 255)
    cv2.putText(frame, f"EyeContact:{eye_contact_score:.0f}%",
                (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ec_color, 1)

    if is_drowsy:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 180), 3)
        cv2.putText(frame, "😴 DROWSY!", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


# ─────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# (call this inside your frame loop, per face)
# ─────────────────────────────────────────────

def process_face_eye_head(
        frame, face_landmarks, face_bbox, student_id,
        tracker_dict,              # dict: student_id → StudentEyeHeadTracker
        img_w, img_h,
        draw_iris=True,
        draw_pose=True,
        draw_blink=True,
        yaw_cheating_threshold=14.0,
        pitch_down_threshold=15.0,  # head down (looking at paper/phone)
        drowsy_duration_alert=3.0   # seconds of drowsy before alert fires
    ):
    """
    Full pipeline for one face in one frame.

    Returns dict with:
      pitch, yaw, roll       — head angles
      gaze_info              — direction + iris coords
      ear                    — eye aspect ratio
      is_blinking            — bool
      is_drowsy              — bool
      drowsy_duration        — seconds
      eye_contact_score      — 0-100
      alerts                 — list of alert strings
    """
    lm = face_landmarks.landmark

    # ── 1. Head Pose ──────────────────────────
    pitch, yaw, roll = get_head_pose_angles(lm, img_w, img_h)

    # ── 2. Eye Gaze (iris) ────────────────────
    gaze_info = get_iris_gaze(lm, img_w, img_h)

    # ── 3. Blink / EAR ───────────────────────
    left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  img_w, img_h)
    right_ear = eye_aspect_ratio(lm, RIGHT_EYE, img_w, img_h)
    ear = (left_ear + right_ear) / 2.0

    # ── 4. Update tracker ────────────────────
    if student_id not in tracker_dict:
        tracker_dict[student_id] = StudentEyeHeadTracker(student_id, str(student_id))
    tracker = tracker_dict[student_id]
    tracker.update(ear, gaze_info, pitch, yaw)

    is_drowsy, avg_ear, drowsy_dur = tracker.get_drowsiness()
    eye_contact_score = tracker.get_eye_contact_score()

    # ── 5. Build alerts ───────────────────────
    alerts = []

    if abs(yaw) > yaw_cheating_threshold and gaze_info['direction'] in ('GAZE_LEFT', 'GAZE_RIGHT'):
        direction = "RIGHT" if yaw > 0 else "LEFT"
        alerts.append(f"🚨 HEAD TURN {direction}: {yaw:+.0f}°")

    if pitch > pitch_down_threshold:
        alerts.append(f"📵 HEAD DOWN (paper/phone?): {pitch:+.0f}°")

    if gaze_info['direction'] in ('GAZE_LEFT', 'GAZE_RIGHT'):
        alerts.append(f"👁️ EYES {gaze_info['direction'].replace('GAZE_','')}")

    if is_drowsy and drowsy_dur > drowsy_duration_alert:
        alerts.append(f"😴 DROWSY for {drowsy_dur:.1f}s (EAR={avg_ear:.2f})")

    if ear < 0.15:
        alerts.append("👁️ EYES CLOSED / BLINK")

    # ── 6. Draw overlays ─────────────────────
    if draw_pose:
        draw_head_pose(frame, pitch, yaw, roll, face_bbox, lm, img_w, img_h)

    if draw_iris:
        draw_iris_gaze(frame, gaze_info, face_bbox)

    if draw_blink:
        draw_blink_drowsy(frame, ear, is_drowsy, tracker.blink_count, face_bbox, eye_contact_score)

    # Draw alerts on frame
    x1, y1, x2, y2 = face_bbox
    for i, alert in enumerate(alerts):
        cv2.putText(frame, alert, (x1, y1 - 50 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)

    return {
        'pitch': pitch, 'yaw': yaw, 'roll': roll,
        'gaze_info': gaze_info,
        'ear': ear,
        'is_blinking': ear < 0.20,
        'is_drowsy': is_drowsy,
        'drowsy_duration': drowsy_dur,
        'eye_contact_score': eye_contact_score,
        'blink_count': tracker.blink_count,
        'alerts': alerts
    }


# ─────────────────────────────────────────────
# HOW TO INTEGRATE INTO app.py
# ─────────────────────────────────────────────
"""
STEP 1 — Replace load_mediapipe() call:
─────────────────────────────────────────
    from head_eye_tracker import load_mediapipe_with_iris
    face_mesh, hands_detector, mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh = load_mediapipe_with_iris()

STEP 2 — Add to session state init:
─────────────────────────────────────
    if 'eye_head_trackers' not in st.session_state:
        st.session_state.eye_head_trackers = {}

STEP 3 — Add sidebar toggles (inside the live monitoring section):
────────────────────────────────────────────────────────────────────
    st.subheader("👁️ Eye & Head Tracking")
    enable_eye_tracking   = st.checkbox("Iris Gaze Tracking",   value=True)
    enable_head_pose      = st.checkbox("Full Head Pose (P/Y/R)", value=True)
    enable_blink_drowsy   = st.checkbox("Blink & Drowsiness",   value=True)
    yaw_pose_threshold    = st.slider("Head Turn Alert (°)", 10, 45, 20, 5)
    pitch_down_threshold  = st.slider("Head Down Alert (°)", 15, 40, 25, 5)
    drowsy_alert_duration = st.slider("Drowsy Alert (sec)",   2, 10, 3,  1)

STEP 4 — Inside the face_results loop, after matched_id is found:
──────────────────────────────────────────────────────────────────
    from head_eye_tracker import process_face_eye_head

    if matched_id:
        face_data = process_face_eye_head(
            frame          = frame,
            face_landmarks = face_lm,
            face_bbox      = (fx1, fy1, fx2, fy2),
            student_id     = matched_id,
            tracker_dict   = st.session_state.eye_head_trackers,
            img_w          = fw,
            img_h          = fh,
            draw_iris      = enable_eye_tracking,
            draw_pose      = enable_head_pose,
            draw_blink     = enable_blink_drowsy,
            yaw_cheating_threshold = yaw_pose_threshold,
            pitch_down_threshold   = pitch_down_threshold,
            drowsy_duration_alert  = drowsy_alert_duration
        )

        for alert in face_data['alerts']:
            st.session_state.activity_log.append(
                f"{alert} — {st.session_state.student_database_facenet.get(matched_id, {}).get('name', matched_id)}"
                f" at {time.strftime('%H:%M:%S')}"
            )

STEP 5 — In the stats panel, show eye contact score:
──────────────────────────────────────────────────────
    tracker = st.session_state.eye_head_trackers.get(sid)
    if tracker:
        st.caption(f"👁️ Eye Contact: {tracker.get_eye_contact_score():.0f}% | "
                   f"Blinks: {tracker.blink_count} | "
                   f"AvgYaw: {tracker.get_avg_yaw():+.0f}° | "
                   f"Gaze: {tracker.get_dominant_gaze()}")

STEP 6 — For Video Analysis mode (process_video_with_all_detections):
───────────────────────────────────────────────────────────────────────
    # Add at top of function:
    from head_eye_tracker import process_face_eye_head, StudentEyeHeadTracker
    v_eye_head_trackers = {}

    # Then inside the face_results loop, same as Step 4 above
    # but use v_eye_head_trackers instead of st.session_state.eye_head_trackers
"""