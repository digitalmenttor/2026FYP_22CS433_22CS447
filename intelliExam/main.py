import streamlit as st
import cv2
import torch
import numpy as np
from collections import deque
import time
import pickle
from pathlib import Path
import os
from PIL import Image
from scipy.spatial.distance import cosine
import threading
import platform
import csv
from datetime import datetime
import pandas as pd
import tempfile

# ============================================
# PLATFORM-AWARE ALARM SYSTEM
# ============================================
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
    except Exception as e:
        print(f"Alarm error: {e}")

# ============================================
# PATCH ULTRALYTICS
# ============================================
import ultralytics.nn.tasks as tasks
_original_torch_safe_load = tasks.torch_safe_load

def patched_torch_safe_load(file, *args, **kwargs):
    try:
        return torch.load(file, map_location='cpu', weights_only=False), file
    except Exception as e:
        return _original_torch_safe_load(file, *args, **kwargs)

tasks.torch_safe_load = patched_torch_safe_load

from ultralytics import YOLO
import mediapipe as mp

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ============================================
# PAGE CONFIG & CUSTOM CSS
# ============================================
st.set_page_config(
    page_title="Smart Classroom - Advanced Cheating Detection",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎓 Smart Classroom — Advanced Cheating Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">FaceNet • YOLOv8 • MediaPipe • Eyes • Head • Peeking • Paper Exchange • Device Detection</p>', unsafe_allow_html=True)

# ============================================
# LOAD ALL MODELS
# ============================================

@st.cache_resource
def load_yolo_models():
    pose_model = YOLO("yolov8s-pose.pt")
    person_model = YOLO("yolov8n.pt")
    try:
        paper_model = YOLO("best.pt")
        paper_model_loaded = True
    except Exception:
        paper_model = None
        paper_model_loaded = False
    try:
        custom_model = YOLO("best1.pt")
        custom_model_loaded = True
    except Exception:
        custom_model = None
        custom_model_loaded = False
    return pose_model, person_model, paper_model, paper_model_loaded, custom_model, custom_model_loaded

pose_model, person_model, paper_model, paper_model_loaded, custom_model, custom_model_loaded = load_yolo_models()

@st.cache_resource
def load_facenet_model():
    try:
        from facenet_pytorch import InceptionResnetV1
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dev)
        return resnet, dev
    except Exception as e:
        st.error(f"FaceNet loading error: {e}")
        return None, None

resnet, device = load_facenet_model()

@st.cache_resource
def load_opencv_face_detector():
    try:
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                prototxt_path)
            urllib.request.urlretrieve(
                "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                model_path)
        return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    except Exception as e:
        st.error(f"Face detector loading error: {e}")
        return None

face_detector = load_opencv_face_detector()

# ── CHANGED: refine_landmarks=True for iris ──
@st.cache_resource
def load_mediapipe():
    mp_fm = mp.solutions.face_mesh
    mp_h  = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    fm = mp_fm.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,          # ← iris landmarks enabled
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    h = mp_h.Hands(
        max_num_hands=20,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return fm, h, mp_draw, mp_draw_styles, mp_h, mp_fm

face_mesh, hands_detector, mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh = load_mediapipe()

# ============================================
# DATABASE & SESSION STATE
# ============================================
FACENET_DB = "facenet_database.pkl"

def init_session_state():
    defaults = {
        'student_database_facenet': {},
        'next_id': 1,
        'eye_head_trackers': {},
        'student_data': {},
        'activity_log': deque(maxlen=100),
        'frame_count': 0,
        'start_time': None,
        'stop_registration': False,
        'seat_mapping': {},
        'uploaded_video_path': None,
        'unknown_student_counter': 1000,
        'unknown_students': {},
        'peeking_detector': None,
        'exchange_detector': None,
        'paper_detector': None,
        'mobile_watch_detector': None,
        'cheating_events': [],
        'video_processing': False,
        'output_video_path': None,
        'yaw_cheating_counter': {},
        'yaw_alert_active': {},
        'yaw_alert_time': {},
        'alarm_playing': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

if Path(FACENET_DB).exists():
    with open(FACENET_DB, 'rb') as f:
        st.session_state.student_database_facenet = pickle.load(f)
    if st.session_state.student_database_facenet:
        st.session_state.next_id = max(st.session_state.student_database_facenet.keys()) + 1

# ============================================
# DETECTION CLASSES
# ============================================

class PeekingDetector:
    def __init__(self, window_size=90, threshold=0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}

    def update(self, student_id, head_direction):
        if student_id not in self.history:
            self.history[student_id] = deque(maxlen=self.window_size)
        self.history[student_id].append({'direction': head_direction, 'timestamp': time.time()})

    def get_peeking_score(self, student_id):
        if student_id not in self.history or len(self.history[student_id]) < 30:
            return {'score': 0, 'direction': None, 'is_peeking': False}
        history = list(self.history[student_id])
        total = len(history)
        left_ratio  = sum(1 for h in history if h['direction'] == 'LEFT')  / total
        right_ratio = sum(1 for h in history if h['direction'] == 'RIGHT') / total
        if left_ratio >= self.threshold:
            return {'score': int(left_ratio*100), 'direction': 'LEFT', 'is_peeking': True,
                    'left_ratio': left_ratio, 'right_ratio': right_ratio}
        if right_ratio >= self.threshold:
            return {'score': int(right_ratio*100), 'direction': 'RIGHT', 'is_peeking': True,
                    'left_ratio': left_ratio, 'right_ratio': right_ratio}
        max_ratio = max(left_ratio, right_ratio)
        return {'score': int(max_ratio*100),
                'direction': 'LEFT' if left_ratio > right_ratio else 'RIGHT',
                'is_peeking': False, 'left_ratio': left_ratio, 'right_ratio': right_ratio}

    def clear_all(self):
        self.history.clear()


class PaperExchangeDetector:
    def __init__(self, time_threshold=0.5):
        self.time_threshold = time_threshold
        self.paper_owner = {}
        self.owner_change_time = {}
        self.event_logged = {}
        self.alert_time_tracker = {}
        self.total_exchanges = 0
        self.exchange_events = []

    def process_frame(self, paper_results, person_results, current_time_sec):
        exchanges_detected = []
        if paper_results.boxes.id is None or person_results.boxes.id is None:
            return exchanges_detected
        paper_boxes   = paper_results.boxes.xyxy.cpu().numpy()
        paper_ids     = paper_results.boxes.id.cpu().numpy()
        person_boxes  = person_results.boxes.xyxy.cpu().numpy()
        person_ids    = person_results.boxes.id.cpu().numpy()
        person_classes= person_results.boxes.cls.cpu().numpy()
        persons = [(person_boxes[i], int(person_ids[i]))
                   for i in range(len(person_classes)) if person_classes[i] == 0]
        for i, p_box in enumerate(paper_boxes):
            p_id = int(paper_ids[i])
            x1, y1, x2, y2 = map(int, p_box)
            px_center = (x1+x2)/2; py_center = (y1+y2)/2
            current_owner = None
            for person_box, person_id in persons:
                if (person_box[0]<px_center<person_box[2] and person_box[1]<py_center<person_box[3]):
                    current_owner = person_id; break
            if p_id not in self.paper_owner:
                self.paper_owner[p_id] = current_owner
                self.event_logged[p_id] = False
                self.alert_time_tracker[p_id] = None
            else:
                old_owner = self.paper_owner[p_id]
                if old_owner != current_owner and current_owner is not None:
                    if p_id not in self.owner_change_time or self.owner_change_time[p_id] is None:
                        self.owner_change_time[p_id] = time.time()
                    elif time.time()-self.owner_change_time[p_id] > self.time_threshold:
                        if not self.event_logged[p_id]:
                            self.total_exchanges += 1
                            self.event_logged[p_id] = True
                            self.alert_time_tracker[p_id] = current_time_sec
                            event = {'timestamp': current_time_sec, 'paper_id': p_id,
                                     'old_owner': old_owner, 'new_owner': current_owner,
                                     'position': (int(px_center), int(py_center))}
                            self.exchange_events.append(event)
                            exchanges_detected.append(event)
                            self.paper_owner[p_id] = current_owner
                            self.owner_change_time[p_id] = None
                else:
                    self.owner_change_time[p_id] = None
                    self.event_logged[p_id] = False
        return exchanges_detected

    def get_active_alerts(self, current_time_sec, alert_duration=2.0):
        return [p_id for p_id,t in self.alert_time_tracker.items()
                if t is not None and current_time_sec-t <= alert_duration]


class ObjectExchangeDetector:
    def __init__(self, proximity_threshold=120, time_window=3.0):
        self.proximity_threshold = proximity_threshold
        self.time_window = time_window
        self.hand_positions = {}
        self.exchange_events = []
        self.last_alert_time = {}
        self.zone_violations = {}

    def update_hand_position(self, student_id, hand_landmarks, fw, fh, hand_type='unknown'):
        if student_id not in self.hand_positions:
            self.hand_positions[student_id] = deque(maxlen=30)
        wrist = hand_landmarks.landmark[0]
        self.hand_positions[student_id].append({
            'x': int(wrist.x*fw), 'y': int(wrist.y*fh),
            'timestamp': time.time(), 'hand_type': hand_type})

    def check_zone_crossing(self, student_id, student_zone, hand_x, frame_width):
        third = frame_width/3
        hand_zone = 'left' if hand_x<third else ('center' if hand_x<third*2 else 'right')
        if hand_zone != student_zone:
            ct = time.time()
            if student_id not in self.zone_violations or (ct-self.zone_violations[student_id])>2.0:
                self.zone_violations[student_id] = ct
                return {'student_id': student_id, 'student_zone': student_zone,
                        'hand_zone': hand_zone, 'hand_position': hand_x, 'timestamp': ct}
        return None

    def detect_hand_proximity(self):
        proximity_events = []
        ct = time.time()
        active = [sid for sid,pos in self.hand_positions.items()
                  if pos and (ct-pos[-1]['timestamp'])<1.0]
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                s1, s2 = active[i], active[j]
                if not self.hand_positions[s1] or not self.hand_positions[s2]: continue
                p1 = self.hand_positions[s1][-1]; p2 = self.hand_positions[s2][-1]
                dist = np.sqrt((p1['x']-p2['x'])**2+(p1['y']-p2['y'])**2)
                if dist < self.proximity_threshold:
                    pair_key = tuple(sorted([s1,s2]))
                    if pair_key not in self.last_alert_time or (ct-self.last_alert_time[pair_key])>1.0:
                        proximity_events.append({'student1':s1,'student2':s2,'distance':dist,
                            'position':((p1['x']+p2['x'])//2,(p1['y']+p2['y'])//2),'timestamp':ct})
                        self.last_alert_time[pair_key] = ct
        return proximity_events

    def clear_old_data(self):
        ct = time.time()
        for sid in list(self.hand_positions.keys()):
            if self.hand_positions[sid]:
                self.hand_positions[sid] = deque(
                    [p for p in self.hand_positions[sid] if (ct-p['timestamp'])<self.time_window], maxlen=30)


class MobileWatchDetector:
    def __init__(self, model):
        self.model = model
        self.detection_history = {}

    def detect(self, frame, confidence_threshold=0.5):
        if self.model is None: return []
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                detections.append({'box':(int(x1),int(y1),int(x2),int(y2)),
                    'confidence':float(box.conf[0]),'class':self.model.names[int(box.cls[0])]})
        return detections

    def update_history(self, student_id, detected):
        if student_id not in self.detection_history:
            self.detection_history[student_id] = deque(maxlen=30)
        self.detection_history[student_id].append({'detected':detected,'timestamp':time.time()})

    def is_persistent_detection(self, student_id, threshold=0.6):
        if student_id not in self.detection_history: return False
        history = list(self.detection_history[student_id])
        if len(history) < 10: return False
        return (sum(1 for h in history if h['detected'])/len(history)) >= threshold

# ============================================
# STUDENT CLASS
# ============================================

class Student:
    def __init__(self, student_id, name="Unknown"):
        self.id = student_id; self.name = name
        self.positions = deque(maxlen=30)
        self.looking_away_frames = 0; self.total_frames = 0
        self.engagement_score = 100; self.last_seen = time.time()
        self.peeking_alerts = 0; self.last_peeking_alert = 0
        self.seat_zone = None; self.current_position = None
        self.exchange_alerts = 0; self.mobile_watch_alerts = 0
        self.last_mobile_alert = 0; self.head_movement_alerts = 0
        self.last_head_movement_alert = 0; self.yaw_cheating_alerts = 0
        self.eye_left_right_alerts = 0   # ← NEW

    def update_position(self, x, y):
        self.positions.append((x,y)); self.last_seen = time.time()
        self.total_frames += 1; self.current_position = (x,y)

    def is_moving(self):
        if len(self.positions)<10: return False
        recent = list(self.positions)[-10:]
        return sum(np.sqrt((recent[i][0]-recent[i-1][0])**2+(recent[i][1]-recent[i-1][1])**2)
                   for i in range(1,len(recent))) > 50

    def calculate_engagement(self):
        if self.total_frames<10: return 100
        self.engagement_score = max(0,100-(self.looking_away_frames/self.total_frames)*100)
        return self.engagement_score

# ============================================
# INIT DETECTORS
# ============================================

if st.session_state.peeking_detector is None:
    st.session_state.peeking_detector = PeekingDetector(window_size=90, threshold=0.5)
if st.session_state.paper_detector is None:
    st.session_state.paper_detector = PaperExchangeDetector(time_threshold=0.5)
if st.session_state.exchange_detector is None:
    st.session_state.exchange_detector = ObjectExchangeDetector(proximity_threshold=120, time_window=3.0)
if st.session_state.mobile_watch_detector is None and custom_model_loaded:
    st.session_state.mobile_watch_detector = MobileWatchDetector(custom_model)

# ============================================
# HELPER FUNCTIONS
# ============================================

def save_facenet_database():
    with open(FACENET_DB,'wb') as f:
        pickle.dump(st.session_state.student_database_facenet, f)


def detect_faces_opencv(frame):
    if face_detector is None: return []
    try:
        h,w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > 0.5:
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                x1,y1,x2,y2 = box.astype("int")
                x1,y1 = max(0,x1),max(0,y1); x2,y2 = min(w,x2),min(h,y2)
                if x2>x1 and y2>y1:
                    faces.append({'box':(x1,y1,x2,y2),'confidence':float(conf)})
        return faces
    except Exception: return []


def get_face_embedding(face_tensor):
    if resnet is None: return None
    try:
        with torch.no_grad():
            if face_tensor.dim()==3: face_tensor = face_tensor.unsqueeze(0)
            elif face_tensor.dim()==5:
                face_tensor = face_tensor.squeeze()
                if face_tensor.dim()==3: face_tensor = face_tensor.unsqueeze(0)
            emb = resnet(face_tensor.to(device))
            return emb.cpu().numpy().flatten()
    except Exception: return None


def register_student_facenet(frame, student_id, student_name):
    try:
        faces = detect_faces_opencv(frame)
        if not faces: return False,None,"No face detected"
        best = max(faces, key=lambda x: x['confidence'])
        x1,y1,x2,y2 = best['box']
        if best['confidence']<0.9: return False,None,f"Low confidence: {best['confidence']*100:.1f}%"
        crop = frame[y1:y2,x1:x2]
        if crop.size==0: return False,None,"Invalid face crop"
        face_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb,(160,160))
        t = torch.from_numpy(face_resized).float()
        t = (t-127.5)/128.0; t = t.permute(2,0,1)
        emb = get_face_embedding(t)
        if emb is not None:
            if student_id not in st.session_state.student_database_facenet:
                st.session_state.student_database_facenet[student_id] = {'name':student_name,'embeddings':[]}
            st.session_state.student_database_facenet[student_id]['embeddings'].append(emb)
            save_facenet_database()
            return True,(x1,y1,x2,y2),f"Captured (conf: {best['confidence']*100:.1f}%)"
        return False,None,"Embedding failed"
    except Exception as e: return False,None,f"Error: {str(e)}"


def recognize_student_facenet(frame, box):
    if not st.session_state.student_database_facenet or resnet is None: return None,0.0
    try:
        x1,y1,x2,y2 = box; crop = frame[y1:y2,x1:x2]
        if crop.size==0: return None,0.0
        face_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb,(160,160))
        t = torch.from_numpy(face_resized).float()
        t = (t-127.5)/128.0; t = t.permute(2,0,1)
        emb = get_face_embedding(t)
        if emb is not None:
            best_id,best_sim = None,0.0
            for sid,data in st.session_state.student_database_facenet.items():
                for stored in data['embeddings']:
                    sim = 1-cosine(emb,stored)
                    if sim>best_sim and sim>0.6: best_sim=sim; best_id=sid
            if best_id: return best_id,best_sim*100
    except Exception: pass
    return None,0.0


def determine_seat_zone(x, y, frame_width):
    third = frame_width/3
    if x<third: return 'left'
    elif x<third*2: return 'center'
    return 'right'


def get_neighbor_direction(student_zone, looking_direction):
    if looking_direction not in ['LEFT','RIGHT']: return None
    if student_zone=='left': return 'NEIGHBOR_RIGHT' if looking_direction=='RIGHT' else 'NO_NEIGHBOR'
    elif student_zone=='center': return 'NEIGHBOR_LEFT' if looking_direction=='LEFT' else 'NEIGHBOR_RIGHT'
    elif student_zone=='right': return 'NEIGHBOR_LEFT' if looking_direction=='LEFT' else 'NO_NEIGHBOR'
    return None


def assign_unknown_student_id(x, y, existing_unknowns, threshold=150):
    for uid,student in existing_unknowns.items():
        if student.current_position:
            sx,sy = student.current_position
            if np.sqrt((x-sx)**2+(y-sy)**2)<threshold and (time.time()-student.last_seen)<2.0:
                return uid
    new_id = st.session_state.unknown_student_counter
    st.session_state.unknown_student_counter += 1
    return new_id

# ============================================
# HEAD DIRECTION FUNCTIONS
# ============================================

def get_head_yaw(landmarks, image_width):
    eye_center_x = (landmarks[33].x+landmarks[263].x)/2
    return (landmarks[1].x-eye_center_x)*image_width


def detect_head_direction_detailed(face_landmarks, image_width=None):
    lm = face_landmarks.landmark
    if image_width:
        yaw = get_head_yaw(lm, image_width)
        if yaw < -20: return 'LEFT'
        if yaw >  20: return 'RIGHT'
    nose=lm[1]; left_eye=lm[33]; right_eye=lm[263]
    eye_cx=(left_eye.x+right_eye.x)/2; eye_cy=(left_eye.y+right_eye.y)/2
    h_dev=nose.x-eye_cx; v_dev=nose.y-eye_cy
    if abs(h_dev)>0.04: return 'LEFT' if h_dev<0 else 'RIGHT'
    if abs(v_dev)>0.03: return 'DOWN' if v_dev>0 else 'UP'
    return 'CENTER'


def detect_head_pose(face_landmarks):
    lm = face_landmarks.landmark
    eye_cx = (lm[33].x+lm[263].x)/2
    return "LOOKING_AWAY" if abs(lm[1].x-eye_cx)>0.1 else "FOCUSED"


# ── NEW: iris gaze detection ──────────────────────────────────────────
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]

def get_iris_gaze(landmarks, img_w, img_h):
    """Returns gaze direction: GAZE_LEFT / GAZE_RIGHT / GAZE_UP / GAZE_DOWN / GAZE_CENTER"""
    try:
        if len(landmarks) <= 477:
            return None   # iris not available

        def iris_center(ids):
            pts = [(landmarks[i].x*img_w, landmarks[i].y*img_h) for i in ids]
            return np.mean(pts, axis=0)

        def eye_bbox(ids):
            pts = [(landmarks[i].x*img_w, landmarks[i].y*img_h) for i in ids]
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            return min(xs),max(xs),min(ys),max(ys)

        l_iris = iris_center(LEFT_IRIS)
        lx_min,lx_max,ly_min,ly_max = eye_bbox(LEFT_EYE)
        lw = lx_max-lx_min or 1; lh = ly_max-ly_min or 1
        l_gx = (l_iris[0]-lx_min)/lw - 0.5
        l_gy = (l_iris[1]-ly_min)/lh - 0.5

        r_iris = iris_center(RIGHT_IRIS)
        rx_min,rx_max,ry_min,ry_max = eye_bbox(RIGHT_EYE)
        rw = rx_max-rx_min or 1; rh = ry_max-ry_min or 1
        r_gx = (r_iris[0]-rx_min)/rw - 0.5
        r_gy = (r_iris[1]-ry_min)/rh - 0.5

        gx = (l_gx+r_gx)/2; gy = (l_gy+r_gy)/2

        H_THRESH = 0.12; V_THRESH = 0.10
        if   gx < -H_THRESH: direction = 'GAZE_LEFT'
        elif gx >  H_THRESH: direction = 'GAZE_RIGHT'
        elif gy < -V_THRESH: direction = 'GAZE_UP'
        elif gy >  V_THRESH: direction = 'GAZE_DOWN'
        else:                direction = 'GAZE_CENTER'

        return {
            'direction': direction,
            'gaze_x': gx, 'gaze_y': gy,
            'left_iris':  (int(l_iris[0]), int(l_iris[1])),
            'right_iris': (int(r_iris[0]), int(r_iris[1]))
        }
    except Exception:
        return None


def get_eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    try:
        pts = [(int(landmarks[i].x*img_w), int(landmarks[i].y*img_h)) for i in eye_indices]
        A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
        return (A+B)/(2.0*C) if C>0 else 0.3
    except Exception:
        return 0.3


def detect_hand_gesture(hand_landmarks):
    tips=[hand_landmarks.landmark[i] for i in [8,12,16,20]]
    wrist=hand_landmarks.landmark[0]
    fingers_up=sum(1 for t in tips if t.y<wrist.y)
    if fingers_up>=3: return "RAISED_HAND"
    elif fingers_up==1 and tips[0].y<wrist.y: return "POINTING"
    elif fingers_up==0: return "FIST"
    return "OPEN_PALM"


def draw_skeleton(frame, kpts, kpts_conf):
    skeleton=[[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
              [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    for c in skeleton:
        try:
            i1,i2=c[0]-1,c[1]-1
            if (len(kpts)>max(i1,i2) and kpts_conf[i1]>0.5 and kpts_conf[i2]>0.5):
                pt1=(int(kpts[i1][0]),int(kpts[i1][1])); pt2=(int(kpts[i2][0]),int(kpts[i2][1]))
                if all(v>0 for v in pt1+pt2): cv2.line(frame,pt1,pt2,(0,255,255),2)
        except Exception: pass

# ============================================
# YAW CHEATING DETECTION CORE
# ============================================

def process_yaw_cheating(face_idx, yaw, frame, landmarks, frame_w,
                          yaw_threshold, cheating_frames_needed, alert_duration, enable_alarm):
    if face_idx not in st.session_state.yaw_cheating_counter:
        st.session_state.yaw_cheating_counter[face_idx] = 0
        st.session_state.yaw_alert_active[face_idx] = False
        st.session_state.yaw_alert_time[face_idx] = 0
        st.session_state.alarm_playing[face_idx] = False

    direction = ("Looking Left"  if yaw < -yaw_threshold else
                 "Looking Right" if yaw >  yaw_threshold else "Center")

    if direction != "Center":
        st.session_state.yaw_cheating_counter[face_idx] += 1
    else:
        st.session_state.yaw_cheating_counter[face_idx] = 0
        st.session_state.alarm_playing[face_idx] = False

    log_msg = None
    if (st.session_state.yaw_cheating_counter[face_idx] > cheating_frames_needed
            and not st.session_state.yaw_alert_active[face_idx]):
        st.session_state.yaw_alert_active[face_idx] = True
        st.session_state.yaw_alert_time[face_idx] = time.time()
        if enable_alarm and not st.session_state.alarm_playing[face_idx]:
            st.session_state.alarm_playing[face_idx] = True
            threading.Thread(target=play_alarm, daemon=True).start()
        log_msg = f"🚨 YAW ALERT: Face {face_idx} cheating ({direction}) at {time.strftime('%H:%M:%S')}"

    nose = landmarks[1]
    x = int(nose.x*frame_w); y = int(nose.y*frame.shape[0])
    dir_color = (0,255,0) if direction=="Center" else (0,0,255)
    cv2.putText(frame, f"Face {face_idx}: {direction}", (x-60,y-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 2)

    if st.session_state.yaw_alert_active[face_idx]:
        if time.time()-st.session_state.yaw_alert_time[face_idx] < alert_duration:
            overlay=frame.copy()
            cv2.rectangle(overlay,(x-140,y-80),(x+140,y-15),(0,0,180),-1)
            cv2.addWeighted(overlay,0.4,frame,0.6,0,frame)
            cv2.putText(frame,f"🚨 CHEATING! Face {face_idx}",(x-130,y-40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
        else:
            st.session_state.yaw_alert_active[face_idx] = False
            st.session_state.yaw_cheating_counter[face_idx] = 0
            st.session_state.alarm_playing[face_idx] = False

    return frame, log_msg, direction

# ============================================
# EXPORT REPORT
# ============================================

def export_cheating_report():
    if not st.session_state.cheating_events: return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"cheating_report_{ts}.csv"
    with open(csv_path,'w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['Timestamp','Event Type','Student ID','Student Name','Details','Severity'])
        for event in st.session_state.cheating_events:
            writer.writerow([event.get('timestamp',''),event.get('type',''),
                             event.get('student_id',''),event.get('student_name',''),
                             event.get('details',''),event.get('severity','')])
    return csv_path

# ============================================
# ── HELPER: face-to-mediapipe matching ──
# FIX: center-distance based matching instead of strict overlap
# ============================================

def match_mediapipe_to_face(mp_x1, mp_y1, mp_x2, mp_y2, recognized_faces):
    """
    Returns (matched_id, matched_bbox) by finding the recognised face whose
    center is closest to the MediaPipe bounding box center.
    Much more robust than pixel-overlap threshold.
    """
    mp_cx = (mp_x1+mp_x2)/2; mp_cy = (mp_y1+mp_y2)/2
    best_id = None; best_bbox = (mp_x1,mp_y1,mp_x2,mp_y2); best_dist = float('inf')

    for fi in recognized_faces:
        fx1,fy1,fx2,fy2 = fi['box']
        fcx=(fx1+fx2)/2; fcy=(fy1+fy2)/2
        dist = np.sqrt((mp_cx-fcx)**2+(mp_cy-fcy)**2)
        # also check any overlap exists at all
        ox = max(0, min(mp_x2,fx2)-max(mp_x1,fx1))
        oy = max(0, min(mp_y2,fy2)-max(mp_y1,fy1))
        if dist < best_dist and (ox>0 or dist < 150):
            best_dist = dist; best_id = fi['student_id']; best_bbox = fi['box']

    return best_id, best_bbox

# ============================================
# VIDEO PROCESSING FUNCTION  (FULLY UPDATED)
# ============================================

def process_video_with_all_detections(video_path, output_path, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video file!")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info(f"📹 {width}x{height} @ {fps:.1f} FPS | {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    v_paper_det   = PaperExchangeDetector(time_threshold=0.5)
    v_peek_det    = PeekingDetector(window_size=90, threshold=0.5)
    v_mobile_det  = MobileWatchDetector(custom_model) if custom_model_loaded else None
    v_student_data = {}

    # per-video counters
    frame_num             = 0
    total_paper_exchanges = 0
    total_head_alerts     = 0   # head LEFT or RIGHT
    total_peeking_alerts  = 0
    total_mobile_alerts   = 0
    total_gaze_left_right = 0   # eyes LEFT or RIGHT
    total_head_down       = 0
    total_drowsy          = 0

    # per-student gaze alert cooldown  {student_id: last_alert_time}
    gaze_alert_cooldown = {}
    head_alert_cooldown = {}

    YAW_DEG        = 18.0   # head turn threshold (degrees)
    PITCH_DOWN_DEG = 25.0
    DROWSY_EAR     = 0.23
    DROWSY_SECS    = 3.0
    drowsy_start   = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_num       += 1
        current_time_sec = frame_num / fps
        progress_bar.progress(min(frame_num/total_frames, 1.0))
        status_text.text(f"Processing: Frame {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%)")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_head_alerts  = 0
        frame_gaze_alerts  = 0
        frame_peek_alerts  = 0
        frame_paper        = 0
        frame_mobile       = 0
        frame_head_down    = 0
        frame_drowsy       = 0

        # ── 1. YOLO PERSON — draw body bounding boxes ─────────────────
        try:
            person_res_body = person_model(frame, verbose=False, conf=0.5)
            for r in person_res_body:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:   # class 0 = person
                        bx1,by1,bx2,by2 = map(int, box.xyxy[0].cpu().numpy())
                        conf_p = float(box.conf[0])
                        # cyan body box
                        cv2.rectangle(frame, (bx1,by1), (bx2,by2), (255,255,0), 2)
                        cv2.putText(frame, f"Person {conf_p:.2f}",
                                    (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        except Exception:
            pass

        # ── 2. FACE DETECTION + RECOGNITION ───────────────────────────
        faces = detect_faces_opencv(frame)
        recognized_faces = []

        for face in faces:
            x1,y1,x2,y2   = face['box']
            student_id,conf = recognize_student_facenet(frame, face['box'])
            cx,cy           = (x1+x2)/2,(y1+y2)/2

            if student_id:
                student_name = st.session_state.student_database_facenet[student_id]['name']
                is_unknown   = False
            else:
                student_id   = f"Unk_{int(cx)}"
                student_name = student_id; conf=0; is_unknown=True

            recognized_faces.append({'box':(x1,y1,x2,y2),'student_id':student_id,
                                      'student_name':student_name,'confidence':conf,
                                      'is_unknown':is_unknown})

            color = (128,128,128) if is_unknown else ((0,255,0) if conf>85 else (0,255,255))
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,student_name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            if not is_unknown:
                cv2.putText(frame,f"Conf:{conf:.0f}%",(x1,y1-28),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)

            if student_id not in v_student_data:
                v_student_data[student_id] = Student(student_id, student_name)
            v_student_data[student_id].update_position(cx,cy)

        # ── 3. MEDIAPIPE FACE MESH ─────────────────────────────────────
        face_results = face_mesh.process(rgb_frame)

        if face_results.multi_face_landmarks:
            for face_lm in face_results.multi_face_landmarks:
                lm = face_lm.landmark

                xc = [l.x*width  for l in lm]; yc = [l.y*height for l in lm]
                mp_x1,mp_y1 = int(min(xc)),int(min(yc))
                mp_x2,mp_y2 = int(max(xc)),int(max(yc))

                # ── FIX: use center-distance matching ─────────────────
                matched_id, matched_bbox = match_mediapipe_to_face(
                    mp_x1,mp_y1,mp_x2,mp_y2, recognized_faces)

                # If no registered faces at all, still process (assign temp id)
                if matched_id is None:
                    mp_cx=(mp_x1+mp_x2)//2; mp_cy=(mp_y1+mp_y2)//2
                    matched_id   = f"Unk_{mp_cx}"
                    matched_bbox = (mp_x1,mp_y1,mp_x2,mp_y2)
                    if matched_id not in v_student_data:
                        v_student_data[matched_id] = Student(matched_id, matched_id)

                bx1,by1,bx2,by2 = matched_bbox

                # ══ a) HEAD DIRECTION (LEFT / RIGHT alert) ═════════════
                head_dir = detect_head_direction_detailed(face_lm, width)

                dir_color = (0,255,0) if head_dir=='CENTER' else (0,0,255)
                cv2.putText(frame, f"Head:{head_dir}", (bx1,by2+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 2)

                if head_dir in ('LEFT','RIGHT'):
                    ct = current_time_sec
                    last = head_alert_cooldown.get(matched_id, -999)
                    if ct - last > 1.5:   # 1.5s cooldown per student
                        head_alert_cooldown[matched_id] = ct
                        frame_head_alerts += 1
                        total_head_alerts += 1
                        cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,0,255),3)
                        cv2.putText(frame, f"HEAD {head_dir}!",
                                    (bx1,by1-15), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
                        if matched_id in v_student_data:
                            v_student_data[matched_id].head_movement_alerts += 1

                # HEAD DOWN (pitch)
                try:
                    import math
                    FACE_3D = np.array([[0.,0.,0.],[0.,-330.,-65.],[-225.,170.,-135.],
                                        [225.,170.,-135.],[-150.,-150.,-125.],[150.,-150.,-125.]],dtype=np.float64)
                    IDS_2D  = [1,152,33,263,61,291]
                    face_2d = np.array([[lm[i].x*width,lm[i].y*height] for i in IDS_2D],dtype=np.float64)
                    fl      = width
                    cam     = np.array([[fl,0,width/2],[0,fl,height/2],[0,0,1]],dtype=np.float64)
                    ok,rvec,tvec = cv2.solvePnP(FACE_3D,face_2d,cam,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
                    if ok:
                        rmat,_ = cv2.Rodrigues(rvec)
                        angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)
                        pitch_deg = angles[0]*360
                        if pitch_deg > PITCH_DOWN_DEG:
                            frame_head_down += 1; total_head_down += 1
                            cv2.putText(frame,f"HEAD DOWN {pitch_deg:+.0f}°",
                                        (bx1,by2+45),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,80,255),2)
                except Exception:
                    pass

                # ══ b) IRIS GAZE (LEFT / RIGHT alert) ═════════════════
                gaze_info = get_iris_gaze(lm, width, height)
                if gaze_info:
                    # draw iris dots
                    cv2.circle(frame, gaze_info['left_iris'],  4, (0,255,255), -1)
                    cv2.circle(frame, gaze_info['right_iris'], 4, (0,255,255), -1)

                    gaze_dir = gaze_info['direction']
                    g_color  = (0,255,0) if gaze_dir=='GAZE_CENTER' else (0,140,255)
                    cv2.putText(frame, gaze_dir, (bx1,by2+68),
                                cv2.FONT_HERSHEY_SIMPLEX,0.5,g_color,2)

                    # ALERT only for LEFT / RIGHT gaze
                    if gaze_dir in ('GAZE_LEFT','GAZE_RIGHT'):
                        ct  = current_time_sec
                        key = f"gaze_{matched_id}"
                        last = gaze_alert_cooldown.get(key, -999)
                        if ct - last > 1.5:
                            gaze_alert_cooldown[key] = ct
                            frame_gaze_alerts  += 1
                            total_gaze_left_right += 1
                            side = gaze_dir.replace('GAZE_','')
                            cv2.putText(frame, f"EYES {side}!",
                                        (bx1,by1-35), cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2)
                            # draw coloured eye outline
                            for idx in LEFT_EYE+RIGHT_EYE:
                                ex=int(lm[idx].x*width); ey=int(lm[idx].y*height)
                                cv2.circle(frame,(ex,ey),2,(0,200,255),-1)
                            if matched_id in v_student_data:
                                v_student_data[matched_id].eye_left_right_alerts += 1

                # ══ c) BLINK / DROWSY ══════════════════════════════════
                try:
                    l_ear = get_eye_aspect_ratio(lm,LEFT_EYE, width,height)
                    r_ear = get_eye_aspect_ratio(lm,RIGHT_EYE,width,height)
                    ear   = (l_ear+r_ear)/2
                    ear_color = (0,0,255) if ear<0.20 else (0,165,255) if ear<DROWSY_EAR else (0,255,0)
                    cv2.putText(frame,f"EAR:{ear:.2f}",(bx1,by2+90),
                                cv2.FONT_HERSHEY_SIMPLEX,0.45,ear_color,1)
                    if ear < DROWSY_EAR:
                        if matched_id not in drowsy_start:
                            drowsy_start[matched_id] = current_time_sec
                        elif current_time_sec-drowsy_start[matched_id] > DROWSY_SECS:
                            frame_drowsy += 1; total_drowsy += 1
                            cv2.putText(frame,"DROWSY!",(bx1,by1-55),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,200),2)
                            cv2.rectangle(frame,(bx1,by1),(bx2,by2),(0,0,180),3)
                    else:
                        drowsy_start.pop(matched_id, None)
                except Exception:
                    pass

                # ══ d) PEEKING (sustained head turn) ══════════════════
                v_peek_det.update(matched_id, head_dir)
                peek_info = v_peek_det.get_peeking_score(matched_id)
                if peek_info['is_peeking']:
                    frame_peek_alerts  += 1
                    total_peeking_alerts += 1
                    cv2.putText(frame,f"PEEKING {peek_info['direction']}! ({peek_info['score']}%)",
                                (bx1,by2+112),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)

        # ── 4. PAPER EXCHANGE ──────────────────────────────────────────
        if paper_model_loaded:
            try:
                p_res  = person_model.track(frame, persist=True, verbose=False, conf=0.5)
                pa_res = paper_model.track(frame,  persist=True, verbose=False, conf=0.5)
                if p_res and pa_res:
                    for box in pa_res[0].boxes:
                        px1,py1,px2,py2=map(int,box.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame,(px1,py1),(px2,py2),(255,165,0),2)
                        cv2.putText(frame,"PAPER",(px1,py1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,165,0),2)
                    for ex in v_paper_det.process_frame(pa_res[0],p_res[0],current_time_sec):
                        frame_paper += 1; total_paper_exchanges += 1
                        px,py=ex['position']
                        cv2.circle(frame,(px,py),50,(0,0,255),4); cv2.circle(frame,(px,py),8,(0,0,255),-1)
                        cv2.putText(frame,"PAPER EXCHANGE!",(px-120,py-60),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
                        cv2.putText(frame,f"P{ex['old_owner']}->P{ex['new_owner']}",(px-110,py-25),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
            except Exception: pass

        # ── 5. MOBILE/WATCH ────────────────────────────────────────────
        if v_mobile_det:
            try:
                for det in v_mobile_det.detect(frame,0.5):
                    frame_mobile += 1; total_mobile_alerts += 1
                    dx1,dy1,dx2,dy2=det['box']
                    cv2.rectangle(frame,(dx1,dy1),(dx2,dy2),(0,0,255),3)
                    cv2.putText(frame,f"{det['class']} {det['confidence']:.2f}",
                                (dx1,dy1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    cv2.putText(frame,"PROHIBITED DEVICE!",(dx1,dy2+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            except Exception: pass

        # ── 6. TOP BANNER ──────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(width,130),(0,0,0),-1)
        cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
        mm=int(current_time_sec//60); ss=int(current_time_sec%60)
        cv2.putText(frame,f"Time:{mm}:{ss:02d}  Students:{len(v_student_data)}",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(frame,f"HeadTurn:{total_head_alerts}  EyesLR:{total_gaze_left_right}  Peek:{total_peeking_alerts}  Paper:{total_paper_exchanges}",
                    (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,0),2)
        cv2.putText(frame,f"Devices:{total_mobile_alerts}  HeadDown:{total_head_down}  Drowsy:{total_drowsy}",
                    (10,85),cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,255,200),2)

        # ── bottom alerts this frame ────────────────────────────────
        yb = height-10
        for label,cnt,color in [
            (f"DEVICE:{frame_mobile}",      frame_mobile,      (0,0,255)),
            (f"PAPER:{frame_paper}",         frame_paper,        (0,0,255)),
            (f"PEEKING:{frame_peek_alerts}", frame_peek_alerts,  (0,0,255)),
            (f"HEAD TURN:{frame_head_alerts}",frame_head_alerts, (0,165,255)),
            (f"EYES LR:{frame_gaze_alerts}", frame_gaze_alerts,  (0,200,255)),
            (f"DROWSY:{frame_drowsy}",       frame_drowsy,       (0,80,200)),
            (f"HEAD DOWN:{frame_head_down}", frame_head_down,    (0,80,255)),
        ]:
            if cnt:
                cv2.putText(frame,label,(10,yb),cv2.FONT_HERSHEY_SIMPLEX,0.62,color,2)
                yb -= 28

        out.write(frame)

    cap.release(); out.release()
    progress_bar.progress(1.0)
    status_text.text(f"✅ Done! {frame_num} frames processed.")

    return {
        'total_frames':      frame_num,
        'paper_exchanges':   total_paper_exchanges,
        'head_alerts':       total_head_alerts,
        'eye_lr_alerts':     total_gaze_left_right,
        'peeking_alerts':    total_peeking_alerts,
        'mobile_alerts':     total_mobile_alerts,
        'head_down_alerts':  total_head_down,
        'drowsy_alerts':     total_drowsy,
        'students_detected': len(v_student_data),
        'student_details':   {
            sid: {
                'name':            s.name,
                'head_alerts':     s.head_movement_alerts,
                'eye_lr_alerts':   s.eye_left_right_alerts,
                'peeking_alerts':  s.peeking_alerts,
            }
            for sid,s in v_student_data.items()
        }
    }

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    with st.expander("🤖 Model Status", expanded=False):
        c1,c2=st.columns(2)
        c1.metric("FaceNet","✅" if resnet else "❌")
        c2.metric("Face Det","✅" if face_detector else "❌")
        c1.metric("Paper Model","✅" if paper_model_loaded else "❌")
        c2.metric("Device Model","✅" if custom_model_loaded else "❌")
        st.caption(f"{'🖥️ GPU (CUDA)' if torch.cuda.is_available() else '💻 CPU'}")

    st.markdown("---")
    mode = st.radio("🎯 Select Mode",
                    ["📷 Register Students","🎥 Live Monitoring","📹 Video Analysis"])

    if mode in ["🎥 Live Monitoring","📹 Video Analysis"]:
        st.markdown("---")
        st.subheader("🔬 Detection Options")
        show_face_mesh       = st.checkbox("🕸️ Face Mesh",       value=True)
        show_hand_landmarks  = st.checkbox("✋ Hand Landmarks",   value=True)
        show_yolo_skeleton   = st.checkbox("🦴 YOLO Skeleton",   value=True)
        show_movement        = st.checkbox("🏃 Movement Trail",  value=True)
        show_gestures        = st.checkbox("👋 Gesture Detection",value=True)

        st.markdown("---")
        st.subheader("🚨 Yaw Cheating + Alarm")
        enable_yaw_cheating    = st.checkbox("Enable Yaw Detection",  value=True)
        yaw_threshold          = st.slider("Yaw Threshold (px)",      5,50,15,1)
        cheating_frames_needed = st.slider("Trigger After (frames)",  5,60,20,5)
        alert_duration_sec     = st.slider("Alert Duration (s)",      1,10,2,1)
        enable_alarm_sound     = st.checkbox("🔊 Alarm Sound",        value=True)
        show_yaw_status        = st.checkbox("Show Yaw on Video",     value=True)
        if platform.system()!="Windows" and enable_alarm_sound:
            st.caption("ℹ️ Non-Windows: terminal bell used")
        if st.button("🔄 Reset Yaw Data"):
            for d in ['yaw_cheating_counter','yaw_alert_active','yaw_alert_time','alarm_playing']:
                st.session_state[d].clear()
            st.success("Yaw data reset!")

        st.markdown("---")
        st.subheader("👀 Peeking Detection")
        enable_peeking_detection = st.checkbox("Enable Peeking",        value=True)
        peeking_threshold        = st.slider("Peeking Sensitivity",     0.3,0.9,0.5,0.1)
        show_peeking_overlay     = st.checkbox("Show Peeking Overlay",  value=True)
        if st.button("🔄 Reset Peeking Data"):
            st.session_state.peeking_detector.clear_all()
            st.success("Peeking data reset!")

        st.markdown("---")
        st.subheader("🪑 Seat Zones")
        enable_seat_mapping = st.checkbox("Smart Seat Detection", value=True)
        show_seat_zones     = st.checkbox("Show Seat Zones",      value=True)

        st.markdown("---")
        st.subheader("📄 Paper / Object Sharing")
        enable_exchange_detection = st.checkbox("Hand-Proximity Detection", value=True)
        enable_paper_yolo = st.checkbox("YOLO Paper Exchange (best.pt)",
                                        value=paper_model_loaded, disabled=not paper_model_loaded)
        exchange_sensitivity = st.slider("Proximity Threshold (px)",50,200,120,10)
        show_exchange_overlay= st.checkbox("Show Exchange Alerts",value=True)

        st.markdown("---")
        st.subheader("📱 Mobile/Watch Detection")
        enable_mobile_detection = st.checkbox("Detect Devices (best1.pt)",
                                              value=custom_model_loaded, disabled=not custom_model_loaded)

        st.markdown("---")
        st.subheader("👤 Unknown Students")
        track_unknown = st.checkbox("Track Unregistered Students", value=True)
        if track_unknown:
            st.caption(f"IDs start at 1000 | Tracking: {len(st.session_state.unknown_students)}")

        st.markdown("---")
        st.subheader("⚙️ Advanced")
        confidence_threshold = st.slider("YOLO Confidence",0.3,0.9,0.5,0.1)
        fps_limit            = st.slider("FPS Limit",5,30,15,5)

    else:
        show_face_mesh=True; show_hand_landmarks=True; show_yolo_skeleton=True
        show_movement=True; show_gestures=True
        enable_yaw_cheating=True; yaw_threshold=15; cheating_frames_needed=20
        alert_duration_sec=2; enable_alarm_sound=True; show_yaw_status=True
        enable_peeking_detection=True; peeking_threshold=0.5; show_peeking_overlay=True
        enable_seat_mapping=True; show_seat_zones=True
        enable_exchange_detection=True; enable_paper_yolo=paper_model_loaded
        exchange_sensitivity=120; show_exchange_overlay=True
        enable_mobile_detection=custom_model_loaded
        track_unknown=True; confidence_threshold=0.5; fps_limit=15

    if mode=="📷 Register Students":
        st.markdown("---")
        st.subheader("Register New Student")
        student_name = st.text_input("Student Name", placeholder="Enter name...")
        if st.button("📸 Start Registration") and student_name:
            st.session_state.capture_mode = True
            st.session_state.capture_name = student_name
            st.session_state.stop_registration = False
        st.markdown("---")
        st.subheader("📋 Registered Students")
        if st.session_state.student_database_facenet:
            for sid,data in st.session_state.student_database_facenet.items():
                c1,c2=st.columns([3,1])
                c1.write(f"**ID {sid}:** {data['name']} ({len(data['embeddings'])} samples)")
                if c2.button("🗑️",key=f"del_{sid}"):
                    del st.session_state.student_database_facenet[sid]
                    save_facenet_database(); st.rerun()
        else:
            st.info("No students registered yet")

    elif mode=="📹 Video Analysis":
        st.markdown("---")
        st.subheader("📹 Upload Video")
        uploaded_file = st.file_uploader("Choose video file",type=['mp4','avi','mov','mkv'])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.uploaded_video_path = tfile.name
            st.success(f"✅ {uploaded_file.name} uploaded!")
        if st.session_state.uploaded_video_path:
            if st.button("▶️ Start Full Analysis",type="primary",use_container_width=True):
                st.session_state.video_processing = True; st.rerun()

    elif mode=="🎥 Live Monitoring":
        run = st.checkbox("🎥 Start Monitoring")

# ============================================
# MAIN DISPLAY
# ============================================

col1,col2 = st.columns([2,1])
with col1:
    frame_window = st.empty()
with col2:
    stats_placeholder    = st.empty()
    activity_placeholder = st.empty()

# ============================================
# REGISTRATION MODE
# ============================================

if mode=="📷 Register Students":
    if 'capture_mode' in st.session_state and st.session_state.capture_mode and not st.session_state.stop_registration:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera not accessible!")
            st.session_state.capture_mode = False
        else:
            captured_count=0; max_captures=10
            progress_bar=st.progress(0); status_text=st.empty()
            if st.button("⏹️ Stop Registration"):
                st.session_state.stop_registration=True; st.session_state.capture_mode=False
                cap.release(); st.rerun()
            while captured_count<max_captures and not st.session_state.stop_registration:
                ret,frame=cap.read()
                if not ret: break
                frame=cv2.flip(frame,1)
                success,box,message=register_student_facenet(frame,st.session_state.next_id,st.session_state.capture_name)
                if success:
                    captured_count+=1; x1,y1,x2,y2=box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    cv2.putText(frame,f"✓ {captured_count}/{max_captures}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                    progress_bar.progress(captured_count/max_captures)
                    status_text.success(f"✅ {captured_count}/{max_captures} captured!")
                    time.sleep(0.5)
                    if captured_count>=max_captures:
                        st.session_state.activity_log.append(f"✅ Student #{st.session_state.next_id} ({st.session_state.capture_name}) registered!")
                        st.session_state.next_id+=1; st.session_state.capture_mode=False
                        cap.release(); st.success(f"🎉 {st.session_state.capture_name} registered!")
                        st.balloons(); time.sleep(2); st.rerun(); break
                else:
                    cv2.putText(frame,message,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    status_text.warning(f"⏳ {message}")
                cv2.putText(frame,f"Capturing: {captured_count}/{max_captures}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                frame_window.image(frame,channels="BGR",use_column_width=True)
            cap.release()
    else:
        frame_window.info("📷 Enter student name and click 'Start Registration'")

# ============================================
# VIDEO ANALYSIS MODE
# ============================================

elif mode=="📹 Video Analysis":
    if st.session_state.video_processing and st.session_state.uploaded_video_path:
        st.markdown("### 🎬 Video Processing in Progress...")
        progress_bar = st.progress(0)
        status_text  = st.empty()
        output_path  = tempfile.NamedTemporaryFile(delete=False,suffix='_output.mp4').name

        results = process_video_with_all_detections(
            st.session_state.uploaded_video_path, output_path, progress_bar, status_text)

        if results:
            st.success("✅ Video processing complete!")

            # ── metrics row ──────────────────────────────────────────
            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
            c1.metric("📄 Paper",      results['paper_exchanges'])
            c2.metric("↔️ Head Turn",  results['head_alerts'])
            c3.metric("👁️ Eyes L/R",   results['eye_lr_alerts'])
            c4.metric("👀 Peeking",    results['peeking_alerts'])
            c5.metric("📱 Devices",    results['mobile_alerts'])
            c6.metric("📵 Head Down",  results['head_down_alerts'])
            c7.metric("😴 Drowsy",     results['drowsy_alerts'])
            c8.metric("👥 Students",   results['students_detected'])

            # ── per-student table ─────────────────────────────────────
            if results['student_details']:
                st.markdown("#### 📊 Per-Student Alert Summary")
                rows = []
                for sid,d in results['student_details'].items():
                    rows.append({
                        'Student':       d['name'],
                        'Head Turns':    d['head_alerts'],
                        'Eye L/R':       d['eye_lr_alerts'],
                        'Peeking':       d['peeking_alerts'],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # ── download button — ALWAYS outside any if block ─────────
            st.markdown("---")
            with open(output_path,'rb') as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f,
                    file_name="cheating_detection_output.mp4",
                    mime="video/mp4",
                    use_container_width=True)

            st.video(output_path)
            st.session_state.video_processing = False

            if st.button("🔄 Process Another Video", use_container_width=True):
                st.session_state.uploaded_video_path = None; st.rerun()

    elif not st.session_state.video_processing:
        st.info("📹 Upload a video from the sidebar and click **Start Full Analysis**")
        st.markdown("""
### 🎯 What This Analyses:
| Feature | Detection Method |
|---------|-----------------|
| Face Recognition | FaceNet + OpenCV DNN |
| **Head LEFT / RIGHT** | **MediaPipe solvePnP — instant alert** |
| **Eyes LEFT / RIGHT** | **MediaPipe Iris Landmarks — instant alert** |
| Sustained Peeking | 90-frame sliding window |
| Head Down | Pitch angle > 25° |
| Drowsiness | Eye Aspect Ratio (EAR) |
| Paper Exchange | YOLOv8 Tracking (best.pt) |
| Mobile / Watch | YOLOv8 Custom (best1.pt) |
| Body Bounding Box | YOLOv8n person detection |
""")

# ============================================
# LIVE MONITORING MODE
# ============================================

elif mode=="🎥 Live Monitoring":
    if 'run' not in dir() or not run:
        frame_window.info("📹 Check 'Start Monitoring' in sidebar to begin")
    elif not st.session_state.student_database_facenet:
        st.warning("⚠️ No students registered! Please register students first.")
    else:
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

        st.session_state.peeking_detector.threshold = peeking_threshold
        if enable_exchange_detection:
            st.session_state.exchange_detector.proximity_threshold = exchange_sensitivity

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Camera not accessible!")
        else:
            last_frame_time=0; frame_delay=1.0/fps_limit
            gaze_alert_cd_live  = {}   # gaze cooldown
            head_alert_cd_live  = {}   # head cooldown

            while run:
                ct=time.time()
                if ct-last_frame_time<frame_delay: time.sleep(0.01); continue
                last_frame_time=ct

                ret,frame=cap.read()
                if not ret: break

                frame=cv2.flip(frame,1)
                st.session_state.frame_count+=1
                fh,fw=frame.shape[:2]

                if enable_seat_mapping and show_seat_zones:
                    t=fw//3
                    cv2.line(frame,(t,0),(t,fh),(255,255,0),2)
                    cv2.line(frame,(t*2,0),(t*2,fh),(255,255,0),2)
                    cv2.putText(frame,"LEFT",(10,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                    cv2.putText(frame,"CENTER",(t+10,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                    cv2.putText(frame,"RIGHT",(t*2+10,fh-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

                rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                person_count=0; recognized_count=0; active_students=[]
                gesture_counts={"RAISED_HAND":0,"POINTING":0,"FIST":0,"OPEN_PALM":0}
                peeking_count=0; peeking_students_list=[]
                exchange_count=0; yaw_alert_count=0
                mobile_count=0; paper_exchange_count=0
                hands_detected_count=0; hands_tracked_count=0
                eye_lr_count=0; head_turn_count=0

                # ── YOLO person body boxes ─────────────────────────────
                try:
                    p_body=person_model(frame,verbose=False,conf=0.5)
                    for r in p_body:
                        for box in r.boxes:
                            if int(box.cls[0])==0:
                                bx1,by1,bx2,by2=map(int,box.xyxy[0].cpu().numpy())
                                cv2.rectangle(frame,(bx1,by1),(bx2,by2),(255,255,0),2)
                except Exception: pass

                # ── FACE DETECTION + RECOGNITION ──────────────────────
                faces=detect_faces_opencv(frame)
                recognized_faces=[]

                for face in faces:
                    x1,y1,x2,y2=face['box']
                    student_id,confidence=recognize_student_facenet(frame,face['box'])
                    cx,cy=(x1+x2)/2,(y1+y2)/2

                    if student_id:
                        person_count+=1; recognized_count+=1
                        student_name=st.session_state.student_database_facenet[student_id]['name']
                        is_unknown=False
                    elif track_unknown:
                        person_count+=1
                        student_id=assign_unknown_student_id(cx,cy,st.session_state.unknown_students)
                        student_name=f"Unknown-{student_id}"; confidence=0; is_unknown=True
                        if student_id not in st.session_state.unknown_students:
                            st.session_state.unknown_students[student_id]=Student(student_id,student_name)
                    else: continue

                    recognized_faces.append({'box':(x1,y1,x2,y2),'student_id':student_id,
                                             'student_name':student_name,'confidence':confidence,
                                             'is_unknown':is_unknown})

                    color=(128,128,128) if is_unknown else ((0,255,0) if confidence>85 else (0,255,255) if confidence>70 else (0,165,255))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                    cv2.putText(frame,f"ID:{student_id}",(x1,y1-65),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                    cv2.putText(frame,student_name,(x1,y1-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)
                    cv2.putText(frame,f"Conf:{confidence:.1f}%" if not is_unknown else f"TempID:{student_id}",
                               (x1,y1-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                    if is_unknown:
                        if student_id not in st.session_state.student_data:
                            st.session_state.student_data[student_id]=st.session_state.unknown_students[student_id]
                            st.session_state.activity_log.append(f"👤 {student_name} detected at {time.strftime('%H:%M:%S')}")
                    else:
                        if student_id not in st.session_state.student_data:
                            st.session_state.student_data[student_id]=Student(student_id,student_name)
                            st.session_state.activity_log.append(f"👋 {student_name} detected at {time.strftime('%H:%M:%S')}")

                    student=st.session_state.student_data[student_id]
                    student.update_position(cx,cy); active_students.append(student_id)

                    if enable_seat_mapping:
                        student.seat_zone=determine_seat_zone(cx,cy,fw)
                        if show_seat_zones:
                            cv2.putText(frame,f"Zone:{student.seat_zone.upper()}",(x1,y2+145),
                                       cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

                    engagement=student.calculate_engagement()
                    cv2.putText(frame,f"Eng:{engagement:.0f}%",(x1,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                    if show_movement and len(student.positions)>2:
                        pts=list(student.positions)
                        for j in range(1,len(pts)):
                            cv2.line(frame,(int(pts[j-1][0]),int(pts[j-1][1])),
                                    (int(pts[j][0]),int(pts[j][1])),color,2)

                # ── MEDIAPIPE FACE MESH ────────────────────────────────
                face_results=face_mesh.process(rgb_frame)
                hand_results=hands_detector.process(rgb_frame)

                if hand_results.multi_hand_landmarks:
                    hands_detected_count=len(hand_results.multi_hand_landmarks)

                if face_results.multi_face_landmarks:
                    for face_idx,face_lm in enumerate(face_results.multi_face_landmarks):
                        lm_list=[l for l in face_lm.landmark]
                        xc=[l.x*fw for l in lm_list]; yc=[l.y*fh for l in lm_list]
                        mp_x1,mp_y1=int(min(xc)),int(min(yc))
                        mp_x2,mp_y2=int(max(xc)),int(max(yc))

                        # ── FIX: center-distance matching ──────────────
                        matched_id,matched_bbox=match_mediapipe_to_face(
                            mp_x1,mp_y1,mp_x2,mp_y2,recognized_faces)

                        if matched_id is None:
                            mp_cx=(mp_x1+mp_x2)//2
                            matched_id=f"Unk_{mp_cx}"
                            matched_bbox=(mp_x1,mp_y1,mp_x2,mp_y2)
                            if matched_id not in st.session_state.student_data:
                                st.session_state.student_data[matched_id]=Student(matched_id,matched_id)

                        fx1,fy1,fx2,fy2=matched_bbox

                        if show_face_mesh:
                            mp_drawing.draw_landmarks(
                                image=frame,landmark_list=face_lm,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                        # engagement
                        if detect_head_pose(face_lm)=="LOOKING_AWAY":
                            if matched_id in st.session_state.student_data:
                                st.session_state.student_data[matched_id].looking_away_frames+=1

                        # ── HEAD DIRECTION alert ───────────────────────
                        head_dir=detect_head_direction_detailed(face_lm,fw)
                        hd_color=(0,0,255) if head_dir in ('LEFT','RIGHT') else (0,255,0)
                        cv2.putText(frame,f"Head:{head_dir}",(fx1,fy2+22),
                                   cv2.FONT_HERSHEY_SIMPLEX,0.55,hd_color,2)

                        if head_dir in ('LEFT','RIGHT'):
                            head_turn_count+=1
                            last_h=head_alert_cd_live.get(matched_id,-999)
                            if time.time()-last_h>2.0:
                                head_alert_cd_live[matched_id]=time.time()
                                cv2.rectangle(frame,(fx1,fy1),(fx2,fy2),(0,0,255),3)
                                cv2.putText(frame,f"HEAD {head_dir}!",(fx1,fy1-15),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
                                st.session_state.activity_log.append(
                                    f"↔️ HEAD {head_dir}: {matched_id} at {time.strftime('%H:%M:%S')}")
                                if matched_id in st.session_state.student_data:
                                    st.session_state.student_data[matched_id].head_movement_alerts+=1

                        # ── IRIS GAZE alert ────────────────────────────
                        gaze_info=get_iris_gaze(lm_list,fw,fh)
                        if gaze_info:
                            cv2.circle(frame,gaze_info['left_iris'], 4,(0,255,255),-1)
                            cv2.circle(frame,gaze_info['right_iris'],4,(0,255,255),-1)
                            gd=gaze_info['direction']
                            gc=(0,255,0) if gd=='GAZE_CENTER' else (0,200,255)
                            cv2.putText(frame,gd,(fx1,fy2+45),cv2.FONT_HERSHEY_SIMPLEX,0.5,gc,2)

                            if gd in ('GAZE_LEFT','GAZE_RIGHT'):
                                eye_lr_count+=1
                                key=f"gaze_{matched_id}"
                                last_g=gaze_alert_cd_live.get(key,-999)
                                if time.time()-last_g>2.0:
                                    gaze_alert_cd_live[key]=time.time()
                                    side=gd.replace('GAZE_','')
                                    cv2.putText(frame,f"EYES {side}!",(fx1,fy1-35),
                                               cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,200,255),2)
                                    for idx in LEFT_EYE+RIGHT_EYE:
                                        ex=int(lm_list[idx].x*fw); ey=int(lm_list[idx].y*fh)
                                        cv2.circle(frame,(ex,ey),2,(0,200,255),-1)
                                    st.session_state.activity_log.append(
                                        f"👁️ EYES {side}: {matched_id} at {time.strftime('%H:%M:%S')}")
                                    if matched_id in st.session_state.student_data:
                                        st.session_state.student_data[matched_id].eye_left_right_alerts+=1

                        # ── PEEKING ────────────────────────────────────
                        if enable_peeking_detection:
                            st.session_state.peeking_detector.update(matched_id,head_dir)
                            peeking_info=st.session_state.peeking_detector.get_peeking_score(matched_id)

                            if matched_id in st.session_state.peeking_detector.history:
                                hlen=len(st.session_state.peeking_detector.history[matched_id])
                                cv2.putText(frame,f"Frames:{hlen}/90",(fx1,fy2+68),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

                            if enable_seat_mapping and matched_id in st.session_state.student_data:
                                sz=st.session_state.student_data[matched_id].seat_zone
                                if sz:
                                    ns=get_neighbor_direction(sz,head_dir)
                                    if ns and ns!='NO_NEIGHBOR':
                                        cv2.putText(frame,f"👥{ns}",(fx1,fy2+88),
                                                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,165,0),2)

                            if peeking_info['is_peeking'] and show_peeking_overlay:
                                peeking_students_list.append(matched_id)
                                cv2.putText(frame,f"PEEKING {peeking_info['direction']}!",(fx1,fy2+108),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
                                s=st.session_state.student_data.get(matched_id)
                                if s and time.time()-s.last_peeking_alert>3:
                                    s.peeking_alerts+=1; s.last_peeking_alert=time.time()
                                    st.session_state.activity_log.append(
                                        f"⚠️ {s.name} peeking {peeking_info['direction']} at {time.strftime('%H:%M:%S')}")

                        # ── YAW CHEATING ───────────────────────────────
                        if enable_yaw_cheating:
                            yaw=get_head_yaw(face_lm.landmark,fw)
                            frame,log_msg,yaw_dir=process_yaw_cheating(
                                face_idx,yaw,frame,face_lm.landmark,fw,
                                yaw_threshold,cheating_frames_needed,alert_duration_sec,enable_alarm_sound)
                            if st.session_state.yaw_alert_active.get(face_idx,False):
                                yaw_alert_count+=1
                            if log_msg:
                                st.session_state.activity_log.append(log_msg)
                                if matched_id and matched_id in st.session_state.student_data:
                                    st.session_state.student_data[matched_id].yaw_cheating_alerts+=1
                            if show_yaw_status:
                                nose=face_lm.landmark[1]
                                nx,ny=int(nose.x*fw),int(nose.y*fh)
                                cv2.putText(frame,f"Yaw:{yaw:.1f}px",(nx-50,ny+25),
                                           cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,50),1)

                # ── HANDS ──────────────────────────────────────────────
                if hand_results.multi_hand_landmarks:
                    for idx,hand_lm in enumerate(hand_results.multi_hand_landmarks):
                        if show_hand_landmarks:
                            mp_drawing.draw_landmarks(frame,hand_lm,mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            wrist=hand_lm.landmark[0]
                            cv2.circle(frame,(int(wrist.x*fw),int(wrist.y*fh)),8,(0,255,0),-1)

                        if enable_exchange_detection:
                            wrist=hand_lm.landmark[0]
                            hx,hy=int(wrist.x*fw),int(wrist.y*fh)
                            min_dist=float('inf'); owner_id=None
                            for fi in recognized_faces:
                                fx1_h,fy1_h,fx2_h,fy2_h=fi['box']
                                d=np.sqrt((hx-(fx1_h+fx2_h)/2)**2+(hy-(fy1_h+fy2_h)/2)**2)
                                if d<min_dist and d<500: min_dist=d; owner_id=fi['student_id']
                            if owner_id:
                                hands_tracked_count+=1
                                st.session_state.exchange_detector.update_hand_position(
                                    owner_id,hand_lm,fw,fh,'right' if idx%2==0 else 'left')
                                if enable_seat_mapping and owner_id in st.session_state.student_data:
                                    sz=st.session_state.student_data[owner_id].seat_zone
                                    if sz:
                                        zv=st.session_state.exchange_detector.check_zone_crossing(owner_id,sz,hx,fw)
                                        if zv and show_exchange_overlay:
                                            cv2.circle(frame,(hx,hy),15,(0,0,255),3)
                                            cv2.putText(frame,"ZONE CROSS!",(hx-70,hy-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                                            st.session_state.activity_log.append(
                                                f"⚠️ Zone crossing: {st.session_state.student_data[owner_id].name} at {time.strftime('%H:%M:%S')}")

                        if show_gestures:
                            gesture=detect_hand_gesture(hand_lm)
                            gesture_counts[gesture]+=1
                            cx2=int(hand_lm.landmark[0].x*fw); cy2=int(hand_lm.landmark[0].y*fh)
                            cv2.putText(frame,gesture,(cx2-50,cy2-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

                # ── YOLO SKELETON ──────────────────────────────────────
                if show_yolo_skeleton:
                    try:
                        for r in pose_model(frame,stream=True,verbose=False,conf=confidence_threshold):
                            if r.keypoints is not None and hasattr(r.keypoints,'xy') and r.keypoints.xy is not None:
                                for i in range(len(r.keypoints.xy)):
                                    kpts=r.keypoints.xy[i].cpu().numpy()
                                    if hasattr(r.keypoints,'conf') and r.keypoints.conf is not None and len(r.keypoints.conf)>i:
                                        draw_skeleton(frame,kpts,r.keypoints.conf[i].cpu().numpy())
                    except Exception: pass

                # ── LIVE PAPER EXCHANGE ────────────────────────────────
                if enable_paper_yolo and paper_model_loaded:
                    try:
                        per=person_model.track(frame,persist=True,verbose=False,conf=0.5)
                        par=paper_model.track(frame,persist=True,verbose=False,conf=0.5)
                        if per and par:
                            for box in par[0].boxes:
                                px1,py1,px2,py2=map(int,box.xyxy[0].cpu().numpy())
                                cv2.rectangle(frame,(px1,py1),(px2,py2),(255,165,0),2)
                                cv2.putText(frame,"PAPER",(px1,py1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,165,0),2)
                            cur_time=st.session_state.frame_count/fps_limit
                            for ex in st.session_state.paper_detector.process_frame(par[0],per[0],cur_time):
                                paper_exchange_count+=1
                                ex_x,ex_y=ex['position']
                                cv2.circle(frame,(ex_x,ex_y),50,(0,0,255),4)
                                cv2.putText(frame,"PAPER EXCHANGE!",(ex_x-120,ex_y-60),cv2.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),3)
                                st.session_state.activity_log.append(f"📄 Paper exchange at {time.strftime('%H:%M:%S')}")
                    except Exception: pass

                # ── LIVE MOBILE/WATCH ──────────────────────────────────
                if enable_mobile_detection and st.session_state.mobile_watch_detector:
                    try:
                        for det in st.session_state.mobile_watch_detector.detect(frame,confidence_threshold):
                            mobile_count+=1
                            dx1,dy1,dx2,dy2=det['box']
                            cv2.rectangle(frame,(dx1,dy1),(dx2,dy2),(0,0,255),3)
                            cv2.putText(frame,f"{det['class']} {det['confidence']:.2f}",
                                       (dx1,dy1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                    except Exception: pass

                # ── HAND PROXIMITY EXCHANGE ────────────────────────────
                if enable_exchange_detection:
                    for ev in st.session_state.exchange_detector.detect_hand_proximity():
                        exchange_count+=1
                        s1n=st.session_state.student_data[ev['student1']].name if ev['student1'] in st.session_state.student_data else str(ev['student1'])
                        s2n=st.session_state.student_data[ev['student2']].name if ev['student2'] in st.session_state.student_data else str(ev['student2'])
                        ex_p,ey_p=ev['position']
                        if show_exchange_overlay:
                            cv2.circle(frame,(ex_p,ey_p),30,(0,140,255),3)
                            cv2.putText(frame,"HANDS CLOSE!",(ex_p-80,ey_p-40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,140,255),2)
                        if ev['student1'] in st.session_state.student_data:
                            st.session_state.student_data[ev['student1']].exchange_alerts+=1
                        st.session_state.activity_log.append(f"🤝 Hand proximity: {s1n}↔{s2n} at {time.strftime('%H:%M:%S')}")
                    st.session_state.exchange_detector.clear_old_data()

                # ── STATS OVERLAY ──────────────────────────────────────
                elapsed=int(time.time()-st.session_state.start_time)
                peeking_count=len(peeking_students_list)

                cv2.putText(frame,f"Total:{person_count}|Reg:{recognized_count}|Unk:{person_count-recognized_count}|Hands:{hands_detected_count}",
                           (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,0),2)

                if yaw_alert_count>0:
                    cv2.putText(frame,f"YAW CHEAT:{yaw_alert_count}",(10,fh-80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if head_turn_count>0:
                    cv2.putText(frame,f"HEAD TURN:{head_turn_count}",(10,fh-55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if eye_lr_count>0:
                    cv2.putText(frame,f"EYES L/R:{eye_lr_count}",(10,fh-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,255),2)
                if peeking_count>0:
                    cv2.putText(frame,f"PEEKING:{peeking_count}",(fw//2,fh-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if paper_exchange_count>0:
                    cv2.putText(frame,f"PAPER:{paper_exchange_count}",(fw-200,fh-55),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                if mobile_count>0:
                    cv2.putText(frame,f"DEVICE:{mobile_count}",(fw-200,fh-30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                frame_window.image(frame,channels="BGR",use_column_width=True)

                # ── STATS PANEL ────────────────────────────────────────
                with stats_placeholder.container():
                    st.markdown("### 📊 Live Statistics")
                    sc=st.columns(8)
                    sc[0].metric("👥 Total",  person_count)
                    sc[1].metric("✅ Reg",    recognized_count)
                    sc[2].metric("❓ Unk",    person_count-recognized_count)
                    sc[3].metric("👋 Hands",  hands_detected_count)
                    sc[4].metric("↔️ Head",   head_turn_count, delta="🚨" if head_turn_count>0 else None)
                    sc[5].metric("👁️ Eyes",   eye_lr_count,    delta="🚨" if eye_lr_count>0    else None)
                    sc[6].metric("👀 Peek",   peeking_count,   delta="⚠️" if peeking_count>0   else None)
                    sc[7].metric("📄 Paper",  paper_exchange_count)
                    st.metric("⏱️ Session",f"{elapsed}s")

                    if st.session_state.student_data:
                        st.markdown("#### 📈 Student Status")
                        for sid in sorted(st.session_state.student_data.keys()):
                            s=st.session_state.student_data[sid]
                            eng=s.calculate_engagement()
                            mov="🏃" if s.is_moving() else "🧍"
                            act="🟢" if sid in active_students else "🔴"
                            pi=st.session_state.peeking_detector.get_peeking_score(sid)
                            peek_icon="⚠️👀" if pi['is_peeking'] else ""
                            zone_txt=f"|🪑{s.seat_zone.upper()}" if s.seat_zone else ""
                            display=f"{act}{mov}{peek_icon} {s.name}(ID {sid}): {eng:.0f}%{zone_txt}"
                            st.progress(eng/100, text=display)
                            capts=[]
                            if s.head_movement_alerts>0: capts.append(f"↔️ Head:{s.head_movement_alerts}")
                            if s.eye_left_right_alerts>0: capts.append(f"👁️ Eyes LR:{s.eye_left_right_alerts}")
                            if s.peeking_alerts>0: capts.append(f"👀 Peek:{s.peeking_alerts}")
                            if s.exchange_alerts>0: capts.append(f"🤝 Hands:{s.exchange_alerts}")
                            if s.yaw_cheating_alerts>0: capts.append(f"🚨 Yaw:{s.yaw_cheating_alerts}")
                            if capts: st.caption("  "+" | ".join(capts))

                # ── ACTIVITY LOG ───────────────────────────────────────
                with activity_placeholder.container():
                    st.markdown("### 📝 Activity Log")
                    for log in reversed(list(st.session_state.activity_log)[-12:]):
                        if "📄" in log:
                            st.markdown(f"<span style='color:#FF8C00;font-weight:bold'>{log}</span>",unsafe_allow_html=True)
                        elif "🚨" in log or "YAW" in log:
                            st.markdown(f"<span style='color:red;font-weight:bold'>{log}</span>",unsafe_allow_html=True)
                        elif "↔️" in log or "HEAD" in log:
                            st.markdown(f"<span style='color:#FF4500;font-weight:bold'>{log}</span>",unsafe_allow_html=True)
                        elif "👁️" in log or "EYES" in log:
                            st.markdown(f"<span style='color:#00BFFF;font-weight:bold'>{log}</span>",unsafe_allow_html=True)
                        elif "⚠️" in log:
                            st.markdown(f"<span style='color:#cc0000;font-weight:bold'>{log}</span>",unsafe_allow_html=True)
                        elif "🤝" in log:
                            st.markdown(f"<span style='color:orange'>{log}</span>",unsafe_allow_html=True)
                        else:
                            st.text(log)

            cap.release()
            st.session_state.start_time=None

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#666;padding:1rem;'>
<p style='font-size:1.1rem;font-weight:600;'>🎓 Smart Classroom — Advanced Cheating Detection System</p>
<p style='font-size:0.9rem;'>
FaceNet • YOLOv8 • MediaPipe • Head Turn Alert • Eye Gaze Alert •
Peeking • Drowsy • Paper Exchange • Device Detection • Body Bounding Box
</p>
</div>
""", unsafe_allow_html=True)