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
import csv
from datetime import datetime
import pandas as pd
import tempfile

# Patch Ultralytics
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

# Enhanced page config
st.set_page_config(
    page_title="Advanced Cheating Detection System", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        margin: 0.5rem 0;
    }
    .alert-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        font-weight: 600;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .info-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎓 Advanced Cheating Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Exam Monitoring with Multi-Modal Detection</p>', unsafe_allow_html=True)

# Load YOLO models
@st.cache_resource
def load_models():
    pose_model = YOLO("yolov8s-pose.pt")
    person_model = YOLO("yolov8n.pt")
    
    try:
        # Paper detection model (your custom model)
        paper_model = YOLO("best.pt")
        paper_model_loaded = True
        st.success("✅ Paper detection model (best.pt) loaded!")
    except:
        st.warning("⚠️ Paper model 'best.pt' not found. Paper exchange detection disabled.")
        paper_model = None
        paper_model_loaded = False
    
    try:
        # Mobile/Watch detection model
        custom_model = YOLO("best1.pt")
        custom_model_loaded = True
    except:
        st.warning("⚠️ Custom model 'best1.pt' not found. Mobile/watch detection disabled.")
        custom_model = None
        custom_model_loaded = False
    
    return pose_model, person_model, paper_model, paper_model_loaded, custom_model, custom_model_loaded

pose_model, person_model, paper_model, paper_model_loaded, custom_model, custom_model_loaded = load_models()

# Load FaceNet
@st.cache_resource
def load_facenet_model():
    try:
        from facenet_pytorch import InceptionResnetV1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet, device
    except Exception as e:
        st.error(f"FaceNet loading error: {e}")
        return None, None

resnet, device = load_facenet_model()

# Load OpenCV DNN Face Detector
@st.cache_resource
def load_opencv_face_detector():
    try:
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            st.info("📥 Downloading face detection models...")
            import urllib.request
            
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
            urllib.request.urlretrieve(model_url, model_path)
        
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        return net
    except Exception as e:
        st.error(f"Face detector loading error: {e}")
        return None

face_detector = load_opencv_face_detector()

# Load MediaPipe
@st.cache_resource
def load_mediapipe():
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    hands = mp_hands.Hands(
        max_num_hands=20,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    
    return face_mesh, hands, mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh

face_mesh, hands, mp_drawing, mp_drawing_styles, mp_hands, mp_face_mesh = load_mediapipe()

# Database
FACENET_DB = "facenet_database.pkl"

# Initialize session state
def init_session_state():
    defaults = {
        'student_database_facenet': {},
        'next_id': 1,
        'student_data': {},
        'activity_log': deque(maxlen=100),
        'frame_count': 0,
        'start_time': None,
        'stop_registration': False,
        'seat_mapping': {},
        'video_mode': False,
        'uploaded_video_path': None,
        'unknown_student_counter': 1000,
        'unknown_students': {},
        'peeking_detector': None,
        'exchange_detector': None,
        'paper_detector': None,
        'head_direction_tracker': {},
        'mobile_watch_detector': None,
        'cheating_events': [],
        'paper_owner': {},
        'owner_change_time': {},
        'event_logged': {},
        'alert_time_tracker': {},
        'total_exchanges': 0,
        'video_processing': False,
        'output_video_path': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Load database
if Path(FACENET_DB).exists():
    with open(FACENET_DB, 'rb') as f:
        st.session_state.student_database_facenet = pickle.load(f)
    if st.session_state.student_database_facenet:
        st.session_state.next_id = max(st.session_state.student_database_facenet.keys()) + 1

# ============================================
# PEEKING DETECTION MODULE
# ============================================

class PeekingDetector:
    """Enhanced peeking detection with head direction tracking"""
    
    def __init__(self, window_size=90, threshold=0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.history = {}
        
    def update(self, student_id, head_direction):
        if student_id not in self.history:
            self.history[student_id] = deque(maxlen=self.window_size)
        
        self.history[student_id].append({
            'direction': head_direction,
            'timestamp': time.time()
        })
    
    def get_peeking_score(self, student_id):
        if student_id not in self.history or len(self.history[student_id]) < 30:
            return {'score': 0, 'direction': None, 'is_peeking': False}
        
        history = list(self.history[student_id])
        left_count = sum(1 for h in history if h['direction'] == 'LEFT')
        right_count = sum(1 for h in history if h['direction'] == 'RIGHT')
        
        total = len(history)
        left_ratio = left_count / total
        right_ratio = right_count / total
        
        is_peeking_left = left_ratio >= self.threshold
        is_peeking_right = right_ratio >= self.threshold
        
        if is_peeking_left:
            return {'score': int(left_ratio * 100), 'direction': 'LEFT', 'is_peeking': True}
        elif is_peeking_right:
            return {'score': int(right_ratio * 100), 'direction': 'RIGHT', 'is_peeking': True}
        else:
            max_ratio = max(left_ratio, right_ratio)
            return {
                'score': int(max_ratio * 100),
                'direction': 'LEFT' if left_ratio > right_ratio else 'RIGHT',
                'is_peeking': False
            }
    
    def clear_all(self):
        self.history.clear()

# ============================================
# PAPER EXCHANGE DETECTION MODULE
# ============================================

class PaperExchangeDetector:
    """Detects paper exchanges using YOLO tracking"""
    
    def __init__(self, time_threshold=0.5):
        self.time_threshold = time_threshold
        self.paper_owner = {}
        self.owner_change_time = {}
        self.event_logged = {}
        self.alert_time_tracker = {}
        self.total_exchanges = 0
        self.exchange_events = []
        
    def process_frame(self, paper_results, person_results, current_time_sec):
        """Process frame for paper exchange detection"""
        exchanges_detected = []
        
        if paper_results.boxes.id is None or person_results.boxes.id is None:
            return exchanges_detected
        
        paper_boxes = paper_results.boxes.xyxy.cpu().numpy()
        paper_ids = paper_results.boxes.id.cpu().numpy()
        person_boxes = person_results.boxes.xyxy.cpu().numpy()
        person_ids = person_results.boxes.id.cpu().numpy()
        person_classes = person_results.boxes.cls.cpu().numpy()
        
        persons = [
            (person_boxes[i], int(person_ids[i]))
            for i in range(len(person_classes)) if person_classes[i] == 0
        ]
        
        for i, p_box in enumerate(paper_boxes):
            p_id = int(paper_ids[i])
            x1, y1, x2, y2 = map(int, p_box)
            px_center = (x1 + x2) / 2
            py_center = (y1 + y2) / 2
            current_owner = None
            
            for person_box, person_id in persons:
                if (person_box[0] < px_center < person_box[2] and
                    person_box[1] < py_center < person_box[3]):
                    current_owner = person_id
                    break
            
            if p_id not in self.paper_owner:
                self.paper_owner[p_id] = current_owner
                self.event_logged[p_id] = False
                self.alert_time_tracker[p_id] = None
            else:
                old_owner = self.paper_owner[p_id]
                
                if self.paper_owner[p_id] != current_owner and current_owner is not None:
                    if p_id not in self.owner_change_time or self.owner_change_time[p_id] is None:
                        self.owner_change_time[p_id] = time.time()
                    elif time.time() - self.owner_change_time[p_id] > self.time_threshold:
                        if not self.event_logged[p_id]:
                            self.total_exchanges += 1
                            self.event_logged[p_id] = True
                            self.alert_time_tracker[p_id] = current_time_sec
                            
                            exchange_event = {
                                'timestamp': current_time_sec,
                                'paper_id': p_id,
                                'old_owner': old_owner,
                                'new_owner': current_owner,
                                'position': (int(px_center), int(py_center))
                            }
                            
                            self.exchange_events.append(exchange_event)
                            exchanges_detected.append(exchange_event)
                            
                            self.paper_owner[p_id] = current_owner
                            self.owner_change_time[p_id] = None
                else:
                    self.owner_change_time[p_id] = None
                    self.event_logged[p_id] = False
        
        return exchanges_detected
    
    def get_active_alerts(self, current_time_sec, alert_duration=2.0):
        """Get currently active alerts"""
        active = []
        for p_id, alert_start in self.alert_time_tracker.items():
            if alert_start is not None and current_time_sec - alert_start <= alert_duration:
                active.append(p_id)
        return active

# ============================================
# MOBILE/WATCH DETECTION MODULE
# ============================================

class MobileWatchDetector:
    """Detects mobile phones and watches using custom YOLO model"""
    
    def __init__(self, model):
        self.model = model
        self.detection_history = {}
        
    def detect(self, frame, confidence_threshold=0.5):
        """Detect mobile/watch in frame"""
        if self.model is None:
            return []
        
        results = self.model(frame, conf=confidence_threshold, verbose=False)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                class_name = self.model.names[cls]
                
                detections.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                })
        
        return detections
    
    def update_history(self, student_id, detected):
        """Update detection history for a student"""
        if student_id not in self.detection_history:
            self.detection_history[student_id] = deque(maxlen=30)
        
        self.detection_history[student_id].append({
            'detected': detected,
            'timestamp': time.time()
        })
    
    def is_persistent_detection(self, student_id, threshold=0.6):
        """Check if detection is persistent (not just a flash)"""
        if student_id not in self.detection_history:
            return False
        
        history = list(self.detection_history[student_id])
        if len(history) < 10:
            return False
        
        detected_count = sum(1 for h in history if h['detected'])
        return (detected_count / len(history)) >= threshold

# ============================================
# HEAD DIRECTION TRACKING
# ============================================

def get_head_yaw(landmarks, image_width):
    """Calculate head yaw using MediaPipe landmarks"""
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]
    
    eye_center_x = (left_eye.x + right_eye.x) / 2
    nose_x = nose_tip.x
    yaw = (nose_x - eye_center_x) * image_width
    
    return yaw

def detect_head_direction_detailed(face_landmarks, image_width):
    """Enhanced head direction detection"""
    yaw = get_head_yaw(face_landmarks.landmark, image_width)
    
    if yaw < -20:
        return 'LEFT'
    elif yaw > 20:
        return 'RIGHT'
    elif abs(yaw) <= 20:
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        forehead = face_landmarks.landmark[10]
        
        vertical_deviation = nose_tip.y - ((chin.y + forehead.y) / 2)
        
        if vertical_deviation > 0.03:
            return 'DOWN'
        elif vertical_deviation < -0.03:
            return 'UP'
        else:
            return 'CENTER'
    
    return 'CENTER'

# ============================================
# STUDENT CLASS
# ============================================

class Student:
    def __init__(self, student_id, name="Unknown"):
        self.id = student_id
        self.name = name
        self.positions = deque(maxlen=30)
        self.hand_raised_frames = 0
        self.looking_away_frames = 0
        self.total_frames = 0
        self.engagement_score = 100
        self.last_seen = time.time()
        self.hand_gestures = deque(maxlen=10)
        self.peeking_alerts = 0
        self.last_peeking_alert = 0
        self.seat_zone = None
        self.current_position = None
        self.exchange_alerts = 0
        self.last_exchange_alert = 0
        self.mobile_watch_alerts = 0
        self.last_mobile_alert = 0
        self.head_direction_history = deque(maxlen=30)
        self.head_movement_alerts = 0
        self.last_head_movement_alert = 0
        
    def update_position(self, x, y):
        self.positions.append((x, y))
        self.last_seen = time.time()
        self.total_frames += 1
        self.current_position = (x, y)
        
    def is_moving(self):
        if len(self.positions) < 10:
            return False
        recent = list(self.positions)[-10:]
        movement = sum(np.sqrt((recent[i][0]-recent[i-1][0])**2 + 
                              (recent[i][1]-recent[i-1][1])**2) 
                      for i in range(1, len(recent)))
        return movement > 50
    
    def calculate_engagement(self):
        if self.total_frames < 10:
            return 100
        distraction = (self.looking_away_frames / self.total_frames) * 100
        self.engagement_score = max(0, 100 - distraction)
        return self.engagement_score

# ============================================
# HELPER FUNCTIONS
# ============================================

def save_facenet_database():
    with open(FACENET_DB, 'wb') as f:
        pickle.dump(st.session_state.student_database_facenet, f)

def detect_faces_opencv(frame):
    """Detect faces using OpenCV DNN"""
    if face_detector is None:
        return []
    
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        face_detector.setInput(blob)
        detections = face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    faces.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': float(confidence)
                    })
        
        return faces
    except Exception as e:
        return []

def get_face_embedding(face_tensor):
    """Generate 512-dimensional embedding"""
    if resnet is None:
        return None
    
    try:
        with torch.no_grad():
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)
            elif face_tensor.dim() == 5:
                face_tensor = face_tensor.squeeze()
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
            
            embedding = resnet(face_tensor.to(device))
            embedding = embedding.cpu().numpy().flatten()
            return embedding
    except Exception as e:
        return None

def register_student_facenet(frame, student_id, student_name):
    """Register student using OpenCV + FaceNet"""
    try:
        faces = detect_faces_opencv(frame)
        
        if not faces:
            return False, None, "No face detected"
        
        best_face = max(faces, key=lambda x: x['confidence'])
        x1, y1, x2, y2 = best_face['box']
        confidence = best_face['confidence']
        
        if confidence < 0.9:
            return False, None, f"Low confidence: {confidence*100:.1f}%"
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return False, None, "Invalid face crop"
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        face_tensor = torch.from_numpy(face_resized).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.permute(2, 0, 1)
        
        embedding = get_face_embedding(face_tensor)
        
        if embedding is not None:
            if student_id not in st.session_state.student_database_facenet:
                st.session_state.student_database_facenet[student_id] = {
                    'name': student_name,
                    'embeddings': []
                }
            
            st.session_state.student_database_facenet[student_id]['embeddings'].append(embedding)
            save_facenet_database()
            
            return True, (x1, y1, x2, y2), f"Captured (conf: {confidence*100:.1f}%)"
        else:
            return False, None, "Embedding failed"
    
    except Exception as e:
        return False, None, f"Error: {str(e)}"

def recognize_student_facenet(frame, box):
    """Recognize student using cosine similarity"""
    if not st.session_state.student_database_facenet or resnet is None:
        return None, 0.0
    
    try:
        x1, y1, x2, y2 = box
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None, 0.0
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        face_tensor = torch.from_numpy(face_resized).float()
        face_tensor = (face_tensor - 127.5) / 128.0
        face_tensor = face_tensor.permute(2, 0, 1)
        
        embedding = get_face_embedding(face_tensor)
        
        if embedding is not None:
            best_match_id = None
            best_similarity = 0.0
            threshold = 0.6
            
            for student_id, data in st.session_state.student_database_facenet.items():
                for stored_embedding in data['embeddings']:
                    similarity = 1 - cosine(embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity > threshold:
                        best_similarity = similarity
                        best_match_id = student_id
            
            if best_match_id:
                return best_match_id, best_similarity * 100
    
    except Exception as e:
        pass
    
    return None, 0.0

def determine_seat_zone(x, y, frame_width):
    """Determine classroom zone"""
    third = frame_width / 3
    
    if x < third:
        return 'left'
    elif x < third * 2:
        return 'center'
    else:
        return 'right'

def export_cheating_report():
    """Export cheating events to CSV"""
    if not st.session_state.cheating_events:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"cheating_report_{timestamp}.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Event Type', 'Student ID', 'Student Name', 'Details', 'Severity'])
        
        for event in st.session_state.cheating_events:
            writer.writerow([
                event.get('timestamp', ''),
                event.get('type', ''),
                event.get('student_id', ''),
                event.get('student_name', ''),
                event.get('details', ''),
                event.get('severity', '')
            ])
    
    return csv_path

# Initialize detectors
if st.session_state.peeking_detector is None:
    st.session_state.peeking_detector = PeekingDetector(window_size=90, threshold=0.5)

if st.session_state.paper_detector is None:
    st.session_state.paper_detector = PaperExchangeDetector(time_threshold=0.5)

if st.session_state.mobile_watch_detector is None and custom_model_loaded:
    st.session_state.mobile_watch_detector = MobileWatchDetector(custom_model)

# ============================================
# VIDEO PROCESSING FUNCTION
# ============================================

def process_video_with_all_detections(video_path, output_path, progress_bar, status_text):
    """Process video with ALL detections: Face, Head, Paper Exchange, Mobile/Watch"""
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video file!")
        return None
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"📹 Video: {width}x{height} @ {fps:.2f} FPS | Total: {total_frames} frames")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Reset detectors
    paper_detector = PaperExchangeDetector(time_threshold=0.5)
    peeking_detector = PeekingDetector(window_size=90, threshold=0.5)
    mobile_detector = MobileWatchDetector(custom_model) if custom_model_loaded else None
    
    # Tracking variables
    student_data = {}
    frame_num = 0
    total_paper_exchanges = 0
    total_head_alerts = 0
    total_peeking_alerts = 0
    total_mobile_alerts = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        current_time_sec = frame_num / fps
        
        # Update progress
        progress = frame_num / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing: Frame {frame_num}/{total_frames} ({progress*100:.1f}%)")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Counters for this frame
        head_movement_count = 0
        peeking_count = 0
        mobile_watch_count = 0
        paper_exchange_count = 0
        
        # ===== 1. FACE DETECTION & RECOGNITION =====
        faces = detect_faces_opencv(frame)
        recognized_faces = []
        
        for face in faces:
            x1, y1, x2, y2 = face['box']
            student_id, confidence = recognize_student_facenet(frame, face['box'])
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            if student_id:
                student_name = st.session_state.student_database_facenet[student_id]['name']
                is_unknown = False
            else:
                student_id = f"Unknown-{int(center_x)}"
                student_name = student_id
                confidence = 0
                is_unknown = True
            
            recognized_faces.append({
                'box': (x1, y1, x2, y2),
                'student_id': student_id,
                'student_name': student_name,
                'confidence': confidence,
                'is_unknown': is_unknown
            })
            
            # Color coding
            if is_unknown:
                color = (128, 128, 128)
            elif confidence > 85:
                color = (0, 255, 0)
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{student_name}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Initialize student
            if student_id not in student_data:
                student_data[student_id] = Student(student_id, student_name)
            
            student = student_data[student_id]
            student.update_position(center_x, center_y)
        
        # ===== 2. HEAD DIRECTION & PEEKING =====
        face_results = face_mesh.process(rgb_frame)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w, _ = frame.shape
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                mp_x1, mp_y1 = int(min(x_coords)), int(min(y_coords))
                mp_x2, mp_y2 = int(max(x_coords)), int(max(y_coords))
                
                matched_student_id = None
                
                for face_info in recognized_faces:
                    fx1, fy1, fx2, fy2 = face_info['box']
                    
                    overlap_x = max(0, min(mp_x2, fx2) - max(mp_x1, fx1))
                    overlap_y = max(0, min(mp_y2, fy2) - max(mp_y1, fy1))
                    
                    if overlap_x > 50 and overlap_y > 50:
                        matched_student_id = face_info['student_id']
                        
                        # Head direction
                        head_direction = detect_head_direction_detailed(face_landmarks, width)
                        peeking_detector.update(matched_student_id, head_direction)
                        
                        peeking_info = peeking_detector.get_peeking_score(matched_student_id)
                        
                        fx1, fy1, fx2, fy2 = face_info['box']
                        dir_color = (0, 255, 0) if head_direction == 'CENTER' else (0, 165, 255)
                        cv2.putText(frame, f"Head: {head_direction}", (fx1, fy2+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 2)
                        
                        # Instant LEFT/RIGHT alert
                        if head_direction in ['LEFT', 'RIGHT']:
                            student = student_data[matched_student_id]
                            
                            if current_time_sec - student.last_head_movement_alert > 2:
                                head_movement_count += 1
                                total_head_alerts += 1
                                
                                cv2.putText(frame, f"⚠️ LOOKING {head_direction}!", 
                                          (fx1, fy2+45),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                cv2.rectangle(frame, (fx1-3, fy1-3), (fx2+3, fy2+3), (0, 165, 255), 2)
                                
                                student.head_movement_alerts += 1
                                student.last_head_movement_alert = current_time_sec
                        
                        # Persistent peeking
                        if peeking_info['is_peeking']:
                            peeking_count += 1
                            total_peeking_alerts += 1
                            
                            cv2.putText(frame, f"🚨 PEEKING {peeking_info['direction']}!", 
                                      (fx1, fy2+70),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        break
        
        # ===== 3. PAPER EXCHANGE DETECTION =====
        if paper_model_loaded:
            person_results = person_model.track(frame, persist=True, verbose=False, conf=0.5)
            paper_results = paper_model.track(frame, persist=True, verbose=False, conf=0.5)
            
            if len(person_results) > 0 and len(paper_results) > 0:
                exchanges = paper_detector.process_frame(
                    paper_results[0], person_results[0], current_time_sec
                )
                
                for exchange in exchanges:
                    paper_exchange_count += 1
                    total_paper_exchanges += 1
                    
                    px, py = exchange['position']
                    cv2.circle(frame, (px, py), 40, (0, 140, 255), 4)
                    cv2.putText(frame, "🚨 PAPER EXCHANGE!", (px - 100, py - 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    cv2.putText(frame, f"Person {exchange['old_owner']} → {exchange['new_owner']}", 
                              (px - 100, py - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # ===== 4. MOBILE/WATCH DETECTION =====
        if mobile_detector:
            detections = mobile_detector.detect(frame, 0.5)
            
            for det in detections:
                mobile_watch_count += 1
                total_mobile_alerts += 1
                
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"🚨 {det['class']} {det['confidence']:.2f}", 
                          (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "PROHIBITED DEVICE!", 
                          (x1, y2+25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # ===== 5. OVERLAY STATS =====
        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"Time: {int(current_time_sec//60)}:{int(current_time_sec%60):02d}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Paper Exchanges: {total_paper_exchanges}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        cv2.putText(frame, f"Head Alerts: {total_head_alerts} | Peeking: {total_peeking_alerts} | Devices: {total_mobile_alerts}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Current frame alerts
        if head_movement_count > 0:
            cv2.putText(frame, f"⚠️ HEAD MOVEMENT: {head_movement_count}", (10, height - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if peeking_count > 0:
            cv2.putText(frame, f"🚨 PEEKING: {peeking_count}", (10, height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mobile_watch_count > 0:
            cv2.putText(frame, f"🚨 DEVICES: {mobile_watch_count}", (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write frame
        out.write(frame)
    
    cap.release()
    out.release()
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Processing complete! {frame_num} frames processed")
    
    return {
        'total_frames': frame_num,
        'paper_exchanges': total_paper_exchanges,
        'head_alerts': total_head_alerts,
        'peeking_alerts': total_peeking_alerts,
        'mobile_alerts': total_mobile_alerts,
        'students_detected': len(student_data)
    }

# ============================================
# SIDEBAR CONTROLS
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # Model status
    with st.expander("🤖 Model Status", expanded=False):
        col1, col2 = st.columns(2)
        col1.metric("FaceNet", "✅" if resnet else "❌")
        col2.metric("Face Detector", "✅" if face_detector else "❌")
        col1.metric("YOLO Pose", "✅")
        col2.metric("Paper Model", "✅" if paper_model_loaded else "❌")
        col1.metric("Mobile/Watch", "✅" if custom_model_loaded else "❌")
        st.caption(f"Device: {'🖥️ GPU (CUDA)' if torch.cuda.is_available() else '💻 CPU'}")
    
    st.markdown("---")
    
    # Mode selection
    mode = st.radio(
        "🎯 Select Mode",
        ["📷 Register Students", "🎥 Live Monitoring", "📹 Video Analysis"],
        help="Choose your operation mode"
    )
    
    if mode == "📹 Video Analysis":
        st.markdown("---")
        st.markdown("### 📹 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload exam footage for complete analysis"
        )
        
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            st.session_state.uploaded_video_path = tfile.name
            st.success(f"✅ {uploaded_file.name} uploaded!")
        
        if st.session_state.uploaded_video_path:
            if st.button("▶️ Start Full Analysis", type="primary", use_container_width=True):
                st.session_state.video_processing = True
                st.rerun()

# ============================================
# MAIN DISPLAY AREA
# ============================================

if mode == "📹 Video Analysis" and st.session_state.video_processing:
    st.markdown("### 🎬 Video Processing in Progress...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Output path
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_output.mp4').name
    
    # Process video
    results = process_video_with_all_detections(
        st.session_state.uploaded_video_path,
        output_path,
        progress_bar,
        status_text
    )
    
    if results:
        st.success("✅ Video processing complete!")
        
        # Display results
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📄 Paper Exchanges", results['paper_exchanges'])
        col2.metric("↔️ Head Alerts", results['head_alerts'])
        col3.metric("👀 Peeking", results['peeking_alerts'])
        col4.metric("📱 Devices", results['mobile_alerts'])
        col5.metric("👥 Students", results['students_detected'])
        
        # Download button
        with open(output_path, 'rb') as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name="cheating_detection_output.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        
        # Video player
        st.video(output_path)
        
        st.session_state.video_processing = False
        
        if st.button("🔄 Process Another Video", use_container_width=True):
            st.session_state.uploaded_video_path = None
            st.rerun()

elif mode == "📹 Video Analysis":
    st.info("📹 Upload a video file to begin comprehensive analysis")
    st.markdown("""
    ### 🎯 Features Included:
    - ✅ **Face Recognition** - Identify all students
    - ✅ **Head Movement Detection** - Left/Right alerts
    - ✅ **Persistent Peeking** - Continuous monitoring
    - ✅ **Paper Exchange Detection** - Track paper movements
    - ✅ **Mobile/Watch Detection** - Prohibited devices
    - ✅ **Complete Statistics** - Full report overlay
    """)

else:
    st.info("🎥 Select 'Video Analysis' mode from sidebar to process exam videos")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem;'>
    <p style='font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;'>🎓 Advanced Cheating Detection System</p>
    <p style='font-size: 0.95rem; margin-bottom: 0.3rem;'>
        <strong>Powered by:</strong> FaceNet • YOLOv8 • MediaPipe • OpenCV DNN • Custom Models
    </p>
    <p style='font-size: 0.85rem; color: #888;'>
        <strong>Features:</strong> Face Recognition • Head Movement • Peeking • Paper Exchange • Device Detection
    </p>
</div>
""", unsafe_allow_html=True)