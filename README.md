# IntelliExam – Invigilator Free Classroom Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.x-black)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![FaceNet](https://img.shields.io/badge/FaceNet-Recognition-orange)
![MySQL](https://img.shields.io/badge/MySQL-8.0-blue)
![License](https://img.shields.io/badge/License-Educational-success)

**AI-Powered Intelligent Examination Monitoring System**

Bachelor of Science in Computer Science Final Year Project

University of Engineering and Technology (UET) Lahore  
Rachna College of Engineering and Technology (RCET), Gujranwala

</div>

---

# Overview

IntelliExam is an AI-powered examination monitoring system developed to improve academic integrity during physical classroom examinations.

The system combines multiple Artificial Intelligence and Computer Vision technologies to automatically identify students, monitor their behaviour, detect suspicious activities, calculate cheating risk scores, and notify instructors in real time.

Unlike conventional examination monitoring systems that rely heavily on manual invigilation or CCTV recordings, IntelliExam performs continuous automated monitoring using live video streams and generates evidence-based alerts whenever suspicious behaviour is detected. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

# Key Features

## Face Recognition

- Automatic student identification
- FaceNet (InceptionResnetV1)
- Attendance marking
- Unknown person detection

---

## Student Behaviour Analysis

- Eye gaze tracking
- Head pose estimation
- Peeking detection
- Continuous behaviour monitoring

Implemented using MediaPipe Face Mesh.

---

## Object Detection

YOLOv8 detects:

- Mobile Phones
- Books
- Notes
- Paper Exchange

The paper exchange detector is trained on a custom dataset developed specifically for this project because no public dataset provides this class. :contentReference[oaicite:2]{index=2}

---

## Intelligent Risk Scoring

The system combines multiple suspicious events into a single dynamic risk score using an Exponential Moving Average (EMA) based approach.

Risk levels:

- 🟢 Low
- 🟡 Medium
- 🔴 High

This prevents false alarms caused by isolated detections.

---

## WhatsApp Alerts

When the predefined risk threshold is exceeded, IntelliExam automatically sends:

- Student name
- Event type
- Timestamp
- Snapshot evidence

through the Twilio WhatsApp API. :contentReference[oaicite:3]{index=3}

---

## Dashboard

Real-time dashboard includes:

- Live camera feed
- Student attendance
- Risk scores
- Detection logs
- Activity timeline
- Session statistics

---

# Technology Stack

| Layer | Technology |
|---------|------------|
| Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Backend | Flask |
| Language | Python 3.10 |
| Database | MySQL 8 |
| Face Recognition | FaceNet |
| Eye Tracking | MediaPipe Face Mesh |
| Object Detection | YOLOv8 |
| Paper Exchange Detection | Custom YOLOv8 |
| Alert Service | Twilio WhatsApp API |
| Development | VS Code |
| Dataset Annotation | Roboflow, LabelImg |
| Training | Google Colab |

Technology stack follows the implementation described in Chapter 5 of the thesis. :contentReference[oaicite:4]{index=4}

---

# System Architecture

```
Camera
      │
      ▼
Video Capture
      │
      ▼
────────────────────────────────────
Face Recognition (FaceNet)
────────────────────────────────────
      │
      ▼
Attendance Management
      │
      ▼
────────────────────────────────────
MediaPipe Face Mesh
────────────────────────────────────
      │
      ▼
Eye Tracking + Head Pose
      │
      ▼
────────────────────────────────────
YOLOv8 Detection
────────────────────────────────────
      │
      ▼
Phone
Book
Notes
Paper Exchange
      │
      ▼
Risk Scoring Engine
      │
      ▼
Database
      │
      ▼
Dashboard
      │
      ▼
WhatsApp Alerts
```

---

# Project Modules

## Student Registration

- Face capture
- Embedding generation
- Database storage

---

## Attendance Module

- Automatic attendance
- Unknown face detection

---

## Monitoring Module

- Live classroom monitoring
- Continuous frame analysis

---

## Behaviour Analysis

- Eye gaze estimation
- Head pose estimation
- Peeking detection

---

## Object Detection

Detects:

- Mobile phone
- Book
- Notebook
- Paper exchange

---

## Risk Assessment

Combines:

- Mobile phone usage
- Head movement
- Eye movement
- Paper exchange
- Book detection

into a unified cheating probability.

---

## Notification System

Real-time WhatsApp alerts with image evidence.

---

# REST API

| Endpoint | Method | Description |
|------------|---------|-------------|
| /api/monitoring/start | POST | Start monitoring |
| /api/monitoring/stop | POST | Stop monitoring |
| /api/video_feed | GET | Live camera feed |
| /api/stats | GET | Session statistics |
| /api/risk_scores | GET | Student risk scores |
| /api/activity_log | GET | Detection history |
| /api/export_report | GET | Export CSV report |
| /api/classrooms | GET/POST | Classroom management |

Based on the Flask API described in the thesis. :contentReference[oaicite:5]{index=5}

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/IntelliExam.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Configure environment variables

```
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=

TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

NGROK_PUBLIC_URL=
```

Run

```bash
python app.py
```

---

# Hardware Requirements

- Intel Core i7 (8th Gen)
- 16 GB RAM
- NVIDIA GTX 1650
- Webcam
- Windows 11

The evaluation environment described in the thesis used this configuration. :contentReference[oaicite:6]{index=6}

---

# Performance Highlights

- Face Recognition Accuracy: **96.2%**
- Paper Exchange Detection mAP50: **98.3%**
- Calculator Detection mAP50: **93.1%**
- Real-time WhatsApp Alerts
- Automatic Attendance
- Evidence-based Risk Assessment

These values are reported in the experimental evaluation chapter. :contentReference[oaicite:7]{index=7}

---

# Future Improvements

- Multi-camera support
- Cloud deployment
- Online examination monitoring
- Audio-based cheating detection
- Student tracking
- Predictive analytics
- Enhanced dashboard
- Expanded paper exchange dataset

Future work follows the recommendations presented in Chapter 7. :contentReference[oaicite:8]{index=8}

---

# 📂 Project Structure

```
IntelliExam
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .env
│
├── src
│   ├── detection
│   ├── tracking
│   ├── notifications
│   ├── database
│   ├── middleware
│   ├── services
│   └── utils
│
├── models
│
├── templates
│
├── static
│
├── videos
│
├── logs
│
└── docs
```

---

# 🚀 How IntelliExam Works

```
Camera Feed
      │
      ▼
Face Recognition
      │
      ▼
Attendance Management
      │
      ▼
Head Pose Estimation
      │
      ▼
Eye Gaze Tracking
      │
      ▼
YOLOv8 Object Detection
      │
      ▼
Risk Score Calculation
      │
      ▼
Dashboard Update
      │
      ▼
WhatsApp Notification
```

---

# 📊 Performance

| Model | Accuracy |
|---------|-----------|
| Face Recognition | 96.2% |
| Paper Exchange Detection | 97.1% |
| Head Pose Estimation | 92.8% |
| Eye Tracking | 90.4% |
| Calculator Detection | 87.7% |
| Smart Watch Detection | 83.1% |

---

# ⚡ Main Functionalities

- Automatic Student Attendance
- Face Recognition
- Face Verification
- Unknown Person Detection
- Eye Tracking
- Head Pose Tracking
- Peeking Detection
- Mobile Phone Detection
- Calculator Detection
- Smart Watch Detection
- Book Detection
- Paper Exchange Detection
- Dynamic Risk Score
- WhatsApp Alert System
- Live Monitoring Dashboard
- Detection History
- CSV Report Export

---

# 📦 Python Libraries

```
Flask
OpenCV
Ultralytics
MediaPipe
TensorFlow
FaceNet
NumPy
Pandas
MySQL Connector
Twilio
Flask-SocketIO
```

Install using

```bash
pip install -r requirements.txt
```

---

# ⚙ Environment Variables

Create a `.env` file.

```env
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=

TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

NGROK_PUBLIC_URL=
```

---

# 📈 Future Improvements

- Multi-camera support
- AI-powered exam analytics
- Cloud deployment
- Student re-identification
- Voice activity detection
- Mobile application
- Admin dashboard
- Online examination support
- AI-generated examination reports

---

# 🎥 Video Demonstration

Full Project Video


> **[https://drive.google.com/file/d/1O-oC9vT70ZWYM6Q8nS2OUdNLA3wxcjTH/view?usp=sharing]**

---

# 🤝 Contributors

Muhammad Usman

Hafiz Syed Habib Ahmad Gillani

BS Computer Science

University of Engineering and Technology (RCET)

---



# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.


# Authors

**Muhammad Usman**  
BS Computer Science  
RCET, UET Lahore

**Hafiz Syed Habib Ahmad Gillani**  
BS Computer Science  
RCET, UET Lahore

---

# Supervisor

**Ma’am Amna Wajid**

Co-Supervisor

**Ma’am Namra Ashraf**

---

# License

This project was developed for academic and educational purposes as a Final Year Project at the Department of Computer Science, RCET, UET Lahore.
