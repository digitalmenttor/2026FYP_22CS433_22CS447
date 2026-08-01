import cv2
import sqlite3
import numpy as np
import pickle
import random
import os
import time

# ── DB SETUP ───────────────────────────────────────────────────────────────────
DB = "classroom.db"

def get_conn():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = get_conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT NOT NULL,
            roll_number    TEXT NOT NULL UNIQUE,
            department     TEXT,
            face_embedding BLOB
        );
        CREATE TABLE IF NOT EXISTS seats (
            seat_id TEXT PRIMARY KEY,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
        );
        CREATE TABLE IF NOT EXISTS seat_assignments (
            student_id INTEGER,
            seat_id    TEXT,
            PRIMARY KEY (student_id, seat_id)
        );
        CREATE TABLE IF NOT EXISTS violations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id    INTEGER,
            assigned_seat TEXT,
            current_seat  TEXT,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    c.commit()
    c.close()

def db_save_seats(seat_list):
    c = get_conn()
    c.execute("DELETE FROM seats")
    c.executemany("INSERT INTO seats VALUES (?,?,?,?,?)", seat_list)
    c.commit(); c.close()

def db_get_seats():
    c = get_conn()
    rows = c.execute("SELECT * FROM seats ORDER BY seat_id").fetchall()
    c.close(); return rows

def db_save_student(name, roll, dept, emb_bytes):
    try:
        c = get_conn()
        c.execute("INSERT INTO students (name,roll_number,department,face_embedding) VALUES (?,?,?,?)",
                  (name, roll, dept, emb_bytes))
        c.commit(); c.close()
        return True, "Registered!"
    except sqlite3.IntegrityError:
        return False, "Roll number already exists."

def db_get_students():
    c = get_conn()
    rows = c.execute("SELECT * FROM students").fetchall()
    c.close(); return rows

def db_save_assignments(pairs):
    c = get_conn()
    c.execute("DELETE FROM seat_assignments")
    c.executemany("INSERT OR REPLACE INTO seat_assignments VALUES (?,?)", pairs)
    c.commit(); c.close()

def db_get_assignments():
    c = get_conn()
    rows = c.execute("""
        SELECT s.name, s.roll_number, sa.seat_id
        FROM students s JOIN seat_assignments sa ON s.id=sa.student_id
    """).fetchall()
    c.close(); return rows

def db_get_assigned_seat(student_id):
    c = get_conn()
    row = c.execute("SELECT seat_id FROM seat_assignments WHERE student_id=?",
                    (student_id,)).fetchone()
    c.close()
    return row["seat_id"] if row else None

def db_save_violation(student_id, assigned_seat, current_seat):
    c = get_conn()
    c.execute("INSERT INTO violations (student_id,assigned_seat,current_seat) VALUES (?,?,?)",
              (student_id, assigned_seat, current_seat))
    c.commit(); c.close()

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print("[INFO] Loading YOLOv8...")
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")

print("[INFO] Loading FaceNet...")
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn  = MTCNN(keep_all=True, device=device, min_face_size=30)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
print(f"[INFO] Models loaded on {device}")

# ── HELPER: DRAW TEXT WITH BACKGROUND ─────────────────────────────────────────
def put_text_bg(frame, text, pos, font_scale=0.6, thickness=2,
                txt_color=(255,255,255), bg_color=(0,0,0)):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - bl - 2), (x + tw + 4, y + bl), bg_color, -1)
    cv2.putText(frame, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, txt_color, thickness)

# ── HELPER: DETECT CHAIRS ─────────────────────────────────────────────────────
def detect_chairs(frame):
    results = yolo(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        if int(box.cls[0]) == 56:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            boxes.append((x1,y1,x2,y2,conf))
    boxes.sort(key=lambda b: b[0])
    return boxes

# ── HELPER: EXTRACT FACE EMBEDDING ────────────────────────────────────────────
def extract_embedding(frame_rgb):
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    faces = mtcnn(pil)
    if faces is None:
        return None
    face = faces[0].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = resnet(face).cpu().numpy()[0]
    return emb

# ── HELPER: IDENTIFY FACES IN FRAME ───────────────────────────────────────────
def identify_faces(frame_rgb, threshold=0.52):
    from PIL import Image
    pil = Image.fromarray(frame_rgb)
    boxes, _ = mtcnn.detect(pil)
    if boxes is None:
        return []
    faces = mtcnn(pil)
    if faces is None:
        return []
    students = db_get_students()
    results = []
    for idx, face_tensor in enumerate(faces):
        face_t = face_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = resnet(face_t).cpu().numpy()[0]
        best, best_sim = None, -1
        for stu in students:
            if stu["face_embedding"] is None:
                continue
            db_emb = pickle.loads(stu["face_embedding"])
            na = np.linalg.norm(emb); nb = np.linalg.norm(db_emb)
            sim = float(np.dot(emb, db_emb) / (na * nb)) if na and nb else 0
            if sim > best_sim:
                best_sim, best = sim, stu
        if best_sim >= threshold and best:
            results.append((best, boxes[idx], best_sim))
    return results

def find_seat_for_point(cx, cy):
    for s in db_get_seats():
        if s["x1"] <= cx <= s["x2"] and s["y1"] <= cy <= s["y2"]:
            return s["seat_id"]
    return None

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LIVE CHAIR DETECTION
# Press S to save chairs, Q to quit
# ══════════════════════════════════════════════════════════════════════════════
def step1_detect_chairs():
    print("\n" + "="*55)
    print("  STEP 1 — LIVE CHAIR DETECTION")
    print("  S = Save chairs & continue    Q = Quit")
    print("="*55)

    cap = cv2.VideoCapture(0)
    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        chairs = detect_chairs(frame)
        vis = frame.copy()

        for i, (x1,y1,x2,y2,conf) in enumerate(chairs):
            sid = f"A{i+1}"
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,230,80), 2)
            put_text_bg(vis, f"{sid} {conf:.0%}", (x1+4, y1+24),
                        bg_color=(0,180,60))

        status = f"Chairs found: {len(chairs)}   |   S=Save  Q=Quit"
        put_text_bg(vis, status, (10, 35), font_scale=0.75,
                    bg_color=(20,20,20))

        cv2.imshow("STEP 1 — Chair Detection", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and chairs:
            seat_data = [(f"A{i+1}", x1,y1,x2,y2)
                         for i,(x1,y1,x2,y2,conf) in enumerate(chairs)]
            db_save_seats(seat_data)
            print(f"[OK] Saved {len(seat_data)} seats: {[s[0] for s in seat_data]}")
            saved = True
            # Flash green confirmation
            confirm = vis.copy()
            cv2.rectangle(confirm, (0,0), (confirm.shape[1], confirm.shape[0]),
                          (0,255,0), 10)
            put_text_bg(confirm, f"SAVED {len(seat_data)} SEATS!", (10, 80),
                        font_scale=1.2, bg_color=(0,160,0))
            cv2.imshow("STEP 1 — Chair Detection", confirm)
            cv2.waitKey(1200)
            break

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STUDENT REGISTRATION (live webcam face capture)
# ══════════════════════════════════════════════════════════════════════════════
def step2_register_students():
    print("\n" + "="*55)
    print("  STEP 2 — STUDENT REGISTRATION")
    print("="*55)

    while True:
        students = db_get_students()
        seats    = db_get_seats()
        print(f"\n  Registered: {len(students)} students | Seats: {len(seats)}")
        print("  Options:")
        print("    1 = Register new student")
        print("    2 = Show registered students")
        print("    3 = Continue to Seating Plan")
        print("    Q = Quit")
        choice = input("  Choose: ").strip().lower()

        if choice == '1':
            name = input("  Name       : ").strip()
            roll = input("  Roll Number: ").strip()
            dept = input("  Department : ").strip()
            if not name or not roll or not dept:
                print("  [!] All fields required.")
                continue

            print("  [CAM] Face capture window khulega...")
            print("        SPACE = capture face   Q = cancel")

            cap = cv2.VideoCapture(0)
            captured_emb = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                vis = frame.copy()
                put_text_bg(vis, "SPACE=Capture Face   Q=Cancel", (10,35),
                            font_scale=0.7, bg_color=(40,40,40))
                put_text_bg(vis, f"Student: {name}", (10, 70),
                            font_scale=0.7, bg_color=(0,100,200))

                # Live face box preview
                from PIL import Image as PILImage
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_prev = PILImage.fromarray(rgb)
                boxes_prev, _ = mtcnn.detect(pil_prev)
                if boxes_prev is not None:
                    for b in boxes_prev:
                        bx1,by1,bx2,by2 = map(int,b)
                        cv2.rectangle(vis,(bx1,by1),(bx2,by2),(0,255,255),2)

                cv2.imshow("STEP 2 — Face Capture", vis)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    emb = extract_embedding(rgb)
                    if emb is None:
                        print("  [!] No face detected. Try again.")
                    else:
                        captured_emb = emb
                        # Flash
                        ok_frame = vis.copy()
                        cv2.rectangle(ok_frame,(0,0),(ok_frame.shape[1],ok_frame.shape[0]),(0,255,0),10)
                        put_text_bg(ok_frame,"FACE CAPTURED!",(10,80),font_scale=1.2,bg_color=(0,160,0))
                        cv2.imshow("STEP 2 — Face Capture", ok_frame)
                        cv2.waitKey(1000)
                        break
                if key == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            if captured_emb is not None:
                ok, msg = db_save_student(name, roll, dept, pickle.dumps(captured_emb))
                print(f"  [{'OK' if ok else '!!'}] {msg}")
            else:
                print("  [!] Registration cancelled.")

        elif choice == '2':
            students = db_get_students()
            if not students:
                print("  No students yet.")
            else:
                print(f"\n  {'ID':<4} {'Name':<20} {'Roll':<15} {'Dept':<20} {'Face'}")
                print("  " + "-"*65)
                for s in students:
                    face = "✓" if s["face_embedding"] else "✗"
                    print(f"  {s['id']:<4} {s['name']:<20} {s['roll_number']:<15} {s['department']:<20} {face}")

        elif choice == '3':
            students = db_get_students()
            if not students:
                print("  [!] Register at least one student first.")
            else:
                break

        elif choice == 'q':
            return False

    return True

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SEATING PLAN
# ══════════════════════════════════════════════════════════════════════════════
def step3_seating_plan():
    print("\n" + "="*55)
    print("  STEP 3 — SEATING PLAN")
    print("="*55)

    students = db_get_students()
    seats    = db_get_seats()

    print(f"  Students: {len(students)}  |  Seats: {len(seats)}")
    print("  Mode:")
    print("    1 = Sequential (A1→first student, A2→second...)")
    print("    2 = Random shuffle")
    mode = input("  Choose (1/2): ").strip()

    stu_list  = list(students)
    seat_list = [s["seat_id"] for s in seats]

    if mode == '2':
        random.shuffle(stu_list)

    pairs = []
    print("\n  SEATING PLAN:")
    print(f"  {'Student':<22} {'Roll':<15} {'Seat'}")
    print("  " + "-"*45)
    for i, stu in enumerate(stu_list):
        if i >= len(seat_list):
            break
        pairs.append((stu["id"], seat_list[i]))
        print(f"  {stu['name']:<22} {stu['roll_number']:<15} {seat_list[i]}")

    confirm = input("\n  Save this plan? (Y/n): ").strip().lower()
    if confirm != 'n':
        db_save_assignments(pairs)
        print(f"  [OK] Seating plan saved — {len(pairs)} assignments.")
        return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — LIVE MONITORING
# ══════════════════════════════════════════════════════════════════════════════
def step4_live_monitoring():
    print("\n" + "="*55)
    print("  STEP 4 — LIVE MONITORING")
    print("  Q = Quit monitoring")
    print("="*55)

    cap = cv2.VideoCapture(0)
    seats = db_get_seats()

    # Violation cooldown: avoid spamming DB (5 sec per student)
    last_violation = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis = frame.copy()

        # Draw seat boxes (faint)
        for s in seats:
            cv2.rectangle(vis, (s["x1"],s["y1"]), (s["x2"],s["y2"]),
                          (0,180,80), 1)
            put_text_bg(vis, s["seat_id"],
                        (s["x1"]+4, s["y1"]+20),
                        font_scale=0.55, bg_color=(0,120,50))

        # Identify faces
        detections = identify_faces(rgb)

        swap_map    = {}   # current_seat → name
        swap_alerts = []

        for stu, bbox, sim in detections:
            x1,y1,x2,y2 = map(int, bbox)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            name  = stu["name"]
            roll  = stu["roll_number"]

            current_seat  = find_seat_for_point(cx, cy)
            assigned_seat = db_get_assigned_seat(stu["id"])

            if current_seat is None:
                color  = (180,180,180)
                status = "Outside"
            elif assigned_seat and current_seat != assigned_seat:
                color  = (0, 0, 255)
                status = f"WRONG! Assigned:{assigned_seat}"
                # Save violation with cooldown
                now = time.time()
                if now - last_violation.get(stu["id"], 0) > 5:
                    db_save_violation(stu["id"], assigned_seat, current_seat)
                    last_violation[stu["id"]] = now
                # swap check
                if current_seat in swap_map:
                    swap_alerts.append((name, swap_map[current_seat], current_seat))
                swap_map[current_seat] = name
            else:
                color  = (0,230,80)
                status = f"OK-{current_seat}"

            # Face box
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            put_text_bg(vis, f"{name} | {status}",
                        (x1, max(y1-8, 20)),
                        font_scale=0.55, bg_color=color,
                        txt_color=(0,0,0) if color==(0,230,80) else (255,255,255))
            put_text_bg(vis, f"{sim:.0%}",
                        (x1, min(y2+18, vis.shape[0]-5)),
                        font_scale=0.45, bg_color=(50,50,50))

        # Swap alert overlay
        for i, (n1, n2, seat) in enumerate(swap_alerts):
            put_text_bg(vis, f"SWAP! {n1} <-> {n2} @ {seat}",
                        (10, 80 + i*30),
                        font_scale=0.65,
                        bg_color=(0,60,220))

        # Status bar
        put_text_bg(vis, f"Monitoring | Faces:{len(detections)} | Q=Quit",
                    (10, 35), font_scale=0.7, bg_color=(20,20,20))

        cv2.imshow("STEP 4 — Live Monitoring  |  Q=Quit", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()

    print("\n" + "█"*55)
    print("  IntelliSeat — Smart Exam Seating Verification")
    print("█"*55)
    print("\n  Menu:")
    print("    1 = Full flow (Chair→Register→Plan→Monitor)")
    print("    2 = Jump to Student Registration")
    print("    3 = Jump to Seating Plan")
    print("    4 = Jump to Live Monitoring")
    print("    Q = Quit")

    choice = input("\n  Choose: ").strip().lower()

    if choice == '1':
        ok = step1_detect_chairs()
        if not ok:
            print("[!] No chairs saved. Exiting.")
            exit()
        ok = step2_register_students()
        if not ok:
            exit()
        ok = step3_seating_plan()
        if not ok:
            exit()
        step4_live_monitoring()

    elif choice == '2':
        step2_register_students()

    elif choice == '3':
        step3_seating_plan()

    elif choice == '4':
        step4_live_monitoring()

    else:
        print("Bye!")