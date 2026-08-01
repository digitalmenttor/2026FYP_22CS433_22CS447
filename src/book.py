"""
Paper + Book Detection Test — yolov8n.pt + best.pt
Sirf testing ke liye
"""

import cv2
from ultralytics import YOLO

book_model  = YOLO("yolov8n.pt")   # COCO class 73 = 'book'
paper_model = YOLO("best.pt")       # sirf 'paper' class

print("Book model classes:", {k: v for k, v in book_model.names.items() if 'book' in v.lower()})
print("Paper model classes:", paper_model.names)

cap = cv2.VideoCapture(0)
print("\nCamera khul gayi — 'q' dabao band karne ke liye\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ── PAPER (best.pt) ──────────────────────────────────────────
    paper_results = paper_model(frame, conf=0.30, verbose=False)
    for r in paper_results:
        for box in r.boxes:
            cls_name = paper_model.names[int(box.cls[0])]
            if cls_name.lower() != 'paper':
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # Orange box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
            cv2.putText(frame, f"PAPER {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            print(f"📄 PAPER detected — conf: {conf:.0%}  box: ({x1},{y1},{x2},{y2})")

    # ── BOOK (yolov8n.pt) ────────────────────────────────────────
    book_results = book_model(frame, conf=0.30, verbose=False)
    for r in book_results:
        for box in r.boxes:
            cls_name = book_model.names[int(box.cls[0])]
            if cls_name.lower() != 'book':
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            # Blue/brown box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 3)
            cv2.putText(frame, f"BOOK {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)
            print(f"📚 BOOK detected  — conf: {conf:.0%}  box: ({x1},{y1},{x2},{y2})")

    # ── LEGEND ───────────────────────────────────────────────────
    cv2.putText(frame, "ORANGE = Paper (best.pt)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, "BLUE = Book (yolov8n.pt)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

    cv2.imshow("Paper + Book Detection Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()