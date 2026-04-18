from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import sqlite3
from datetime import datetime
import easyocr

model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)
reader = easyocr.Reader(['en'], gpu=False)

VEHICLE_CLASSES = {2: 'car', 3: 'bike', 5: 'bus', 7: 'truck'}

VIDEO_PATH = 'data/videos/sample_video2.mp4'
OUTPUT_PATH = 'output/output_real_plates.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
LINE_Y = height // 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

crossed_ids = set()
final_logs = []
frame_count = 0
vehicle_count = {'car': 0, 'bike': 0, 'bus': 0, 'truck': 0}
plate_assigned = {}

conn = sqlite3.connect('database/traffic.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vehicle_id INTEGER,
        vehicle_type TEXT,
        plate_number TEXT,
        timestamp TEXT
    )
''')
conn.commit()

def read_plate(frame, x1, y1, x2, y2):
    try:
        # Plate area neeche wale hisse mein hoti hai gaadi ke
        plate_y1 = y1 + int((y2 - y1) * 0.6)
        crop = frame[max(0, plate_y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return 'UNREADABLE'

        # Image sharpen karo
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = reader.readtext(thresh)
        if results:
            best = max(results, key=lambda x: x[2])
            text = best[1].upper().strip()
            conf = best[2]
            # Clean text - sirf letters aur numbers rakho
            text = re.sub(r'[^A-Z0-9\-]', '', text)
            if conf > 0.25 and len(text) >= 3:
                return text
        return 'UNREADABLE'
    except:
        return 'UNREADABLE'

print("Real OCR processing shuru...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    # Info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, 'SMART CITY ANPR', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Total Crossed : {len(crossed_ids)}', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Cars          : {vehicle_count["car"]}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Bikes         : {vehicle_count["bike"]}', (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Buses         : {vehicle_count["bus"]}', (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Trucks        : {vehicle_count["truck"]}', (10, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, ts, (width - 220, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 255), 2)
    cv2.putText(frame, 'DETECTION LINE', (10, LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    results = model(frame, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                conf = float(box.conf[0])
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2-x1, y2-y1], conf, VEHICLE_CLASSES[cls_id]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        centroid_y = (y1 + y2) // 2

        vtype = 'car'
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASSES:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    if abs(bx1 - x1) < 20:
                        vtype = VEHICLE_CLASSES[cls_id]

        if track_id not in plate_assigned:
            plate_assigned[track_id] = 'SCANNING...'

        plate = plate_assigned[track_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{vtype.upper()} ID:{track_id}', (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y2), (x1 + 120, y2 + 20), (0, 0, 0), -1)
        cv2.putText(frame, plate, (x1 + 2, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if centroid_y > LINE_Y and track_id not in crossed_ids:
            crossed_ids.add(track_id)

            print(f"Vehicle {track_id} crossed - OCR chal raha hai...")
            real_plate = read_plate(frame, x1, y1, x2, y2)
            plate_assigned[track_id] = real_plate

            if vtype in vehicle_count:
                vehicle_count[vtype] += 1

            cursor.execute('''
                INSERT INTO vehicle_logs (vehicle_id, vehicle_type, plate_number, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (track_id, vtype, real_plate, datetime.now().strftime('%H:%M:%S')))
            conn.commit()

            final_logs.append({
                'vehicle_id': track_id,
                'vehicle_type': vtype,
                'plate_number': real_plate,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            print(f"Plate: {real_plate}")

    out.write(frame)

cap.release()
out.release()
conn.close()

df = pd.DataFrame(final_logs)
df.to_csv('output/final_logs.csv', index=False)

print(f"\nDone!")
print(f"Total vehicles crossed: {len(crossed_ids)}")
print(df)
