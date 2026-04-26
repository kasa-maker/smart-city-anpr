import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime
import cv2
import tempfile
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import easyocr
import re

st.set_page_config(page_title="Smart City Traffic Monitor", page_icon="🚦", layout="wide")
st.title("🚦 Smart City AI Traffic Monitoring System")
st.markdown("**Real-time Vehicle Detection & License Plate Recognition**")

DB_PATH = r'C:\Users\LENOVO\Desktop\smart_city_traffic\smart-traffic-anpr\database\traffic.db'

def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
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
        df = pd.read_sql_query("SELECT * FROM vehicle_logs", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def process_video(video_path, output_path):
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30)
    reader = easyocr.Reader(['en'], gpu=False)

    VEHICLE_CLASSES = {2: 'car', 3: 'bike', 5: 'bus', 7: 'truck'}

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    LINE_Y = int(height * 0.75)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    crossed_ids = set()
    final_logs = []
    frame_count = 0
    vehicle_count = {'car': 0, 'bike': 0, 'bus': 0, 'truck': 0}
    plate_assigned = {}

    conn = sqlite3.connect(DB_PATH)
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
            plate_y1 = y1 + int((y2 - y1) * 0.4)
            crop = frame[max(0, plate_y1):y2, max(0, x1):x2]
            if crop.size == 0:
                return 'UNREADABLE'
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            results = reader.readtext(thresh)
            if results:
                best = max(results, key=lambda x: x[2])
                text = best[1].upper().strip()
                conf = best[2]
                text = re.sub(r'[^A-Z0-9\-]', '', text)
                if conf > 0.15 and len(text) >= 2:
                    return text
            return 'UNREADABLE'
        except:
            return 'UNREADABLE'

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    status = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        progress.progress(min(frame_count / total_frames, 1.0))
        status.text(f"Processing frame {frame_count}/{total_frames} - Vehicles crossed: {len(crossed_ids)}")

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, 'SMART CITY ANPR', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Total Crossed : {len(crossed_ids)}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Cars          : {vehicle_count["car"]}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Bikes         : {vehicle_count["bike"]}', (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Buses         : {vehicle_count["bus"]}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Trucks        : {vehicle_count["truck"]}', (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, ts, (width - 220, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.line(frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 255), 2)
        cv2.putText(frame, 'DETECTION LINE', (10, LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
            plate_text = f'PLATE: {plate}'
            (tw, th), _ = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-16), (x1+tw+10, y1), (0, 255, 0), -1)
            cv2.putText(frame, plate_text, (x1+5, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.rectangle(frame, (x1, y2), (x1+130, y2+22), (0, 255, 0), -1)
            cv2.putText(frame, f'ID:{track_id} {vtype.upper()}', (x1+3, y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if centroid_y > LINE_Y and track_id not in crossed_ids:
                crossed_ids.add(track_id)
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

        out.write(frame)

    cap.release()
    out.release()
    conn.close()

    df = pd.DataFrame(final_logs)
    df.to_csv(r'C:\Users\LENOVO\Desktop\smart_city_traffic\smart-traffic-anpr\output\final_logs.csv', index=False)
    return output_path, len(crossed_ids)

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🎥 Upload & Detect"])

with tab1:
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Vehicles", len(df))
    with col2:
        cars = len(df[df['vehicle_type'] == 'car']) if not df.empty else 0
        st.metric("Cars", cars)
    with col3:
        buses = len(df[df['vehicle_type'] == 'bus']) if not df.empty else 0
        st.metric("Buses", buses)
    with col4:
        plates = len(df[df['plate_number'] != 'UNREADABLE']) if not df.empty else 0
        st.metric("Plates Read", plates)

    st.divider()

    if not df.empty:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Vehicle Type Distribution")
            fig = px.pie(df, names='vehicle_type', title='Vehicles by Type')
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.subheader("Detection Timeline")
            fig2 = px.bar(df, x='timestamp', color='vehicle_type', title='Vehicles Over Time')
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("License Plate Logs")
        st.dataframe(df[['vehicle_id', 'vehicle_type', 'plate_number', 'timestamp']], use_container_width=True, hide_index=True)
    else:
        st.warning("No data found in database")

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with tab2:
    st.subheader("Upload Video for Detection")
    uploaded_file = st.file_uploader("Traffic video upload karo", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        output_path = r'C:\Users\LENOVO\Desktop\smart_city_traffic\smart-traffic-anpr\output\output_dashboard.mp4'

        if st.button("Start Detection"):
            st.info("Processing shuru ho rahi hai - thoda wait karo...")
            out_path, total = process_video(tfile.name, output_path)
            st.success(f"Done! Total {total} vehicles detected!")

            st.subheader("Output Video")
            st.video(out_path)

            with open(out_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="detection_output.mp4",
                mime="video/mp4"
            )

            st.subheader("Updated Results")
            df_new = load_data()
            st.dataframe(df_new[['vehicle_id', 'vehicle_type', 'plate_number', 'timestamp']], use_container_width=True, hide_index=True)