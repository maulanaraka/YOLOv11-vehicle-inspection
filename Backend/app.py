import sqlite3
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import os
import sys
import uuid
import json
import time
import threading
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
from contextlib import contextmanager
import eventlet
from time import time as current_time

eventlet.monkey_patch()
connected_sids = set()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    ping_interval=10, ping_timeout=60)
CORS(app)

# Configuration
class Config:
    MODEL_TRACK_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\yolo11n.pt"
    MODEL_SEGMENT_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\best-small.pt"
    OUTPUT_DIR = "Result/API"
    CAR_IMG_DIR = os.path.join(OUTPUT_DIR, "cars")
    PART_IMG_DIR = os.path.join(OUTPUT_DIR, "parts")
    UPLOAD_FOLDER = os.path.join(OUTPUT_DIR, "uploads")
    
    for directory in [OUTPUT_DIR, CAR_IMG_DIR, PART_IMG_DIR, UPLOAD_FOLDER]:
        os.makedirs(directory, exist_ok=True)

# Global variables
detection_active = False # This primarily for camera/video
camera_active = False
current_detection_id = None
models_loaded = False
model_track = None
model_seg = None
names_track = None
names_seg = None

captured_car_info_for_review = None 

# Database setup
def init_database():
    conn = sqlite3.connect('car_detection.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            source_type TEXT,
            source_path TEXT,
            status TEXT,
            total_cars INTEGER
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cars (
            id TEXT PRIMARY KEY,
            detection_id TEXT,
            car_id INTEGER,
            condition_percentage INTEGER,
            car_image_path TEXT,
            detected_parts INTEGER,
            total_parts INTEGER,
            confirmed BOOLEAN DEFAULT FALSE,
            confirmation_time TEXT,
            FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE CASCADE
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parts (
            id TEXT PRIMARY KEY,
            car_db_id TEXT,
            part_name TEXT,
            part_id TEXT,
            status INTEGER, 
            confidence REAL,
            image_path TEXT,
            FOREIGN KEY (car_db_id) REFERENCES cars (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('car_detection.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def load_models():
    global models_loaded, model_track, model_seg, names_track, names_seg
    
    if models_loaded:
        return True
    
    try:
        if not os.path.isfile(Config.MODEL_TRACK_PATH):
            print(f"[ERROR] Tracking model not found: {Config.MODEL_TRACK_PATH}")
            return False
        if not os.path.isfile(Config.MODEL_SEGMENT_PATH):
            print(f"[ERROR] Segmentation model not found: {Config.MODEL_SEGMENT_PATH}")
            return False
        
        model_track = YOLO(Config.MODEL_TRACK_PATH)
        model_seg = YOLO(Config.MODEL_SEGMENT_PATH)
        names_track = model_track.names
        names_seg = model_seg.names
        
        models_loaded = True
        print("[INFO] Models loaded successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return False

class CarTrackingProcessor: 
    def __init__(self, detection_id, source_type, source_path=None):
        self.detection_id = detection_id
        self.source_type = source_type
        self.source_path = source_path
        self.track_history = {}
        self.captured_cars_data = {} 
        self.frame_count = 0
        self.car_was_captured = False 

    def process_frame(self, frame):
        """Process a single frame for car tracking only."""
        self.frame_count += 1
        
        frame_processed = cv2.resize(frame, (640, 480))
        
        results_track = model_track.track(frame_processed, persist=True, conf=0.4, classes=[2])
        
        detected_car_for_this_frame_info = None

        if results_track[0].boxes.id is not None:
            ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results_track[0].boxes.cls.int().cpu().tolist()
            
            for box, class_id, track_id in zip(boxes, class_ids, ids):
                if names_track[class_id] != "car":
                    continue
                
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_processed.shape[1], x2), min(frame_processed.shape[0], y2)
                car_crop = frame_processed[y1:y2, x1:x2] 
                
                if car_crop.shape[0] == 0 or car_crop.shape[1] == 0:
                    continue

                if track_id not in self.track_history:
                    self.track_history[track_id] = 0
                self.track_history[track_id] += 1

                if not self.car_was_captured and self.track_history[track_id] >= 5: 
                    car_key = f"car_{track_id}"
                    car_img_name = f"{self.detection_id}_{car_key}_{uuid.uuid4().hex[:8]}.jpg"
                    car_img_path = os.path.join(Config.CAR_IMG_DIR, car_img_name)
                    
                    try:
                        cv2.imwrite(car_img_path, car_crop)
                        detected_car_for_this_frame_info = {
                            "id": int(track_id),
                            "car_image_path": os.path.join(Config.CAR_IMG_DIR, car_img_name).replace('\\', '/'),
                            "detection_id": self.detection_id,
                            "car_key": car_key,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.captured_cars_data[car_key] = detected_car_for_this_frame_info
                        self.car_was_captured = True 
                        print(f"[INFO] Car {track_id} captured: {car_img_path}")
                        eventlet.sleep(0.001) 
                    except Exception as e:
                        print(f"[ERROR] Failed to save captured car image {car_img_path}: {e}")
                
                cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame_processed, f'Car ID:{track_id}', (x1, y1 - 10), scale=1, thickness=2)

        return frame_processed, detected_car_for_this_frame_info 

    def save_detection_record(self, total_cars=0, status='captured'):
        """Save initial detection record to database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO detections (id, timestamp, source_type, source_path, status, total_cars)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.detection_id,
                datetime.now().isoformat(),
                self.source_type,
                self.source_path,
                status, 
                total_cars 
            ))
            conn.commit()

    def update_car_and_parts_in_db(self, car_db_id, segmentation_data):
        """Updates car and parts information in the database after segmentation, setting confirmed to true."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            car_id = segmentation_data['id']
            condition_percentage = segmentation_data['condition']
            car_image_path = segmentation_data['car_image_path']
            parts = segmentation_data['parts']
            
            detected_count = sum(1 for p_data in parts.values() if p_data["status"] == 1)
            total_parts = len(parts)

            cursor.execute('''
                INSERT OR REPLACE INTO cars (id, detection_id, car_id, condition_percentage, car_image_path, detected_parts, total_parts, confirmed, confirmation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                car_db_id,
                self.detection_id,
                car_id,
                condition_percentage,
                car_image_path,
                detected_count,
                total_parts,
                True, # Auto-confirm here
                datetime.now().isoformat() # Set confirmation time
            ))

            cursor.execute('DELETE FROM parts WHERE car_db_id = ?', (car_db_id,))
            
            for part_name_key, part_data in parts.items(): 
                cursor.execute('''
                    INSERT INTO parts (id, car_db_id, part_name, part_id, status, confidence, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{car_db_id}_{part_data['name']}_{uuid.uuid4().hex[:4]}", 
                    car_db_id,
                    part_data["name"],
                    part_name_key, 
                    part_data["status"],
                    part_data["confidence"],
                    part_data.get("image_path") 
                ))
            
            # For camera captures, total_cars is dynamic. For image it's always 1.
            # This is specific for video/camera, for image it's updated in upload_image
            # cursor.execute('''
            #     UPDATE detections SET status = 'processed', total_cars = ? WHERE id = ?
            # ''', (len(self.captured_cars_data) if self.source_type == 'video' else 1, self.detection_id))
            
            conn.commit()


# API Routes
@app.route('/')
def index():
    return render_template_string('''
    <h1>Car Detection System API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>POST /api/upload-image - Upload image for detection</li>
        <li>GET /api/camera/start - Start camera tracking (capture car)</li>
        <li>GET /api/camera/stop - Stop camera tracking</li>
        <li>POST /api/segment-captured-car - Run segmentation on captured car</li>
        <li>GET /api/detections - Get all detection records</li>
        <li>GET /api/detections/&lt;detection_id&gt;/details - Get detailed report for a detection ID</li>
        <li>GET /api/cars-for-review - Get all cars for review with their details</li>
        <li>DELETE /api/delete-car/&lt;car_db_id&gt; - Delete a car and its parts</li>
        <li>GET /api/reports - Get detection reports</li>
    </ul>
    ''')

@app.route('/Result/API/parts/<path:filename>')
def serve_part_images(filename):
    return send_from_directory(Config.PART_IMG_DIR, filename)

@app.route('/Result/API/cars/<path:filename>')
def serve_car_images(filename):
    return send_from_directory(Config.CAR_IMG_DIR, filename)

# Serve uploaded images (for the source_path if needed, or if stored in CAR_IMG_DIR directly)
@app.route('/Result/API/uploads/<path:filename>')
def serve_uploaded_images(filename):
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if not load_models():
        return jsonify({"error": "Failed to load models"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)

    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save image file: {e}"}), 500

    detection_id = f"image_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    car_key = "car_1" # For single image, only one "car"
    car_db_id = f"{detection_id}_{car_key}"

    # Insert initial detection record
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO detections (id, timestamp, source_type, source_path, status, total_cars)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            detection_id,
            datetime.now().isoformat(),
            "image",
            file_path.replace('\\', '/'), # Store the uploaded file path
            "processing",
            1 # One car expected for single image
        ))
        conn.commit()

    # Run segmentation on the uploaded image
    # For a single image upload, car_id_int can be 1 as there's no tracking ID.
    segmentation_result = run_segmentation_on_image(file_path, 1)

    if segmentation_result:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                detected_count = sum(1 for p_data in segmentation_result['parts'].values() if p_data["status"] == 1)
                total_parts = len(segmentation_result['parts'])

                cursor.execute('''
                    INSERT OR REPLACE INTO cars (id, detection_id, car_id, condition_percentage, car_image_path, detected_parts, total_parts, confirmed, confirmation_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    car_db_id,
                    detection_id,
                    1, # car_id
                    segmentation_result['condition'],
                    file_path.replace('\\', '/'), # Use the uploaded file path as car_image_path
                    detected_count,
                    total_parts,
                    True, # Auto-confirm here for image uploads
                    datetime.now().isoformat()
                ))

                # Clear existing parts for this car_db_id if it's an update (unlikely for new image)
                cursor.execute('DELETE FROM parts WHERE car_db_id = ?', (car_db_id,))
                
                for part_name_key, part_data in segmentation_result['parts'].items():
                    cursor.execute('''
                        INSERT INTO parts (id, car_db_id, part_name, part_id, status, confidence, image_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"{car_db_id}_{part_data['name']}_{uuid.uuid4().hex[:4]}",
                        car_db_id,
                        part_data["name"],
                        part_name_key,
                        part_data["status"],
                        part_data["confidence"],
                        part_data.get("image_path") # This image_path refers to the segmented part image
                    ))
                
                # Update detection status to completed
                cursor.execute('''
                    UPDATE detections SET status = 'completed', total_cars = 1 WHERE id = ?
                ''', (detection_id,))
                
                conn.commit()

            return jsonify({
                "message": "Image processed successfully",
                "detection_id": detection_id,
                "car_db_id": car_db_id,
                "results": {
                    car_key: {
                        'id': 1, # car_id
                        'car_image_path': file_path.replace('\\', '/'),
                        'car_key': car_key,
                        'detection_id': detection_id,
                        **segmentation_result
                    }
                }
            })
        except Exception as e:
            print(f"[ERROR] Failed to save image segmentation results to DB: {e}", exc_info=True)
            with get_db_connection() as conn_err:
                cursor_err = conn_err.cursor()
                cursor_err.execute("UPDATE detections SET status = 'error' WHERE id = ?", (detection_id,))
                conn_err.commit()
            if os.path.exists(file_path):
                os.remove(file_path) # Clean up uploaded image if not fully processed
            return jsonify({"error": f"Failed to save segmentation results: {e}"}), 500
    else:
        with get_db_connection() as conn_err:
            cursor_err = conn_err.cursor()
            cursor_err.execute("UPDATE detections SET status = 'failed_segmentation' WHERE id = ?", (detection_id,))
            conn_err.commit()
        if os.path.exists(file_path):
            os.remove(file_path) # Clean up uploaded image
        return jsonify({"error": "Image segmentation failed."}), 500


@app.route('/api/camera/start', methods=['GET'])
def start_camera():
    global camera_active, current_detection_id, captured_car_info_for_review
    
    if camera_active:
        return jsonify({"error": "Camera already active"}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load models"}), 500
    
    current_detection_id = f"camera_{int(time.time())}"
    captured_car_info_for_review = None 
    
    eventlet.spawn_n(process_camera_tracking_stream, current_detection_id)
    
    return jsonify({
        "message": "Camera tracking started",
        "detection_id": current_detection_id
    })

@app.route('/api/camera/stop', methods=['GET'])
def stop_camera():
    global camera_active
    if camera_active:
        camera_active = False 
        print("[INFO] API stop requested. Signalling camera_active = False.")
    else:
        print("[INFO] API stop requested, but camera was not active.")
    
    return jsonify({"message": "Camera tracking stopped"})

def process_camera_tracking_stream(detection_id):
    global camera_active, captured_car_info_for_review
    
    camera_active = True 

    processor = CarTrackingProcessor(detection_id, "camera")
    processor.save_detection_record(status='tracking') 

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("[ERROR] Cannot access camera.")
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': 'Cannot access camera'
        })
        camera_active = False 
        return

    socketio.emit('camera_started', {
        'detection_id': detection_id,
        'status': 'Camera tracking started'
    })
    
    last_frame_emit_time = 0
    frame_emit_interval = 0.1 

    try:
        while camera_active: 
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break 

            processed_frame, detected_car_info = processor.process_frame(frame)

            if detected_car_info and not captured_car_info_for_review: 
                captured_car_info_for_review = detected_car_info 
                print(f"[INFO] Car {detected_car_info['id']} captured. Notifying frontend.")

            now = current_time()
            if now - last_frame_emit_time >= frame_emit_interval:
                last_frame_emit_time = now

                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                for sid in list(connected_sids):
                    try:
                        socketio.emit('camera_frame', {
                            'detection_id': detection_id,
                            'frame': frame_base64,
                            'captured_car_info': detected_car_info if detected_car_info else None 
                        }, to=sid)
                    except Exception as e:
                        print(f"[ERROR] Failed to emit camera_frame to {sid}: {e}")

            eventlet.sleep(0.01) 

    except Exception as e:
        print(f"[ERROR] Camera streaming greenlet error: {e}", exc_info=True)
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': str(e)
        })
    finally:
        def release_camera_async(camera_obj):
            try:
                camera_obj.release()
                print("[INFO] Camera released asynchronously.")
            except Exception as release_e:
                print(f"[ERROR] Error during async camera release: {release_e}", exc_info=True)

        eventlet.spawn_after(0.1, release_camera_async, cap) 
        
        stop_reason = 'captured' if captured_car_info_for_review else 'manual_stop'
        socketio.emit('camera_stopped', {
            'detection_id': detection_id,
            'reason': stop_reason,
            'captured_car_info': captured_car_info_for_review 
        })
        camera_active = False


def run_segmentation_on_image(image_path, car_id_int):
    """Helper function to run segmentation model on a given image path."""
    # This logic handles paths from CAR_IMG_DIR, PART_IMG_DIR, and UPLOAD_FOLDER
    if image_path.startswith('Result/API/cars/') or \
       image_path.startswith('Result/API/parts/') or \
       image_path.startswith('Result/API/uploads/'): # Added uploads
        
        # Determine the base directory to resolve the path correctly
        if image_path.startswith('Result/API/cars/'):
            base_dir = Config.CAR_IMG_DIR
        elif image_path.startswith('Result/API/parts/'):
            base_dir = Config.PART_IMG_DIR
        elif image_path.startswith('Result/API/uploads/'): # New for uploaded images
            base_dir = Config.UPLOAD_FOLDER
        else: # Fallback, though current logic should catch it
            base_dir = Config.OUTPUT_DIR 

        # Extract just the filename or relative part after the base directory
        relative_path_segment = image_path.split('/', 3)[-1] # Gets segment after 'Result/API/X/'
        full_system_path = os.path.join(base_dir, relative_path_segment).replace('\\', os.sep)
    else:
        full_system_path = image_path 

    if not os.path.exists(full_system_path):
        print(f"[ERROR] Image not found for segmentation: {full_system_path}")
        return None
    
    car_image = cv2.imread(full_system_path)
    if car_image is None:
        print(f"[ERROR] Failed to read image for segmentation: {full_system_path}")
        return None

    try:
        car_image_resized = cv2.resize(car_image, (640, 480))
        seg_result = model_seg.predict(car_image_resized, conf=0.4)[0]
        
        parts_data = {}
        for part_name_val in names_seg.values():
            parts_data[part_name_val] = {"name": part_name_val, "status": 0, "confidence": 0, "image_path": None}

        if seg_result.masks is not None:
            masks = seg_result.masks.data.cpu().numpy()
            boxes_seg = seg_result.boxes.xyxy.cpu().numpy().astype(int)
            classes = seg_result.boxes.cls.cpu().numpy().astype(int)
            confs_seg = seg_result.boxes.conf.cpu().numpy().astype(float)

            for i, (cls, conf_seg) in enumerate(zip(classes, confs_seg)):
                part_name = names_seg[cls].strip()
                if part_name in parts_data: 
                    x1, y1, x2, y2 = boxes_seg[i]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(car_image_resized.shape[1], x2), min(car_image_resized.shape[0], y2)

                    part_crop = car_image_resized[y1:y2, x1:x2]
                    
                    part_img_name = f"{car_id_int}_{part_name}_{uuid.uuid4().hex[:8]}.jpg"
                    part_img_path = os.path.join(Config.PART_IMG_DIR, part_img_name)

                    try:
                        mask = masks[i]
                        mask_resized = cv2.resize(mask.astype(np.uint8), (part_crop.shape[1], part_crop.shape[0]))
                        binary_mask = mask_resized > 0.5
                        
                        masked_part = np.zeros_like(part_crop)
                        masked_part[binary_mask] = part_crop[binary_mask]

                        color_overlay = np.zeros_like(part_crop, dtype=np.uint8)
                        color_overlay[binary_mask] = (0, 255, 0) 
                        combined_image = cv2.addWeighted(part_crop, 0.7, color_overlay, 0.3, 0)
                        
                        cv2.imwrite(part_img_path, combined_image)
                        parts_data[part_name]["image_path"] = part_img_path.replace('\\', '/') 
                    except Exception as e:
                        print(f"[ERROR] Failed to save segmented part image {part_img_path}: {e}")
                        parts_data[part_name]["image_path"] = None

                    parts_data[part_name]["status"] = 1 
                    parts_data[part_name]["confidence"] = float(conf_seg)
        
        detected_count = sum(1 for p_data in parts_data.values() if p_data["status"] == 1)
        total_parts = len(parts_data)
        condition_percentage = int((detected_count / total_parts * 100)) if total_parts > 0 else 0

        return {
            "condition": condition_percentage,
            "detected_parts": detected_count,
            "total_parts": total_parts,
            "parts": parts_data 
        }

    except Exception as e:
        print(f"[ERROR] Segmentation failed for image {full_system_path}: {e}", exc_info=True)
        return None


@app.route('/api/segment-captured-car', methods=['POST'])
def segment_captured_car():
    data = request.json
    car_image_path_from_frontend = data.get('car_image_path')
    car_id_int_from_frontend = data.get('car_id')
    detection_id = data.get('detection_id')
    car_key = data.get('car_key')

    if not car_image_path_from_frontend or car_id_int_from_frontend is None or not detection_id or not car_key:
        return jsonify({"error": "Missing car_image_path, car_id, detection_id, or car_key"}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load segmentation model."}), 500

    # Determine the full system path for the image based on its relative URL
    if car_image_path_from_frontend.startswith('Result/API/cars/'):
        relative_path_segment = car_image_path_from_frontend.replace('Result/API/cars/', '', 1)
        system_image_path = os.path.join(Config.CAR_IMG_DIR, relative_path_segment).replace('/', os.sep)
    elif car_image_path_from_frontend.startswith('Result/API/uploads/'): # New for uploaded images
        relative_path_segment = car_image_path_from_frontend.replace('Result/API/uploads/', '', 1)
        system_image_path = os.path.join(Config.UPLOAD_FOLDER, relative_path_segment).replace('/', os.sep)
    else:
        # Fallback, or handle other prefixes if they exist
        system_image_path = car_image_path_from_frontend


    segmentation_results = run_segmentation_on_image(system_image_path, car_id_int_from_frontend)

    if segmentation_results:
        car_db_id = f"{detection_id}_{car_key}"
        
        # We don't need a full CarTrackingProcessor instance for this,
        # just directly update the database.
        # This function implicitly confirms the car.
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            detected_count = sum(1 for p_data in segmentation_results['parts'].values() if p_data["status"] == 1)
            total_parts = len(segmentation_results['parts'])

            cursor.execute('''
                INSERT OR REPLACE INTO cars (id, detection_id, car_id, condition_percentage, car_image_path, detected_parts, total_parts, confirmed, confirmation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                car_db_id,
                detection_id,
                car_id_int_from_frontend,
                segmentation_results['condition'],
                car_image_path_from_frontend, 
                detected_count,
                total_parts,
                True, # Auto-confirm here
                datetime.now().isoformat() # Set confirmation time
            ))

            cursor.execute('DELETE FROM parts WHERE car_db_id = ?', (car_db_id,))
            
            for part_name_key, part_data in segmentation_results['parts'].items(): 
                cursor.execute('''
                    INSERT INTO parts (id, car_db_id, part_name, part_id, status, confidence, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{car_db_id}_{part_data['name']}_{uuid.uuid4().hex[:4]}", 
                    car_db_id,
                    part_data["name"],
                    part_name_key, 
                    part_data["status"],
                    part_data["confidence"],
                    part_data.get("image_path") 
                ))

            # Update the detection record status if needed, assuming this completes the 'processing' for a camera capture
            cursor.execute('''
                UPDATE detections SET status = 'completed_segmentation' WHERE id = ?
            ''', (detection_id,))
            
            conn.commit()
        
        return jsonify({
            "message": "Segmentation completed successfully and car confirmed",
            "detection_id": detection_id,
            "car_db_id": car_db_id,
            "results": {
                car_key: { 
                    'id': car_id_int_from_frontend,
                    'car_image_path': car_image_path_from_frontend,
                    'car_key': car_key, # Ensure car_key is in the returned results
                    'detection_id': detection_id, # Ensure detection_id is in the returned results
                    **segmentation_results
                }
            }
        })
    else:
        return jsonify({"error": "Segmentation failed for the captured car."}), 500


@app.route('/api/delete-car/<string:car_db_id>', methods=['DELETE'])
def delete_car(car_db_id):
    """Deletes a car record and its associated parts from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch car details to get image paths before deletion
            cursor.execute('SELECT car_image_path FROM cars WHERE id = ?', (car_db_id,))
            car_record = cursor.fetchone()
            if car_record:
                car_image_path = car_record['car_image_path']
                
                # Delete associated part images
                cursor.execute('SELECT image_path FROM parts WHERE car_db_id = ?', (car_db_id,))
                part_image_paths = cursor.fetchall()
                for part_path_row in part_image_paths:
                    if part_path_row['image_path']:
                        # Adjust path if it's a relative URL to system path
                        if part_path_row['image_path'].startswith('Result/API/parts/'):
                            part_full_path = os.path.join(Config.PART_IMG_DIR, part_path_row['image_path'].replace('Result/API/parts/', '')).replace('\\', os.sep)
                        else:
                            part_full_path = part_path_row['image_path'] # Assume it's already a system path

                        try:
                            if os.path.exists(part_full_path):
                                os.remove(part_full_path)
                                print(f"[INFO] Deleted part image: {part_full_path}")
                        except OSError as e:
                            print(f"[WARNING] Could not delete part image {part_full_path}: {e}")

                # Delete car record (parts will be cascade deleted by FK constraint)
                cursor.execute('DELETE FROM cars WHERE id = ?', (car_db_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    # Delete car image - this could be from CAR_IMG_DIR (camera) or UPLOAD_FOLDER (image upload)
                    if car_image_path:
                        car_full_path = None
                        if car_image_path.startswith('Result/API/cars/'):
                            car_full_path = os.path.join(Config.CAR_IMG_DIR, car_image_path.replace('Result/API/cars/', '')).replace('\\', os.sep)
                        elif car_image_path.startswith('Result/API/uploads/'):
                            car_full_path = os.path.join(Config.UPLOAD_FOLDER, car_image_path.replace('Result/API/uploads/', '')).replace('\\', os.sep)
                        else:
                            car_full_path = car_image_path # Assume it's a direct system path

                        if car_full_path:
                            try:
                                if os.path.exists(car_full_path):
                                    os.remove(car_full_path)
                                    print(f"[INFO] Deleted car image: {car_full_path}")
                            except OSError as e:
                                print(f"[WARNING] Could not delete car image {car_full_path}: {e}")
                    return jsonify({"message": "Car and associated data deleted successfully."}), 200
                else:
                    return jsonify({"error": "Car not found."}), 404
            else:
                return jsonify({"error": "Car not found."}), 404

    except Exception as e:
        print(f"[ERROR] Failed to delete car {car_db_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/reports', methods=['GET'])
def get_reports():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch all confirmed cars to be displayed in the reports table
            cursor.execute('''
                SELECT c.id, c.car_id, c.detection_id, d.timestamp as detection_date, 
                       c.condition_percentage, c.car_image_path, d.source_type
                FROM cars c
                JOIN detections d ON c.detection_id = d.id
                WHERE c.confirmed = 1
                ORDER BY d.timestamp DESC, c.car_id ASC
            ''')
            
            cars = cursor.fetchall()
            reports = []
            
            for car in cars:
                reports.append({
                    'id': car['id'], # This is the car_db_id (detection_id_car_key)
                    'car_id': car['car_id'], # This is the integer car ID (e.g., 1, 2, 3)
                    'detection_id': car['detection_id'],
                    'detection_date': car['detection_date'],
                    'condition': car['condition_percentage'],
                    'car_image_path': car['car_image_path'],
                    'source_type': car['source_type']
                })
        
        return jsonify({
            'reports': reports,
            'total_count': len(reports)
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to get reports: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Returns a summary of all detection records."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT d.*, 
                       (SELECT COUNT(*) FROM cars WHERE detection_id = d.id AND confirmed = 1) as confirmed_cars
                FROM detections d
                ORDER BY d.timestamp DESC
            ''')
            detections = cursor.fetchall()
            
            result = []
            for detection in detections:
                result.append({
                    'id': detection['id'],
                    'timestamp': detection['timestamp'],
                    'source_type': detection['source_type'],
                    'source_path': detection['source_path'],
                    'status': detection['status'],
                    'total_cars': detection['total_cars'],
                    'confirmed_cars': detection['confirmed_cars']
                })
        
        return jsonify({'detections': result})
        
    except Exception as e:
        print(f"[ERROR] Failed to get detections: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/detections/<string:detection_id>/details', methods=['GET'])
def get_detection_details(detection_id):
    """Returns comprehensive details for a specific detection ID, including all cars and their parts."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch detection record summary
            cursor.execute('SELECT * FROM detections WHERE id = ?', (detection_id,))
            detection_record = cursor.fetchone()
            if not detection_record:
                return jsonify({"error": "Detection record not found"}), 404
            
            # Fetch all cars associated with this detection
            cursor.execute('SELECT * FROM cars WHERE detection_id = ?', (detection_id,))
            cars = cursor.fetchall()
            
            cars_details = {}
            for car in cars:
                car_db_id = car['id']
                
                # Fetch all parts for the current car
                cursor.execute('SELECT * FROM parts WHERE car_db_id = ?', (car_db_id,))
                parts = cursor.fetchall()
                
                parts_data = {}
                for part in parts:
                    parts_data[part['part_id']] = { # Use part_id (e.g., 'headlight_1') as key
                        'name': part['part_name'],
                        'status': part['status'],
                        'confidence': part['confidence'],
                        'image_path': part['image_path']
                    }
                
                # Reconstruct car_key from car_db_id for frontend consistency
                # Assuming car_db_id is like 'detection_id_car_key' (e.g., 'video_12345_car_3')
                # For image upload it will be image_{timestamp}_{uuid}_car_1
                parts_of_car_db_id = car_db_id.split('_')
                if car_db_id.startswith('image_'):
                     car_key_from_db_id = parts_of_car_db_id[-1] # This will be 'car_1'
                else: # For camera/video
                    car_key_from_db_id = '_'.join(parts_of_car_db_id[len(parts_of_car_db_id)-2:]) if len(parts_of_car_db_id) >= 2 else None


                cars_details[car_key_from_db_id] = { 
                    'id': car['car_id'],
                    'condition': car['condition_percentage'],
                    'car_image_path': car['car_image_path'],
                    'detected_parts': car['detected_parts'],
                    'total_parts': car['total_parts'],
                    'confirmed': bool(car['confirmed']),
                    'confirmation_time': car['confirmation_time'],
                    'parts': parts_data,
                    'car_key': car_key_from_db_id, # Ensure car_key is explicitly included
                    'detection_id': detection_id # Ensure detection_id is explicitly included
                }
        
        full_report = {
            "detection_summary": dict(detection_record),
            "cars": cars_details
        }
        
        return jsonify(full_report)
        
    except Exception as e:
        print(f"[ERROR] Failed to get detection details for {detection_id}: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/cars-for-review', methods=['GET'])
def get_all_cars_for_review():
    """
    Returns all cars in the database that are either unconfirmed (for review)
    or have confirmed status, along with their full segmentation details.
    This replaces the need to iterate through detections and fetch details one by one on the frontend.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Fetch all cars
            cursor.execute('SELECT * FROM cars')
            all_cars_records = cursor.fetchall()
            
            cars_for_review_list = []
            for car_record in all_cars_records:
                car_db_id = car_record['id']
                
                # Fetch parts for each car
                cursor.execute('SELECT * FROM parts WHERE car_db_id = ?', (car_db_id,))
                parts = cursor.fetchall()
                
                parts_data = {}
                for part in parts:
                    parts_data[part['part_id']] = {
                        'name': part['part_name'],
                        'status': part['status'],
                        'confidence': part['confidence'],
                        'image_path': part['image_path']
                    }
                
                # Reconstruct car_key and detection_id from car_db_id
                parts_of_car_db_id = car_db_id.split('_')
                if car_db_id.startswith('image_'):
                    detection_id_from_db = '_'.join(parts_of_car_db_id[0:3]) # e.g., 'image_timestamp_uuid'
                    car_key_from_db = parts_of_car_db_id[-1] # 'car_1'
                else: # For camera/video
                    detection_id_from_db = '_'.join(parts_of_car_db_id[:-1])
                    car_key_from_db = parts_of_car_db_id[-1] 


                # Determine status for frontend
                status_for_frontend = 'Done'
                if car_record['condition_percentage'] is None:
                    status_for_frontend = 'Captured'
                elif car_record['confirmed']: # This will now always be true if segmented
                    status_for_frontend = 'Confirmed'


                cars_for_review_list.append({
                    'id': car_record['car_id'],
                    'detection_id': detection_id_from_db,
                    'car_key': car_key_from_db,
                    'car_image_path': car_record['car_image_path'],
                    'segmented': car_record['condition_percentage'] is not None, # True if segmentation results exist
                    'confirmed': bool(car_record['confirmed']), # Include confirmed status
                    'segmentation_results': {
                        'id': car_record['car_id'],
                        'condition': car_record['condition_percentage'],
                        'car_image_path': car_record['car_image_path'],
                        'detected_parts': car_record['detected_parts'],
                        'total_parts': car_record['total_parts'],
                        'confirmed': bool(car_record['confirmed']),
                        'confirmation_time': car_record['confirmation_time'],
                        'parts': parts_data,
                        'car_key': car_key_from_db,
                        'detection_id': detection_id_from_db
                    },
                    'status': status_for_frontend
                })
        
        return jsonify({'cars': cars_for_review_list})

    except Exception as e:
        print(f"[ERROR] Failed to get all cars for review: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print(f'[SocketIO] Client connected: {request.sid}')
    connected_sids.add(request.sid)
    emit('connected', {'message': 'Connected to Car Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'[SocketIO] Client disconnected: {request.sid}')
    connected_sids.discard(request.sid)

@socketio.on('ping_check')
def handle_ping():
    emit('pong_check', {'status': 'ok'})

if __name__ == '__main__':
    init_database()
    
    print("Loading AI models...")
    if load_models():
        print("✅ Models loaded successfully!")
    else:
        print("❌ Failed to load models. Check model paths and ensure required libraries are installed.")
        sys.exit(1)
    
    print("🚀 Starting Car Detection System...")
    print(f"Server will be available at http://0.0.0.0:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)