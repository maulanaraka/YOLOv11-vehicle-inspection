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
import sqlite3
from contextlib import contextmanager
import eventlet
from time import time as current_time

eventlet.monkey_patch()
connected_sids = set()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    ping_interval=10, ping_timeout=60) # Increased timeout
CORS(app)

# Configuration
class Config:
    MODEL_TRACK_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\yolo11n.pt"
    MODEL_SEGMENT_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\best-small.pt"
    OUTPUT_DIR = "Result/API"
    CAR_IMG_DIR = os.path.join(OUTPUT_DIR, "cars") # For captured car images
    PART_IMG_DIR = os.path.join(OUTPUT_DIR, "parts") # For segmented part images (if saved)
    UPLOAD_FOLDER = os.path.join(OUTPUT_DIR, "uploads") # For uploaded videos
    
    # Create directories
    for directory in [OUTPUT_DIR, CAR_IMG_DIR, PART_IMG_DIR, UPLOAD_FOLDER]:
        os.makedirs(directory, exist_ok=True)

# Global variables
detection_active = False # For video detection
camera_active = False # Flag for camera greenlet loop. Controlled by /api/camera/stop
current_detection_id = None
models_loaded = False
model_track = None
model_seg = None
names_track = None
names_seg = None

# New global for captured car information (temporarily stores until frontend requests segmentation)
captured_car_info_for_review = None # Stores { 'detection_id', 'car_key', 'car_image_path', 'car_id', 'timestamp' }

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
            FOREIGN KEY (detection_id) REFERENCES detections (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parts (
            id TEXT PRIMARY KEY,
            car_db_id TEXT,
            part_name TEXT,
            part_id TEXT,
            status INTEGER, -- 1 for detected, 0 for not detected
            confidence REAL,
            image_path TEXT,
            FOREIGN KEY (car_db_id) REFERENCES cars (id)
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

class CarTrackingProcessor: # Renamed for clarity (only tracking now)
    def __init__(self, detection_id, source_type, source_path=None):
        self.detection_id = detection_id
        self.source_type = source_type
        self.source_path = source_path
        self.track_history = {}
        self.captured_cars_data = {} # Stores captured car data before segmentation
        self.frame_count = 0
        self.car_was_captured = False # Flag to indicate if a car has been captured in this session

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
                # Ensure coordinates are within frame bounds before cropping
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_processed.shape[1], x2), min(frame_processed.shape[0], y2)
                car_crop = frame_processed[y1:y2, x1:x2] # Crop the car from the frame
                
                # Skip if crop is empty (e.g., due to invalid box)
                if car_crop.shape[0] == 0 or car_crop.shape[1] == 0:
                    continue

                if track_id not in self.track_history:
                    self.track_history[track_id] = 0
                self.track_history[track_id] += 1

                # Capture car image if it's new or sufficiently tracked and not already captured in this session
                if not self.car_was_captured and self.track_history[track_id] >= 5: # Track for 5 frames before capturing
                    car_key = f"car_{track_id}"
                    car_img_name = f"{self.detection_id}_{car_key}_{uuid.uuid4().hex[:8]}.jpg"
                    car_img_path = os.path.join(Config.CAR_IMG_DIR, car_img_name)
                    
                    try:
                        # Save the cropped car image
                        cv2.imwrite(car_img_path, car_crop)
                        detected_car_for_this_frame_info = {
                            "id": int(track_id),
                            "car_image_path": os.path.join(Config.CAR_IMG_DIR, car_img_name).replace('\\', '/'), # Use forward slashes for URL
                            "detection_id": self.detection_id,
                            "car_key": car_key,
                            "timestamp": datetime.now().isoformat()
                        }
                        self.captured_cars_data[car_key] = detected_car_for_this_frame_info
                        self.car_was_captured = True # Mark a car as captured for this session
                        print(f"[INFO] Car {track_id} captured: {car_img_path}")
                        # Add a small yield after potentially blocking I/O
                        eventlet.sleep(0.001) 
                    except Exception as e:
                        print(f"[ERROR] Failed to save captured car image {car_img_path}: {e}")
                
                # Draw detection on frame
                cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame_processed, f'Car ID:{track_id}', (x1, y1 - 10), scale=1, thickness=2)

        return frame_processed, detected_car_for_this_frame_info # Return captured car info if any

    def save_detection_record(self, total_cars=0, status='captured'):
        """Save initial detection record to database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (id, timestamp, source_type, source_path, status, total_cars)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.detection_id,
                datetime.now().isoformat(),
                self.source_type,
                self.source_path,
                status, # Status is now 'captured' or 'processing'
                total_cars # Will be updated after segmentation
            ))
            conn.commit()

    def update_car_and_parts_in_db(self, car_db_id, segmentation_data):
        """Updates car and parts information in the database after segmentation."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Extract data from segmentation_data
            car_id = segmentation_data['id']
            condition_percentage = segmentation_data['condition']
            car_image_path = segmentation_data['car_image_path']
            parts = segmentation_data['parts']
            
            detected_count = sum(1 for p_data in parts.values() if p_data["status"] == 1)
            total_parts = len(parts)

            # Update cars table (or insert if not already present from initial capture)
            cursor.execute('''
                INSERT OR REPLACE INTO cars (id, detection_id, car_id, condition_percentage, car_image_path, detected_parts, total_parts, confirmed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                car_db_id,
                self.detection_id,
                car_id,
                condition_percentage,
                car_image_path,
                detected_count,
                total_parts,
                False # Not confirmed yet
            ))

            # Delete old parts for this car_db_id if they exist
            cursor.execute('DELETE FROM parts WHERE car_db_id = ?', (car_db_id,))
            
            # Insert new part records
            for part_name_key, part_data in parts.items(): # part_name_key like 'headlight_1'
                cursor.execute('''
                    INSERT INTO parts (id, car_db_id, part_name, part_id, status, confidence, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    f"{car_db_id}_{part_data['name']}_{uuid.uuid4().hex[:4]}", # More unique part ID for DB
                    car_db_id,
                    part_data["name"],
                    part_name_key, # Use the original key for consistency if needed, or part_data['name']
                    part_data["status"],
                    part_data["confidence"],
                    part_data.get("image_path") # Use the image_path generated during segmentation
                ))
            
            # Update detection record status if all cars for this detection are processed
            # For camera, we'll set it to 'processed' once segmented.
            cursor.execute('''
                UPDATE detections SET status = 'processed', total_cars = ? WHERE id = ?
            ''', (len(self.captured_cars_data), self.detection_id))
            
            conn.commit()


# API Routes
@app.route('/')
def index():
    return render_template_string('''
    <h1>Car Detection System API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>POST /api/upload-video - Upload video for detection</li>
        <li>GET /api/camera/start - Start camera tracking (capture car)</li>
        <li>GET /api/camera/stop - Stop camera tracking</li>
        <li>POST /api/segment-captured-car - Run segmentation on captured car</li>
        <li>GET /api/detections - Get all detection records</li>
        <li>POST /api/confirm-car - Confirm car detection results</li>
        <li>GET /api/reports - Get detection reports</li>
    </ul>
    ''')

# Serving static files (captured car images, part images)
@app.route('/Result/API/parts/<path:filename>')
def serve_part_images(filename):
    return send_from_directory(Config.PART_IMG_DIR, filename)

@app.route('/Result/API/cars/<path:filename>')
def serve_car_images(filename):
    return send_from_directory(Config.CAR_IMG_DIR, filename)


# Video detection (remains similar, segments all cars in the video after tracking)
@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    global detection_active, current_detection_id
    
    if detection_active:
        return jsonify({"error": "Detection already in progress"}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load models"}), 500
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save video file: {e}"}), 500
    
    current_detection_id = f"video_{int(time.time())}"
    # Video will still do segmentation automatically for now
    eventlet.spawn_n(process_video_detection_with_segmentation, current_detection_id, file_path)
    
    return jsonify({
        "message": "Video uploaded successfully, processing started",
        "detection_id": current_detection_id
    })

def process_video_detection_with_segmentation(detection_id, video_path):
    global detection_active
    detection_active = True
    
    try:
        processor = CarTrackingProcessor(detection_id, "video", video_path) # Now handles initial capture
        processor.save_detection_record(status='processing') # Set initial status to processing
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            socketio.emit('detection_error', {
                'detection_id': detection_id,
                'error': 'Cannot open video file'
            })
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        socketio.emit('detection_progress', {
            'detection_id': detection_id,
            'status': 'started',
            'progress': 0,
            'message': 'Tracking cars in video...'
        })
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # For video, we don't automatically stop after one capture, we capture all
            processed_frame, captured_car_data_this_frame = processor.process_frame(frame) 
            # If a car was captured, add it to captured_cars_data in the processor
            if captured_car_data_this_frame and captured_car_data_this_frame['car_key'] not in processor.captured_cars_data:
                processor.captured_cars_data[captured_car_data_this_frame['car_key']] = captured_car_data_this_frame


            frame_count += 1
            
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = min(100, (frame_count / total_frames) * 100)
                socketio.emit('detection_progress', {
                    'detection_id': detection_id,
                    'status': 'tracking',
                    'progress': progress,
                    'message': f'Tracking {frame_count}/{total_frames} frames...',
                    'current_tracked_cars_count': len(processor.captured_cars_data) # Show tracked cars count
                })
        
        cap.release()
        
        # Now, process all captured cars from the video for segmentation
        captured_cars_list = list(processor.captured_cars_data.values())
        final_results = {}

        if not captured_cars_list:
            print(f"[INFO] No cars captured in video {detection_id}.")
            # Update detection status to completed, no cars found
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE detections SET status = 'completed_no_cars' WHERE id = ?", (detection_id,))
                conn.commit()
            socketio.emit('detection_complete', {
                'detection_id': detection_id,
                'status': 'completed',
                'results': {},
                'message': 'No cars detected in video.'
            })
            return # Exit if no cars to segment


        for i, car_info in enumerate(captured_cars_list):
            socketio.emit('detection_progress', {
                'detection_id': detection_id,
                'status': 'segmenting',
                'progress': 100 + ((i + 1) / len(captured_cars_list)) * 100, # Progress beyond 100 for segmentation phase
                'message': f'Segmenting car {i+1}/{len(captured_cars_list)}...'
            })
            segmentation_result = run_segmentation_on_image(car_info['car_image_path'], car_info['id'])
            if segmentation_result:
                car_db_id = f"{detection_id}_{car_info['car_key']}"
                processor.update_car_and_parts_in_db(car_db_id, {
                    'id': car_info['id'],
                    'car_image_path': car_info['car_image_path'],
                    **segmentation_result # Merge segmentation results
                })
                final_results[car_info['car_key']] = {
                    'id': car_info['id'],
                    'car_image_path': car_info['car_image_path'],
                    **segmentation_result
                }

        # Final status for detection record (total_cars will be updated by update_car_and_parts_in_db)
        # We need to explicitly update the overall detection status to 'completed' here.
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE detections SET status = 'completed' WHERE id = ?", (detection_id,))
            conn.commit()
        
        socketio.emit('detection_complete', {
            'detection_id': detection_id,
            'status': 'completed',
            'results': final_results
        })
        
    except Exception as e:
        print(f"[ERROR] Video detection error: {e}", exc_info=True)
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': str(e)
        })
    finally:
        detection_active = False
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/api/camera/start', methods=['GET'])
def start_camera():
    global camera_active, current_detection_id, captured_car_info_for_review
    
    if camera_active:
        return jsonify({"error": "Camera already active"}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load models"}), 500
    
    current_detection_id = f"camera_{int(time.time())}"
    captured_car_info_for_review = None # Reset captured car info
    
    # Start the greenlet that streams camera frames
    eventlet.spawn_n(process_camera_tracking_stream, current_detection_id)
    
    return jsonify({
        "message": "Camera tracking started",
        "detection_id": current_detection_id
    })

@app.route('/api/camera/stop', methods=['GET'])
def stop_camera():
    global camera_active
    # This endpoint is now the definitive way to stop the camera greenlet.
    # The greenlet will exit its loop because camera_active is set to False.
    if camera_active:
        camera_active = False # Signal the processing greenlet to stop
        # Give the greenlet a moment to exit its loop and release resources
        # We don't need a specific sleep here, eventlet will handle yielding
        print("[INFO] API stop requested. Signalling camera_active = False.")
    else:
        print("[INFO] API stop requested, but camera was not active.")
    
    return jsonify({"message": "Camera tracking stopped"})

# This greenlet now ONLY streams the frames and detects the first car.
# It does NOT stop the camera itself.
def process_camera_tracking_stream(detection_id):
    global camera_active, captured_car_info_for_review
    
    # Initialize camera_active to True when this greenlet starts
    # It will be set to False only by the /api/camera/stop endpoint
    camera_active = True 

    processor = CarTrackingProcessor(detection_id, "camera")
    processor.save_detection_record(status='tracking') # Initial status for camera detection

    cap = cv2.VideoCapture(0)  # Use default camera

    if not cap.isOpened():
        print("[ERROR] Cannot access camera.")
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': 'Cannot access camera'
        })
        # Reset camera_active if camera fails to open
        camera_active = False 
        return

    socketio.emit('camera_started', {
        'detection_id': detection_id,
        'status': 'Camera tracking started'
    })
    
    last_frame_emit_time = 0
    frame_emit_interval = 0.1 # Emit frame every 0.1 seconds (10 FPS)

    try:
        while camera_active: # Loop runs as long as camera_active is True
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break 

            processed_frame, detected_car_info = processor.process_frame(frame)

            if detected_car_info and not captured_car_info_for_review: # Only store the first captured car
                captured_car_info_for_review = detected_car_info # Store globally for review/segmentation
                print(f"[INFO] Car {detected_car_info['id']} captured. Notifying frontend.")
                # We do NOT set camera_active = False here. Frontend will call /api/camera/stop.

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

            eventlet.sleep(0.01) # Always yield

    except Exception as e:
        print(f"[ERROR] Camera streaming greenlet error: {e}", exc_info=True)
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': str(e)
        })
    finally:
        # This finally block executes when the 'while camera_active' loop exits.
        # This happens when /api/camera/stop sets camera_active to False.
        # SCHEDULE THE RELEASE AS A SEPARATE, SLIGHTLY DELAYED TASK
        def release_camera_async(camera_obj):
            try:
                camera_obj.release()
                print("[INFO] Camera released asynchronously.")
            except Exception as release_e:
                print(f"[ERROR] Error during async camera release: {release_e}", exc_info=True)

        eventlet.spawn_after(0.1, release_camera_async, cap) # Release 100ms later
        
        stop_reason = 'captured' if captured_car_info_for_review else 'manual_stop'
        socketio.emit('camera_stopped', {
            'detection_id': detection_id,
            'reason': stop_reason,
            'captured_car_info': captured_car_info_for_review # Send it one last time
        })
        camera_active = False


def run_segmentation_on_image(image_path, car_id_int):
    """Helper function to run segmentation model on a given image path."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found for segmentation: {image_path}")
        return None
    
    car_image = cv2.imread(image_path)
    if car_image is None:
        print(f"[ERROR] Failed to read image for segmentation: {image_path}")
        return None

    try:
        # Resize image for consistent segmentation input
        car_image_resized = cv2.resize(car_image, (640, 480))
        seg_result = model_seg.predict(car_image_resized, conf=0.4)[0]
        
        parts_data = {}
        # Initialize all expected parts as not detected
        for part_name_val in names_seg.values():
            parts_data[part_name_val] = {"name": part_name_val, "status": 0, "confidence": 0, "image_path": None}

        if seg_result.masks is not None:
            masks = seg_result.masks.data.cpu().numpy()
            boxes_seg = seg_result.boxes.xyxy.cpu().numpy().astype(int)
            classes = seg_result.boxes.cls.cpu().numpy().astype(int)
            confs_seg = seg_result.boxes.conf.cpu().numpy().astype(float)

            for i, (cls, conf_seg) in enumerate(zip(classes, confs_seg)):
                part_name = names_seg[cls].strip()
                if part_name in parts_data: # Ensure we only process expected parts
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
                        color_overlay[binary_mask] = (0, 255, 0) # Green overlay for detected part
                        combined_image = cv2.addWeighted(part_crop, 0.7, color_overlay, 0.3, 0)
                        
                        cv2.imwrite(part_img_path, combined_image)
                        parts_data[part_name]["image_path"] = part_img_path.replace('\\', '/') # Use forward slashes for URL
                    except Exception as e:
                        print(f"[ERROR] Failed to save segmented part image {part_img_path}: {e}")
                        parts_data[part_name]["image_path"] = None

                    parts_data[part_name]["status"] = 1 # Detected
                    parts_data[part_name]["confidence"] = float(conf_seg)
        
        detected_count = sum(1 for p_data in parts_data.values() if p_data["status"] == 1)
        total_parts = len(parts_data)
        condition_percentage = int((detected_count / total_parts * 100)) if total_parts > 0 else 0

        return {
            "condition": condition_percentage,
            "detected_parts": detected_count,
            "total_parts": total_parts,
            "parts": parts_data # Comprehensive part data
        }

    except Exception as e:
        print(f"[ERROR] Segmentation failed for image {image_path}: {e}", exc_info=True)
        return None


@app.route('/api/segment-captured-car', methods=['POST'])
def segment_captured_car():
    global captured_car_info_for_review
    
    if not captured_car_info_for_review:
        return jsonify({"error": "No car has been captured for segmentation."}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load segmentation model."}), 500

    car_info = captured_car_info_for_review
    car_image_path = car_info['car_image_path']
    
    # Ensure car_image_path is a valid local path for cv2.imread
    # It might be in URL format from the frontend, convert back if needed
    if car_image_path.startswith('Result/API'):
        car_image_path = os.path.join(Config.OUTPUT_DIR, car_image_path.replace('Result/API/', '')).replace('/', os.sep)


    segmentation_results = run_segmentation_on_image(car_image_path, car_info['id'])

    if segmentation_results:
        car_db_id = f"{car_info['detection_id']}_{car_info['car_key']}"
        
        # Use a temporary processor instance just for DB updates
        temp_processor = CarTrackingProcessor(car_info['detection_id'], "camera", car_info['car_image_path'])
        temp_processor.captured_cars_data[car_info['car_key']] = car_info # Populate captured_cars_data for update_car_and_parts_in_db
        temp_processor.update_car_and_parts_in_db(car_db_id, {
            'id': car_info['id'],
            'car_image_path': car_info['car_image_path'],
            **segmentation_results # Merge segmentation results
        })

        # Clear captured car info after processing
        captured_car_info_for_review = None

        return jsonify({
            "message": "Segmentation completed successfully",
            "detection_id": car_info['detection_id'],
            "car_db_id": car_db_id,
            "results": {
                car_info['car_key']: { # Return a dictionary matching frontend's expected format
                    'id': car_info['id'],
                    'car_image_path': car_info['car_image_path'],
                    **segmentation_results
                }
            }
        })
    else:
        return jsonify({"error": "Segmentation failed for the captured car."}), 500


@app.route('/api/confirm-car', methods=['POST'])
def confirm_car():
    data = request.json
    car_db_id = data.get('car_db_id')
    confirmed = data.get('confirmed', False)
    
    if not car_db_id:
        return jsonify({"error": "car_db_id required"}), 400
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE cars 
                SET confirmed = ?, confirmation_time = ?
                WHERE id = ?
            ''', (confirmed, datetime.now().isoformat(), car_db_id))
            conn.commit()
            
            if cursor.rowcount == 0:
                return jsonify({"error": "Car not found"}), 404
            
            # Optionally update the parent detection record's confirmed_cars count
            detection_id_from_car_db_id = car_db_id.split('_')[0]
            
            # Recalculate confirmed cars for this detection_id
            cursor.execute('''
                SELECT COUNT(*) FROM cars WHERE detection_id = ? AND confirmed = 1
            ''', (detection_id_from_car_db_id,))
            current_confirmed_count = cursor.fetchone()[0]

            # Get total cars for this detection_id
            cursor.execute('''
                SELECT total_cars FROM detections WHERE id = ?
            ''', (detection_id_from_car_db_id,))
            total_cars_in_detection_row = cursor.fetchone()
            total_cars_in_detection = total_cars_in_detection_row[0] if total_cars_in_detection_row else 0


            new_status = 'processed' # Default status for a detection with processed cars
            if total_cars_in_detection > 0:
                if current_confirmed_count == total_cars_in_detection:
                    new_status = 'completed_all_confirmed'
                elif current_confirmed_count > 0:
                    new_status = 'partial_confirmed'
                
            cursor.execute('''
                UPDATE detections
                SET status = ?
                WHERE id = ?
            ''', (new_status, detection_id_from_car_db_id))
            conn.commit()

        return jsonify({
            "message": f"Car {'confirmed' if confirmed else 'rejected'} successfully",
            "car_db_id": car_db_id,
            "confirmed": confirmed
        })
        
    except Exception as e:
        print(f"[ERROR] Confirmation failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.*, d.timestamp as detection_date, d.source_type
                FROM cars c
                JOIN detections d ON c.detection_id = d.id
                WHERE c.confirmed = 1
                ORDER BY d.timestamp DESC
            ''')
            
            cars = cursor.fetchall()
            reports = []
            
            for car in cars:
                cursor.execute('''
                    SELECT * FROM parts WHERE car_db_id = ?
                ''', (car['id'],))
                parts = cursor.fetchall()
                
                parts_data = {}
                for part in parts:
                    parts_data[part['part_name']] = { # Use part_name as key, more readable
                        'name': part['part_name'],
                        'status': part['status'],
                        'confidence': part['confidence'],
                        'image_path': part['image_path']
                    }
                
                reports.append({
                    'id': car['id'],
                    'car_id': car['car_id'],
                    'detection_date': car['detection_date'],
                    'confirmation_date': car['confirmation_time'],
                    'condition': car['condition_percentage'],
                    'parts': parts_data,
                    'source_type': car['source_type'],
                    'status': 'confirmed',
                    'car_image_path': car['car_image_path'] # Include car image for reports
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

# The /show-report route remains the same as it relies on frontend redirection
# and static file serving set up above.

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