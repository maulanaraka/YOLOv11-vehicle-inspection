from flask import Flask, request, jsonify, render_template_string
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
# Increase ping_timeout as detection can be intensive
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet',
                    ping_interval=10, ping_timeout=60) # Increased timeout
CORS(app)

# Configuration
class Config:
    # Use raw strings for paths to avoid issues with backslashes
    MODEL_TRACK_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\yolo11n.pt"
    MODEL_SEGMENT_PATH = r"D:\School\Semester 8\TUGAS AKHIR SEASON 2\Yolo Development\component-detection-webapp\Backend\Model\best-small.pt"
    OUTPUT_DIR = "Result/API"
    CAR_IMG_DIR = os.path.join(OUTPUT_DIR, "cars")
    PART_IMG_DIR = os.path.join(OUTPUT_DIR, "parts")
    UPLOAD_FOLDER = os.path.join(OUTPUT_DIR, "uploads")
    
    # Create directories
    for directory in [OUTPUT_DIR, CAR_IMG_DIR, PART_IMG_DIR, UPLOAD_FOLDER]:
        os.makedirs(directory, exist_ok=True)

# Global variables
detection_active = False
camera_active = False
current_detection_id = None
models_loaded = False
model_track = None
model_seg = None
names_track = None
names_seg = None

# Database setup
def init_database():
    conn = sqlite3.connect('car_detection.db')
    cursor = conn.cursor()
    
    # Create tables
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
        # Validate model paths
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

class CarDetectionProcessor:
    def __init__(self, detection_id, source_type, source_path=None):
        self.detection_id = detection_id
        self.source_type = source_type
        self.source_path = source_path
        self.track_history = {}
        # Stores comprehensive data for each car, including parts
        self.json_data = {} 
        self.frame_count = 0
        self.expected_parts = set(names_seg.values()) # All possible parts the segmentation model can detect

    def process_frame(self, frame, fps=30):
        """Process a single frame and return detection results (tracking and segmentation)"""
        self.frame_count += 1
        
        # Skip every other frame for performance (only for tracking and segmentation)
        # For real-time display, you might want to process more frames, but segment less often.
        process_detection = (self.frame_count % 5 == 0) # Process detection every 5th frame
        
        # Resize frame for consistent processing
        frame_display = frame.copy() # Keep a copy for display without drawing segmentation results
        frame_processed = cv2.resize(frame, (640, 480))
        
        # Track cars using tracking model
        results_track = model_track.track(frame_processed, persist=True, conf=0.4, classes=[2])
        
        current_frame_car_ids = set()

        if results_track[0].boxes.id is not None:
            ids = results_track[0].boxes.id.cpu().numpy().astype(int)
            boxes = results_track[0].boxes.xyxy.cpu().numpy().astype(int)
            class_ids = results_track[0].boxes.cls.int().cpu().tolist()
            
            for box, class_id, track_id in zip(boxes, class_ids, ids):
                if names_track[class_id] != "car":
                    continue
                
                x1, y1, x2, y2 = box
                car_crop = frame_processed[y1:y2, x1:x2]  # Crop the car from the processed frame
                car_key = f"car_{track_id}"
                current_frame_car_ids.add(track_id)
                
                # Initialize car data if not exists
                if car_key not in self.json_data:
                    car_img_name = f"{track_id}_{uuid.uuid4().hex[:8]}.jpg"
                    car_img_path = os.path.join(Config.CAR_IMG_DIR, car_img_name)
                    try:
                        cv2.imwrite(car_img_path, car_crop)  # Save cropped car image
                    except Exception as e:
                        print(f"[ERROR] Failed to save car image {car_img_path}: {e}")
                        car_img_path = None # Mark as none if saving failed
                    
                    self.json_data[car_key] = {
                        "id": int(track_id),
                        "car_image": car_img_path,
                        "parts": {part_name: {"name": part_name, "status": 0, "confidence": 0, "image": None} for part_name in self.expected_parts}, # Initialize all parts as not detected
                        "condition": 0, # Initial condition
                        "detected_parts": 0,
                        "total_parts": len(self.expected_parts)
                    }
                
                # Perform segmentation on the car crop (less frequently for performance)
                if process_detection and car_crop.size > 0: # Ensure car_crop is not empty
                    try:
                        seg_result = model_seg.predict(car_crop, conf=0.4)[0]
                        if seg_result.masks is not None:
                            masks = seg_result.masks.data.cpu().numpy()
                            classes_seg = seg_result.boxes.cls.cpu().numpy().astype(int)
                            confs_seg = seg_result.boxes.conf.cpu().numpy().astype(float)

                            # Reset detected parts for this car in this frame
                            for part_name in self.expected_parts:
                                self.json_data[car_key]["parts"][part_name]["status"] = 0
                                self.json_data[car_key]["parts"][part_name]["confidence"] = 0
                                self.json_data[car_key]["parts"][part_name]["image"] = None

                            for i, (cls_seg, conf_seg) in enumerate(zip(classes_seg, confs_seg)):
                                part_name = names_seg[cls_seg].strip()
                                # Only update if this part is actually expected
                                if part_name in self.json_data[car_key]["parts"]:
                                    self.json_data[car_key]["parts"][part_name]["status"] = 1 # Detected
                                    self.json_data[car_key]["parts"][part_name]["confidence"] = float(conf_seg)
                                    
                                    # Generate and save part image (optional, can be memory intensive)
                                    # mask = masks[i]
                                    # part_img_name = f"{track_id}_{part_name}_{uuid.uuid4().hex[:8]}.jpg"
                                    # part_img_path = os.path.join(Config.PART_IMG_DIR, part_img_name)
                                    # mask_resized = cv2.resize(mask.astype(np.uint8), (car_crop.shape[1], car_crop.shape[0]))
                                    # binary_mask = mask_resized > 0.5
                                    # color_mask = np.zeros_like(car_crop)
                                    # color_mask[binary_mask] = (0, 255, 255)
                                    # highlighted = cv2.addWeighted(car_crop, 0.7, color_mask, 0.3, 0)
                                    # try:
                                    #     cv2.imwrite(part_img_path, highlighted)
                                    #     self.json_data[car_key]["parts"][part_name]["image"] = part_img_path
                                    # except Exception as e:
                                    #     print(f"[ERROR] Failed to save part image {part_img_path}: {e}")
                                    #     self.json_data[car_key]["parts"][part_name]["image"] = None
                                    
                        # Update condition based on detected parts
                        detected_count = sum(1 for p_data in self.json_data[car_key]["parts"].values() if p_data["status"] == 1)
                        total_parts = len(self.json_data[car_key]["parts"])
                        self.json_data[car_key]["detected_parts"] = detected_count
                        self.json_data[car_key]["total_parts"] = total_parts
                        self.json_data[car_key]["condition"] = int((detected_count / total_parts * 100)) if total_parts > 0 else 0

                    except Exception as e:
                        print(f"[ERROR] Segmentation failed for car {track_id}: {e}")
                
                # Draw detection on frame_processed (for display)
                cv2.rectangle(frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                condition_text = f"Car ID:{track_id} ({self.json_data[car_key]['condition']}%)"
                cvzone.putTextRect(frame_processed, condition_text, (x1, y1 - 10), scale=1, thickness=2)
        
        # Remove cars that are no longer detected (optional, depends on tracking logic)
        # If a car is not seen for multiple frames, it could be removed or marked as "gone"
        
        return frame_processed, self.get_current_results()
    
    def get_current_results(self):
        """Get current detection results (car IDs, condition, and parts status)"""
        results_for_emit = {}
        for car_key, car_data in self.json_data.items():
            # Create a serializable version of parts
            serializable_parts = {
                part_name: {
                    "name": p_data["name"],
                    "status": p_data["status"],
                    "confidence": p_data["confidence"]
                    # Do not include image path here for live updates to avoid excessive data
                } for part_name, p_data in car_data["parts"].items()
            }

            results_for_emit[car_key] = {
                "id": car_data["id"],
                "condition": car_data["condition"],
                "detected_parts": car_data["detected_parts"],
                "total_parts": car_data["total_parts"],
                "parts": serializable_parts # Include parts data
            }
        return results_for_emit

    
    def save_to_database(self):
        """Save detection results to database"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Save detection record
            cursor.execute('''
                INSERT INTO detections (id, timestamp, source_type, source_path, status, total_cars)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.detection_id,
                datetime.now().isoformat(),
                self.source_type,
                self.source_path,
                'completed',
                len(self.json_data)
            ))
            
            # Save car records
            for car_key, car_data in self.json_data.items():
                car_db_id = f"{self.detection_id}_{car_key}"
                
                cursor.execute('''
                    INSERT INTO cars (id, detection_id, car_id, condition_percentage, car_image_path, detected_parts, total_parts)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    car_db_id,
                    self.detection_id,
                    car_data["id"],
                    car_data["condition"],
                    car_data["car_image"],
                    car_data["detected_parts"],
                    car_data["total_parts"]
                ))
                
                # Save part records
                for part_id, part_data in car_data["parts"].items():
                    cursor.execute('''
                        INSERT INTO parts (id, car_db_id, part_name, part_id, status, confidence, image_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"{car_db_id}_{part_data['name']}", # Unique ID for part
                        car_db_id,
                        part_data["name"],
                        part_id, # Use original part_id if available, otherwise generated
                        part_data["status"],
                        part_data["confidence"],
                        part_data.get("image")
                    ))
            
            conn.commit()

# API Routes
@app.route('/')
def index():
    return render_template_string('''
    <h1>Car Detection System API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li>POST /api/upload-video - Upload video for detection</li>
        <li>GET /api/camera/start - Start camera detection</li>
        <li>GET /api/camera/stop - Stop camera detection</li>
        <li>GET /api/detections - Get all detection records</li>
        <li>POST /api/confirm-car - Confirm car detection results</li>
        <li>GET /api/reports - Get detection reports</li>
    </ul>
    ''')

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
    
    # Save uploaded file
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save video file: {e}"}), 500
    
    # Start detection in background thread
    current_detection_id = f"video_{int(time.time())}"
    eventlet.spawn_n(process_video_detection, current_detection_id, file_path)
    
    return jsonify({
        "message": "Video uploaded successfully, processing started",
        "detection_id": current_detection_id
    })

def process_video_detection(detection_id, video_path):
    global detection_active
    detection_active = True
    
    try:
        processor = CarDetectionProcessor(detection_id, "video", video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            socketio.emit('detection_error', {
                'detection_id': detection_id,
                'error': 'Cannot open video file'
            })
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Notify client that detection has started
        socketio.emit('detection_progress', {
            'detection_id': detection_id,
            'status': 'started',
            'progress': 0,
            'message': 'Processing video...'
        })
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, results = processor.process_frame(frame, fps) # results are now comprehensive
            frame_count += 1
            
            # Update progress periodically (e.g., every 30 frames)
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = min(100, (frame_count / total_frames) * 100)
                socketio.emit('detection_progress', {
                    'detection_id': detection_id,
                    'status': 'processing',
                    'progress': progress,
                    'message': f'Processed {frame_count}/{total_frames} frames',
                    'current_results': results
                })
        
        cap.release()
        
        # Save results to database
        processor.save_to_database()
        
        # Final results
        final_results = processor.get_current_results()
        socketio.emit('detection_complete', {
            'detection_id': detection_id,
            'status': 'completed',
            'results': final_results
        })
        
    except Exception as e:
        print(f"[ERROR] Video detection error: {e}", exc_info=True) # Print traceback
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': str(e)
        })
    finally:
        detection_active = False
        # Clean up uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/api/camera/start', methods=['GET'])
def start_camera():
    global camera_active, current_detection_id
    
    if camera_active:
        return jsonify({"error": "Camera already active"}), 400
    
    if not load_models():
        return jsonify({"error": "Failed to load models"}), 500
    
    current_detection_id = f"camera_{int(time.time())}"
    eventlet.spawn_n(process_camera_detection, current_detection_id)

    return jsonify({
        "message": "Camera detection started",
        "detection_id": current_detection_id
    })

@app.route('/api/camera/stop', methods=['GET'])
def stop_camera():
    global camera_active
    camera_active = False # Signal the processing thread to stop
    # Wait a bit for the thread to clean up
    eventlet.sleep(0.5) 
    return jsonify({"message": "Camera detection stopped"})

def process_camera_detection(detection_id):
    global camera_active
    camera_active = True

    try:
        processor = CarDetectionProcessor(detection_id, "camera")
        cap = cv2.VideoCapture(0)  # Use default camera

        if not cap.isOpened():
            socketio.emit('detection_error', {
                'detection_id': detection_id,
                'error': 'Cannot access camera'
            })
            return

        socketio.emit('camera_started', {
            'detection_id': detection_id,
            'status': 'Camera detection started'
        })
        
        last_frame_emit_time = 0
        last_status_emit_time = 0
        frame_emit_interval = 0.1 # Emit frame every 0.1 seconds (10 FPS)
        status_emit_interval = 1 # Emit status every 1 second

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            processed_frame, results_for_emit = processor.process_frame(frame) # Now returns full results

            now = current_time()
            
            # Emit processed frame to connected clients (less frequently)
            if now - last_frame_emit_time >= frame_emit_interval:
                last_frame_emit_time = now

                # Encode and emit frame
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Compress JPEG
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                for sid in list(connected_sids):
                    try:
                        socketio.emit('camera_frame', {
                            'detection_id': detection_id,
                            'frame': frame_base64,
                            'results': results_for_emit # Send comprehensive results with frame
                        }, to=sid)
                    except Exception as e:
                        print(f"[ERROR] Failed to emit camera_frame to {sid}: {e}")

            # Emit detailed status update (even less frequently)
            if now - last_status_emit_time >= status_emit_interval:
                last_status_emit_time = now
                
                car_status_list = []
                for car_data in results_for_emit.values():
                    car_status_list.append({
                        'id': car_data["id"],
                        'condition': car_data["condition"],
                        'detected_parts': car_data["detected_parts"],
                        'total_parts': car_data["total_parts"]
                    })

                for sid in list(connected_sids):
                    try:
                        socketio.emit('camera_status', {
                            'detection_id': detection_id,
                            'status': 'active',
                            'car_count': len(car_status_list),
                            'cars': car_status_list
                        }, to=sid)
                    except Exception as e:
                        print(f"[ERROR] Failed to emit camera_status to {sid}: {e}")

            eventlet.sleep(0.01) # Small sleep to yield to other greenlets

        cap.release()

        # Save results when stopping
        if processor.json_data:
            processor.save_to_database()

        socketio.emit('camera_stopped', {
            'detection_id': detection_id,
            'final_results': processor.get_current_results()
        })

    except Exception as e:
        print(f"[ERROR] Camera detection error: {e}", exc_info=True) # Print traceback
        socketio.emit('detection_error', {
            'detection_id': detection_id,
            'error': str(e)
        })
    finally:
        camera_active = False


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
            
            # Get all confirmed cars with their parts
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
                # Get parts for this car
                cursor.execute('''
                    SELECT * FROM parts WHERE car_db_id = ?
                ''', (car['id'],))
                parts = cursor.fetchall()
                
                parts_data = {}
                for part in parts:
                    parts_data[part['part_id']] = { # Use part['part_id'] which is more unique per car
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
                    'status': 'confirmed'
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

# This endpoint is probably not needed if segmentation is done live in process_frame
# Keeping it for now but consider removing if not used.
@app.route('/api/segment-cars', methods=['POST'])
def segment_cars():
    # This endpoint seems to be for batch segmentation of already cropped cars.
    # If the goal is real-time, the segmentation logic should be in process_frame.
    return jsonify({"error": "This endpoint is typically for batch processing. Real-time segmentation is integrated into camera/video processing."}), 400

@app.route('/show-report')
def show_report():
    # This endpoint is designed to render a report based on query parameters.
    # The frontend should navigate to this URL with the results.
    segmentation_results_str = request.args.get('segmentation_results')
    segmentation_results = {}
    if segmentation_results_str:
        try:
            segmentation_results = json.loads(segmentation_results_str)
        except json.JSONDecodeError:
            print("Error decoding segmentation_results from query param.")
            segmentation_results = {"error": "Invalid segmentation results data"}
    
    # You will need to serve the images from Config.CAR_IMG_DIR and Config.PART_IMG_DIR
    # A simple way for development is to add a static route:
    # app.add_url_rule('/Result/API/parts/<path:filename>', endpoint='part_images',
    #                  view_func=lambda filename: send_from_directory(Config.PART_IMG_DIR, filename))
    # app.add_url_rule('/Result/API/cars/<path:filename>', endpoint='car_images',
    #                  view_func=lambda filename: send_from_directory(Config.CAR_IMG_DIR, filename))
    from flask import send_from_directory
    @app.route('/Result/API/parts/<path:filename>')
    def serve_part_images(filename):
        return send_from_directory(Config.PART_IMG_DIR, filename)

    @app.route('/Result/API/cars/<path:filename>')
    def serve_car_images(filename):
        return send_from_directory(Config.CAR_IMG_DIR, filename)

    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Car Segmentation Report</title>
            <style>
                body { font-family: sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }
                .container { max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1, h2 { color: #4a5568; }
                .car-report { border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 20px; background-color: #fcfcfc; }
                .car-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
                .car-id-label { font-size: 1.8rem; font-weight: bold; color: #2d3748; }
                .car-condition { font-size: 1.5rem; font-weight: bold; padding: 5px 15px; border-radius: 5px; color: white; }
                .condition-good { background-color: #48bb78; }
                .condition-fair { background-color: #ed8936; }
                .condition-poor { background-color: #f56565; }
                .car-image { max-width: 100%; height: auto; border-radius: 5px; margin-top: 15px; border: 1px solid #ccc;}
                .parts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
                .part-item { border: 1px solid #cbd5e0; border-radius: 8px; padding: 10px; text-align: center; background-color: #f7fafc; }
                .part-item.detected { border-left: 5px solid #48bb78; background: #f0fff4; }
                .part-item.missing { border-left: 5px solid #f56565; background: #fffaf0; }
                .part-name { font-weight: bold; color: #2d3748; margin-bottom: 5px; }
                .part-status { font-size: 0.9em; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
                .status-ok { background-color: #c6f6d5; color: #22543d; }
                .status-missing { background-color: #fed7d7; color: #742a2a; }
                .part-image { max-width: 100%; height: auto; border-radius: 5px; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Car Segmentation Report</h1>
                {% if segmentation_results.error %}
                    <p style="color: red;">Error: {{ segmentation_results.error }}</p>
                {% elif not segmentation_results %}
                    <p>No segmentation results available. Please ensure a video was processed or camera detection was run and confirmed.</p>
                {% else %}
                    {% for car_key, car_data in segmentation_results.items() %}
                        <div class="car-report">
                            <div class="car-header">
                                <div class="car-id-label">Car ID: {{ car_data.id }}</div>
                                <div class="car-condition 
                                    {% if car_data.condition >= 80 %}condition-good
                                    {% elif car_data.condition >= 60 %}condition-fair
                                    {% else %}condition-poor{% endif %}">
                                    {{ car_data.condition }}%
                                </div>
                            </div>
                            {% if car_data.car_image %}
                                <p>Original Car Image:</p>
                                <img src="/{{ car_data.car_image }}" alt="Car {{ car_data.id }}" class="car-image">
                            {% endif %}
                            
                            <h3>Detected Parts ({{ car_data.detected_parts }} / {{ car_data.total_parts }})</h3>
                            <div class="parts-grid">
                                {% for part_name, part in car_data.parts.items() %}
                                    <div class="part-item {% if part.status %}detected{% else %}missing{% endif %}">
                                        <span class="part-name">{{ part.name.replace('_', ' ').upper() }}</span>
                                        <span class="part-status {% if part.status %}status-ok{% else %}status-missing{% endif %}">
                                            {{ 'DETECTED' if part.status else 'NOT DETECTED' }}
                                            {% if part.status and part.confidence %} ({{ '%.1f' % (part.confidence * 100) }}%) {% endif %}
                                        </span>
                                        {% if part.image_path %}
                                            <img src="/{{ part.image_path }}" alt="{{ part.name }}" class="part-image">
                                        {% endif %}
                                    </div>
                                {% endfor %}
                            </div>
                        </div>
                    {% endfor %}
                {% endif %}
            </div>
        </body>
        </html>
    ''', segmentation_results=segmentation_results)


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
    # Initialize database
    init_database()
    
    # Load models on startup
    print("Loading AI models...")
    if load_models():
        print("✅ Models loaded successfully!")
    else:
        print("❌ Failed to load models. Check model paths and ensure required libraries are installed.")
        sys.exit(1) # Exit if models fail to load
    
    # Run the application
    print("🚀 Starting Car Detection System...")
    print(f"Server will be available at http://0.0.0.0:5000")
    print("📹 Video detection: POST /api/upload-video")
    print("📷 Camera detection: GET /api/camera/start")
    print("📊 Reports: GET /api/reports")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, use_reloader=False)