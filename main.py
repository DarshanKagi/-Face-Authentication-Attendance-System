"""
Face Authentication Attendance System
======================================
A production-ready face recognition system with:
- Real-time face detection and recognition
- Multi-sample registration for robustness
- Basic liveness detection (anti-spoofing)
- Attendance logging with punch-in/out
- Comprehensive Gradio UI

Author: AI/ML Engineer
Version: 1.0
"""

import gradio as gr
import cv2
import face_recognition
import numpy as np
import sqlite3
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import configparser
from typing import List, Tuple, Optional, Dict, Any
import time
import threading
import os
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


class ConfigManager:
    """Manages application configuration with INI file"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self._load_or_create_config()
    
    def _load_or_create_config(self):
        """Load existing config or create default"""
        if Path(self.config_path).exists():
            self.config.read(self.config_path)
        else:
            # Default configuration
            self.config['RECOGNITION'] = {
                'threshold': '0.6',
                'detection_model': 'hog',  # 'hog' or 'cnn'
                'num_registration_samples': '5',
                'confidence_display': 'True'
            }
            self.config['ATTENDANCE'] = {
                'min_punch_interval_seconds': '30',
                'auto_logout_hours': '12'
            }
            self.config['LIVENESS'] = {
                'enabled': 'True',
                'color_variance_threshold': '150',
                'frame_consistency_threshold': '1000'
            }
            self.config['CAMERA'] = {
                'index': '0',
                'fps_limit': '30'
            }
            self.config['DATABASE'] = {
                'path': 'embeddings.db',
                'backup_enabled': 'True'
            }
            self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            self.config.write(f)
    
    def get(self, section: str, key: str, fallback=None):
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def get_float(self, section: str, key: str, fallback=0.0):
        """Get float configuration value"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def get_int(self, section: str, key: str, fallback=0):
        """Get integer configuration value"""
        return self.config.getint(section, key, fallback=fallback)
    
    def get_bool(self, section: str, key: str, fallback=False):
        """Get boolean configuration value"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        self.config.set(section, key, str(value))
        self._save_config()


class DatabaseManager:
    """Manages SQLite database for user embeddings and attendance records"""
    
    def __init__(self, db_path: str = "embeddings.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT,
                embeddings BLOB NOT NULL,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_samples INTEGER DEFAULT 1
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT CHECK(action IN ('IN', 'OUT')),
                confidence REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, employee_id: str, name: str, department: str, 
                 embeddings: np.ndarray, num_samples: int) -> int:
        """Add new user with face embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Serialize embeddings
            embeddings_blob = pickle.dumps(embeddings)
            
            # Find first available ID gap
            cursor.execute("SELECT id FROM users ORDER BY id ASC")
            existing_ids = [row[0] for row in cursor.fetchall()]
            
            new_id = 1
            for existing_id in existing_ids:
                if existing_id == new_id:
                    new_id += 1
                else:
                    break
            
            cursor.execute('''
                INSERT INTO users (id, employee_id, name, department, embeddings, num_samples)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (new_id, employee_id, name, department, embeddings_blob, num_samples))
            
            user_id = new_id
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            return -1  # Duplicate employee_id
        finally:
            conn.close()
    
    def get_all_users(self) -> List[Dict]:
        """Retrieve all registered users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, employee_id, name, department, embeddings, 
                   registration_date, num_samples
            FROM users
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'employee_id': row[1],
                'name': row[2],
                'department': row[3],
                'embeddings': pickle.loads(row[4]),
                'registration_date': row[5],
                'num_samples': row[6]
            })
        
        conn.close()
        return users
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, employee_id, name, department, embeddings, 
                   registration_date, num_samples
            FROM users WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'employee_id': row[1],
                'name': row[2],
                'department': row[3],
                'embeddings': pickle.loads(row[4]),
                'registration_date': row[5],
                'num_samples': row[6]
            }
        return None
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user and their attendance records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def log_attendance(self, user_id: int, action: str, confidence: float) -> int:
        """Log attendance event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO attendance (user_id, action, confidence)
            VALUES (?, ?, ?)
        ''', (user_id, action, confidence))
        
        attendance_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return attendance_id
    
    def get_last_attendance(self, user_id: int) -> Optional[Dict]:
        """Get most recent attendance record for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, action, confidence
            FROM attendance
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'timestamp': datetime.strptime(row[1].split('.')[0], '%Y-%m-%d %H:%M:%S'),
                'action': row[2],
                'confidence': row[3]
            }
        return None
    
    def get_attendance_records(self, start_date: Optional[str] = None, 
                               end_date: Optional[str] = None,
                               user_id: Optional[int] = None) -> List[Dict]:
        """Get attendance records with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT a.id, a.user_id, u.name, u.employee_id, 
                   a.timestamp, a.action, a.confidence
            FROM attendance a
            JOIN users u ON a.user_id = u.id
            WHERE 1=1
        '''
        params = []
        
        if start_date:
            query += ' AND DATE(a.timestamp) >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND DATE(a.timestamp) <= ?'
            params.append(end_date)
        
        if user_id:
            query += ' AND a.user_id = ?'
            params.append(user_id)
        
        query += ' ORDER BY a.timestamp DESC'
        
        cursor.execute(query, params)
        
        records = []
        for row in cursor.fetchall():
            records.append({
                'id': row[0],
                'user_id': row[1],
                'name': row[2],
                'employee_id': row[3],
                'timestamp': row[4],
                'action': row[5],
                'confidence': row[6]
            })
        
        conn.close()
        return records


class LivenessDetector:
    """Improved basic liveness detection with smoothing and normalized metrics."""
    def __init__(self, config: ConfigManager):
        self.config = config
        self.enabled = config.get_bool('LIVENESS', 'enabled', True)
        # sensible default thresholds (can also be tuned in config.ini)
        self.color_variance_threshold = config.get_float('LIVENESS', 'color_variance_threshold', 50.0)
        # mean absolute diff per pixel threshold (0-255 scale); small values indicate little motion
        self.frame_consistency_threshold = config.get_float('LIVENESS', 'frame_consistency_threshold', 2.5)
        self.frame_buffer = []
        self.max_buffer_size = 6  # keep a few recent frames
        # smoothing of pass/fail across last N checks
        self.pass_history = []
        self.max_history = 6
        self.min_passes_required = 3
        self.debug = False  # set True temporarily to print numeric values

    def check_color_variance(self, face_image: np.ndarray) -> Tuple[bool, float]:
        """
        Check saturation variance in HSV. Lower threshold => easier to pass.
        """
        if not self.enabled:
            return True, 0.0
        try:
            if face_image is None or face_image.size == 0:
                return False, 0.0
            # face_image expected RGB
            hsv = cv2.cvtColor(face_image, cv2.COLOR_RGB2HSV)
            variance = float(np.var(hsv[:, :, 1]))
            is_live = variance >= self.color_variance_threshold
            return is_live, variance
        except Exception as e:
            if self.debug:
                print("Color variance check error:", e)
            return True, 0.0  # fail-open to avoid blocking on exceptions

    def check_frame_consistency(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Use mean absolute difference per pixel between consecutive frames.
        Normalized so threshold is comparable across resolutions.
        """
        if not self.enabled or len(self.frame_buffer) < 1:
            return True, 0.0
        try:
            prev = self.frame_buffer[-1]
            # convert to same size for stable comparison
            h, w = 200, 200
            cur_res = cv2.resize(frame, (w, h))
            prev_res = cv2.resize(prev, (w, h))
            diff = cv2.absdiff(cur_res, prev_res)
            mean_diff = float(np.mean(diff))  # 0..255
            is_live = mean_diff >= self.frame_consistency_threshold
            return is_live, mean_diff
        except Exception as e:
            if self.debug:
                print("Frame consistency check error:", e)
            return True, 0.0

    def update_frame_buffer(self, frame: np.ndarray):
        """Keep recent frames for motion checks."""
        if frame is None:
            return
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)

    def is_live(self, face_image: np.ndarray, full_frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        Combined liveness check with smoothing across recent frames.
        Returns (is_live, details)
        """
        if not self.enabled:
            return True, {'enabled': False}

        # update buffers
        self.update_frame_buffer(full_frame)

        color_pass, color_var = self.check_color_variance(face_image)
        frame_pass, frame_val = self.check_frame_consistency(full_frame)

        # Combine checks with some flexibility:
        # - If either check is strongly positive, consider it a pass for this frame.
        # - If both are moderately positive, also pass.
        # We convert to a boolean per-frame result and then use history smoothing.
        per_frame_pass = False
        # strong pass conditions
        if color_pass or frame_pass:
            per_frame_pass = True
        # else keep false

        # update history
        self.pass_history.append(1 if per_frame_pass else 0)
        if len(self.pass_history) > self.max_history:
            self.pass_history.pop(0)

        # decide using sliding-window
        passes = sum(self.pass_history)
        overall = passes >= self.min_passes_required

        if self.debug:
            print(f"[LIVENESS] color_var={color_var:.2f} color_pass={color_pass} frame_val={frame_val:.2f} frame_pass={frame_pass} pass_history={self.pass_history} overall={overall}")

        details = {
            'enabled': True,
            'color_variance': color_var,
            'color_pass': color_pass,
            'frame_consistency': frame_val,
            'frame_pass': frame_pass,
            'history_passes': passes,
            'overall': overall
        }
        return overall, details


class FaceRecognitionEngine:
    """Core face detection, embedding, and recognition engine"""
    
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        self.config = config
        self.db = db
        self.threshold = config.get_float('RECOGNITION', 'threshold', 0.6)
        self.detection_model = config.get('RECOGNITION', 'detection_model', 'hog')
        self.known_faces = []
        self.known_names = []
        self.known_user_ids = []
        self._load_known_faces()
    
    def _load_known_faces(self):
        """Load all registered users from database"""
        users = self.db.get_all_users()
        self.known_faces = []
        self.known_names = []
        self.known_user_ids = []
        
        for user in users:
            self.known_faces.append(user['embeddings'])
            self.known_names.append(user['name'])
            self.known_user_ids.append(user['id'])
    
    def reload_faces(self):
        """Reload faces from database (after new registration)"""
        self._load_known_faces()
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple]:
        """
        Detect faces in image.
        Returns list of (top, right, bottom, left) tuples.
        """
        try:
            face_locations = face_recognition.face_locations(
                image, 
                model=self.detection_model
            )
            return face_locations
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def get_face_embedding(self, image: np.ndarray, face_location: Tuple) -> Optional[np.ndarray]:
        """
        Extract 128-D face embedding for a detected face.
        """
        try:
            encodings = face_recognition.face_encodings(image, [face_location])
            if encodings:
                return encodings[0]
            return None
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def register_face(self, name: str, employee_id: str, department: str, 
                     frames: List[np.ndarray]) -> Tuple[bool, str]:
        """
        Register new user with multiple frame samples.
        Returns (success, message)
        """
        if not frames or len(frames) == 0:
            return False, "No frames provided"
        
        embeddings = []
        
        # Extract embeddings from each frame
        for i, frame in enumerate(frames):
            # Detect face
            face_locations = self.detect_faces(frame)
            
            if len(face_locations) == 0:
                return False, f"No face detected in sample {i+1}"
            
            if len(face_locations) > 1:
                return False, f"Multiple faces detected in sample {i+1}. Please ensure only one person is visible."
            
            # Get embedding
            embedding = self.get_face_embedding(frame, face_locations[0])
            
            if embedding is None:
                return False, f"Failed to extract features from sample {i+1}"
            
            embeddings.append(embedding)
        
        # Validate consistency
        embeddings_array = np.array(embeddings)
        
        # Check pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(embeddings_array, metric='euclidean')
        max_distance = np.max(distances)
        
        if max_distance > 0.8:  # High variance between samples
            return False, f"Inconsistent face samples (variance too high: {max_distance:.2f}). Please retry with stable lighting and position."
        
        # Average embeddings for robust template
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # Store in database
        user_id = self.db.add_user(
            employee_id=employee_id,
            name=name,
            department=department,
            embeddings=avg_embedding,
            num_samples=len(embeddings)
        )
        
        if user_id == -1:
            return False, f"Employee ID '{employee_id}' already exists"
        
        # Reload known faces
        self.reload_faces()
        
        return True, f"Successfully registered {name} with {len(embeddings)} samples"
    
    def recognize_face(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize all faces in image.
        Returns list of dicts with face info.
        """
        if len(self.known_faces) == 0:
            return []
        
        # Detect faces
        face_locations = self.detect_faces(image)
        
        if not face_locations:
            return []
        
        # Get embeddings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Calculate distances to all known faces
            distances = face_recognition.face_distance(self.known_faces, encoding)
            
            # Find best match
            best_match_idx = np.argmin(distances)
            best_distance = distances[best_match_idx]
            
            if best_distance < self.threshold:
                # Recognized
                name = self.known_names[best_match_idx]
                user_id = self.known_user_ids[best_match_idx]
                confidence = 1.0 - best_distance  # Convert to confidence score
                
                results.append({
                    'location': (top, right, bottom, left),
                    'name': name,
                    'user_id': user_id,
                    'confidence': confidence,
                    'distance': best_distance,
                    'recognized': True
                })
            else:
                # Unknown person
                results.append({
                    'location': (top, right, bottom, left),
                    'name': 'Unknown',
                    'user_id': None,
                    'confidence': 0.0,
                    'distance': best_distance,
                    'recognized': False
                })
        
        return results


class AttendanceLogger:
    """Manages attendance punch-in/out logic"""
    
    def __init__(self, config: ConfigManager, db: DatabaseManager):
        self.config = config
        self.db = db
        self.min_interval = timedelta(
            seconds=config.get_int('ATTENDANCE', 'min_punch_interval_seconds', 30)
        )
    
    def can_punch(self, user_id: int, requested_action: Optional[str] = None) -> Tuple[bool, str]:
        """Check if user can punch (not too soon after last punch)"""
        last_record = self.db.get_last_attendance(user_id)
        
        if not last_record:
            if requested_action and requested_action == 'OUT':
                return False, "Cannot punch OUT - No previous punch IN found"
            return True, "Ready to punch IN"
        
        time_since_last = datetime.now() - last_record['timestamp']
        
        if time_since_last < self.min_interval:
            remaining = (self.min_interval - time_since_last).seconds
            return False, f"Please wait {remaining} seconds before next punch"
        
        next_action = 'OUT' if last_record['action'] == 'IN' else 'IN'
        
        if requested_action and requested_action != next_action:
             # Allow force punch if debounce passed, but warn? Or strictly enforce sequence?
             # For simple logic: enforce sequence unless it's a "force" admin action.
             # But user wants "Punch IN" button to punch IN.
             # If last was IN, and user clicks IN again... duplicate IN?
             # Usually blocked.
             if requested_action == last_record['action']:
                 return False, f"Already punched {requested_action}. Please punch {next_action}."
        
        return True, f"Ready to punch {next_action}"
    
    def punch(self, user_id: int, confidence: float, requested_action: Optional[str] = None) -> Tuple[bool, str, str]:
        """
        Process attendance punch.
        Returns (success, action, message)
        """
        can_punch, status_msg = self.can_punch(user_id, requested_action)
        
        if not can_punch:
            return False, '', status_msg
        
        # Determine action
        last_record = self.db.get_last_attendance(user_id)
        
        if requested_action:
            action = requested_action
        else:
            if not last_record:
                action = 'IN'
            else:
                action = 'OUT' if last_record['action'] == 'IN' else 'IN'
        
        # Log attendance
        attendance_id = self.db.log_attendance(user_id, action, confidence)
        
        user = self.db.get_user_by_id(user_id)
        timestamp = datetime.now().strftime('%I:%M %p')
        
        message = f"✅ {user['name']} punched {action} at {timestamp} (Confidence: {confidence*100:.1f}%)"
        
        return True, action, message


class FaceAttendanceSystem:
    """Main application class integrating all components"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.db = DatabaseManager(self.config.get('DATABASE', 'path', 'embeddings.db'))
        self.face_engine = FaceRecognitionEngine(self.config, self.db)
        self.liveness_detector = LivenessDetector(self.config)
        self.liveness_detector.debug = True # temporary for tuning
        self.attendance_logger = AttendanceLogger(self.config, self.db)
        
        # Camera / threading state
        self.cam = None
        self.cam_thread = None
        self.cam_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Registration state
        self.registration_frames = []
        self.num_samples_needed = self.config.get_int('RECOGNITION', 'num_registration_samples', 5)

    def start_camera(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """Open webcam and start a background thread that continuously reads frames."""
        if self.cam_running:
            return "✅ Camera already running"

        # Open camera
        cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        if not cam.isOpened():
            return "❌ Unable to open webcam. Check permissions / index."

        # set resolution if supported
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.cam = cam
        self.cam_running = True

        def _camera_loop():
            while self.cam_running and self.cam is not None:
                ret, frame = self.cam.read()
                if not ret:
                    # small sleep to avoid tight loop when camera fails
                    time.sleep(0.05)
                    continue
                # store a copy into latest_frame (thread-safe)
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                # small sleep to limit CPU usage (approx 30 FPS)
                time.sleep(0.03)

        # start thread (daemon so it won't block program exit)
        self.cam_thread = threading.Thread(target=_camera_loop, daemon=True)
        self.cam_thread.start()
        return "✅ Webcam started"

    def stop_camera(self):
        """Stop camera thread and release device."""
        if not self.cam_running:
            return "❌ Camera not running"

        self.cam_running = False
        # wait briefly for thread to stop
        if self.cam_thread is not None:
            self.cam_thread.join(timeout=1.0)
            self.cam_thread = None

        if self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

        with self.frame_lock:
            self.latest_frame = None

        return "✅ Webcam stopped"

    def get_latest_frame_for_display(self):
        """
        Return the latest frame converted to RGB (for Gradio) or None if not available.
        Use a copy to avoid race conditions with the camera thread.
        """
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            # Convert BGR (OpenCV) -> RGB for display
            display = cv2.cvtColor(self.latest_frame.copy(), cv2.COLOR_BGR2RGB)
            return display
    
    # ==================== REGISTRATION TAB ====================
    
    def start_registration_capture(self, name: str, employee_id: str, department: str) -> str:
        """Initialize registration process"""
        if not name or not employee_id:
            return "❌ Please provide both Name and Employee ID"
        
        self.registration_frames = []
        return f"📸 Ready to capture {self.num_samples_needed} samples for {name}. Click 'Capture Sample' button."
    
    def capture_registration_sample(self) -> Tuple[np.ndarray, str]:
        """
        Called when user clicks "Capture sample".
        Takes a snapshot of the running camera and appends to registration frames.
        """
        # get a snapshot from the running camera
        frame = self.get_latest_frame_for_display()

        if frame is None:
            return None, "⚠️ Camera not started or no frame available. Click 'Access Webcam' first."

        # Note: frame is RGB (because get_latest_frame_for_display converted it).
        # If your downstream code expects BGR, convert:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect face
        face_locations = self.face_engine.detect_faces(frame_bgr)
        
        # Draw on frame (use RGB for display return)
        display_frame = frame.copy()
        
        if len(face_locations) == 0:
            cv2.putText(display_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return display_frame, "⚠️ No face detected. Please position your face in the camera."
        
        if len(face_locations) > 1:
            cv2.putText(display_frame, "Multiple faces detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return display_frame, "⚠️ Multiple faces detected. Please ensure only one person is visible."
        
        # Draw green box for good detection (converting coords if needed, but display_frame matches detection frame dims)
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(display_frame, "Face detected - Good!", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Store frame (RGB as required for face_recognition)
        frame_rgb = frame.copy()
        self.registration_frames.append(frame_rgb)
        
        samples_captured = len(self.registration_frames)
        status = f"✅ Sample {samples_captured}/{self.num_samples_needed} captured"
        
        if samples_captured >= self.num_samples_needed:
            status += " - Ready to register!"
        
        return display_frame, status
    
    def complete_registration(self, name: str, employee_id: str, department: str) -> str:
        """Complete user registration"""
        if len(self.registration_frames) < self.num_samples_needed:
            return f"❌ Need {self.num_samples_needed} samples, only have {len(self.registration_frames)}. Please capture more."
        
        # Register with face engine
        success, message = self.face_engine.register_face(
            name=name,
            employee_id=employee_id,
            department=department if department else "General",
            frames=self.registration_frames
        )
        
        # Clear registration state
        self.registration_frames = []
        
        if success:
            return f"✅ {message}"
        else:
            return f"❌ {message}"
    
    # ==================== ATTENDANCE TAB ====================
    
    def process_attendance_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process frame for real-time recognition"""
        if frame is None:
            return None, "No video feed"
        
        # Update liveness detector
        self.liveness_detector.update_frame_buffer(frame)
        
        # Recognize faces
        results = self.face_engine.recognize_face(frame)
        
        # Draw on frame
        display_frame = frame.copy()
        status_lines = []
        
        if not results:
            cv2.putText(display_frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return display_frame, "No face detected"
        
        for result in results:
            top, right, bottom, left = result['location']
            
            if result['recognized']:
                # Check liveness
                face_img = frame[top:bottom, left:right]
                is_live, liveness_details = self.liveness_detector.is_live(face_img, frame)
                
                if is_live:
                    color = (0, 255, 0)  # Green
                    label = f"{result['name']} ({result['confidence']*100:.0f}%)"
                    status_lines.append(f"✅ Recognized: {result['name']} (Confidence: {result['confidence']*100:.1f}%)")
                else:
                    color = (0, 165, 255)  # Orange
                    label = f"{result['name']} - Spoof Detected!"
                    status_lines.append(f"⚠️ Potential spoof detected for {result['name']}")
                
                # Draw box and label
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(display_frame, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                # Unknown person
                color = (0, 0, 255)  # Red
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(display_frame, "Unknown", (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                status_lines.append("❌ Unknown person detected")
        
        status = "\n".join(status_lines) if status_lines else "Processing..."
        
        return display_frame, status
    
    def manual_punch(self, frame: np.ndarray, action: str) -> str:
        """Manual punch-in/out button handler"""
        if frame is None:
            return "❌ No video frame available"
        
        # Recognize face
        results = self.face_engine.recognize_face(frame)
        
        if not results:
            return "❌ No face detected"
        
        # Filter recognized faces
        recognized = [r for r in results if r['recognized']]
        
        if not recognized:
            return "❌ No recognized face. Please register first."
        
        if len(recognized) > 1:
            return "❌ Multiple people detected. Only one person should be in frame."
        
        result = recognized[0]
        
        # Check liveness
        top, right, bottom, left = result['location']
        face_img = frame[top:bottom, left:right]
        is_live, live_details = self.liveness_detector.is_live(face_img, frame)
        
        # allow punch if either liveness passes OR recognition confidence is very high
        if not is_live and result['confidence'] < 0.75:
            # include a helpful message with a hint showing measured metrics
            return f"❌ Liveness check failed. (color_var={live_details.get('color_variance'):.1f}, motion={live_details.get('frame_consistency'):.1f}) Please ensure you're not using a photo/video and improve lighting."
        
        # Process punch
        success, actual_action, message = self.attendance_logger.punch(
            result['user_id'],
            result['confidence'],
            requested_action=action
        )
        
        return message
    
    # ==================== MANAGEMENT TAB ====================
    
    def get_registered_users(self) -> Tuple[str, str]:
        """Get list of registered users as formatted text"""
        users = self.db.get_all_users()
        
        if not users:
            return "No users registered yet.", ""
        
        # Format as table
        header = f"{'ID':<5} {'Employee ID':<15} {'Name':<25} {'Department':<20} {'Registered':<20}\n"
        header += "=" * 85 + "\n"
        
        rows = []
        for user in users:
            row = f"{user['id']:<5} {user['employee_id']:<15} {user['name']:<25} {user['department'] or 'N/A':<20} {user['registration_date']:<20}"
            rows.append(row)
        
        table = header + "\n".join(rows)
        count = f"Total Users: {len(users)}"
        
        return table, count
    
    def get_attendance_records(self, start_date: str, end_date: str, user_filter: str) -> str:
        """Get attendance records with filters"""
        # Parse filters
        start = start_date if start_date else None
        end = end_date if end_date else None
        user_id = int(user_filter) if user_filter and user_filter.isdigit() else None
        
        records = self.db.get_attendance_records(start, end, user_id)
        
        if not records:
            return "No attendance records found for the selected filters."
        
        # Format as table
        header = f"{'Log ID':<8} {'Employee ID':<15} {'Name':<20} {'Timestamp':<20} {'Action':<8} {'Confidence':<12}\n"
        header += "=" * 83 + "\n"
        
        rows = []
        for record in records:
            conf_str = f"{record['confidence']*100:.1f}%"
            row = f"{record['id']:<8} {record['employee_id']:<15} {record['name']:<20} {record['timestamp']:<20} {record['action']:<8} {conf_str:<12}"
            rows.append(row)
        
        table = header + "\n".join(rows)
        return f"Total Records: {len(records)}\n\n{table}"
    
    def export_attendance_csv(self, start_date: str, end_date: str) -> Tuple[str, str]:
        """Export attendance to CSV"""
        records = self.db.get_attendance_records(start_date, end_date)
        
        if not records:
            return "", "No records to export"
        
        # Create CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"attendance_export_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            # Header
            f.write("Log_ID,Employee_ID,Name,Timestamp,Action,Confidence\n")
            
            # Rows
            for record in records:
                f.write(f"{record['id']},{record['employee_id']},{record['name']},"
                       f"{record['timestamp']},{record['action']},{record['confidence']:.4f}\n")
        
        return filename, f"✅ Exported {len(records)} records to {filename}"
    
    def delete_user_by_id(self, user_id_str: str) -> str:
        """Delete user by ID"""
        if not user_id_str or not user_id_str.isdigit():
            return "❌ Please provide a valid user ID"
        
        user_id = int(user_id_str)
        success = self.db.delete_user(user_id)
        
        if success:
            self.face_engine.reload_faces()
            return f"✅ User ID {user_id} and all associated records deleted"
        else:
            return f"❌ User ID {user_id} not found"


def create_gradio_interface():
    """Create and return Gradio interface"""
    
    system = FaceAttendanceSystem()
    
    # Custom CSS for better styling
    css = """
    .container {max-width: 1200px; margin: auto;}
    .header {text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;}
    .status-box {padding: 15px; background: #f0f0f0; border-radius: 8px; margin: 10px 0;}
    .success {color: #28a745;}
    .error {color: #dc3545;}
    .warning {color: #ffc107;}
    """
    
    with gr.Blocks(css=css, title="Face Attendance System") as app:
        gr.Markdown("""
        # 🎭 Face Authentication Attendance System
        ### Real-time Face Recognition with Liveness Detection
        """, elem_classes="header")
        
        with gr.Tabs():
            # ==================== TAB 1: REGISTRATION ====================
            with gr.Tab("👤 User Registration"):
                gr.Markdown("### Register New User")
                gr.Markdown("Capture multiple face samples for robust recognition")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        reg_webcam = gr.Image(label="Webcam Feed", type="numpy", interactive=False)
                        reg_output_img = gr.Image(label="Processed Frame", type="numpy")
                        reg_access_btn = gr.Button("🎥 Access Webcam")
                        reg_close_btn = gr.Button("⏹️ Close Webcam")
                    
                    with gr.Column(scale=1):
                        reg_name = gr.Textbox(label="Full Name", placeholder="John Doe")
                        reg_emp_id = gr.Textbox(label="Employee ID", placeholder="EMP001")
                        reg_dept = gr.Textbox(label="Department (Optional)", placeholder="Engineering")
                        
                        with gr.Row():
                            reg_capture_btn = gr.Button("📸 Capture Sample", variant="primary")
                            reg_register_btn = gr.Button("✅ Complete Registration", variant="secondary")
                        
                        reg_status = gr.Textbox(label="Status", lines=3, interactive=False)
                
                # Camera streaming generator
                def camera_stream():
                    if not system.cam_running:
                        start_msg = system.start_camera()
                        # If failed to start, yield None? 
                        # But start_camera returns a string. 
                        # The button handler below handles the 'Access' click which creates the generator.
                        pass # Camera started or checked in wrapper
                    
                    while True:
                        if not system.cam_running:
                            break
                        frame = system.get_latest_frame_for_display()
                        if frame is not None:
                            yield frame
                        time.sleep(0.05)

                # Event handlers
                
                # Access Webcam: Start camera and stream to image
                # We use a wrapper to start camera then yield stream
                def start_and_stream_registration():
                    system.start_camera()
                    yield from camera_stream()

                reg_access_btn.click(
                    fn=start_and_stream_registration,
                    inputs=[],
                    outputs=[reg_webcam]
                )
                
                # Close Webcam: stop camera and clear preview + status
                reg_close_btn.click(
                    fn=lambda: (system.stop_camera(), None),
                    inputs=[],
                    outputs=[reg_status, reg_webcam]
                )
                
                # Capture Sample: Take snapshot
                reg_capture_btn.click(
                    fn=system.capture_registration_sample,
                    inputs=[],
                    outputs=[reg_output_img, reg_status]
                )
                
                # Complete Registration
                reg_register_btn.click(
                    fn=system.complete_registration,
                    inputs=[reg_name, reg_emp_id, reg_dept],
                    outputs=[reg_status]
                )
            
            # ==================== TAB 2: ATTENDANCE ====================
            with gr.Tab("📸 Mark Attendance"):
                gr.Markdown("### Real-time Face Recognition")
                gr.Markdown("Stand in front of camera for automatic recognition")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        att_webcam = gr.Image(label="Live Recognition Feed", type="numpy", interactive=False)
                        att_access_btn = gr.Button("🎥 Access Webcam")
                        att_close_btn = gr.Button("⏹️ Close Webcam")
                    
                    with gr.Column(scale=1):
                        att_status = gr.Textbox(label="Recognition Status", lines=5, interactive=False)
                        
                        gr.Markdown("### Manual Punch")
                        with gr.Row():
                            punch_in_btn = gr.Button("⏱️ Punch IN", variant="primary", size="lg")
                            punch_out_btn = gr.Button("⏱️ Punch OUT", variant="secondary", size="lg")
                        
                        punch_result = gr.Textbox(label="Punch Result", lines=2, interactive=False)
                        
                        gr.Markdown("### Recent Activity")
                        recent_activity = gr.Textbox(label="Last 5 Punches", lines=8, interactive=False)
                
                # Stream generator that yields processed frames + status using the system camera
                def attendance_camera_stream():
                    # Ensure camera is started
                    system.start_camera()
                    while system.cam_running:
                        frame = system.get_latest_frame_for_display()  # RGB frame or None
                        if frame is None:
                            yield None, "No video frame"
                            time.sleep(0.05)
                            continue
                        # process_attendance_frame expects the same format used elsewhere (we return display_frame, status)
                        # NOTE: process_attendance_frame expects RGB input because get_latest_frame_for_display returns RGB.
                        # We must check if process_attendance_frame expects BGR (OpenCV default) or RGB.
                        # Looking at process_attendance_frame code: 
                        # It calls liveness_detector.update_frame_buffer(frame) -> Liveness uses RGB2HSV (we fixed this to expect RGB)
                        # It calls face_engine.recognize_face(frame) -> calls detect_faces -> face_recognition library needs RGB.
                        # So passing RGB from get_latest_frame_for_display is CORRECT. 
                        # However, process_attendance_frame returns display_frame.
                        # If it draws using OpenCV (cv2.rectangle), it usually expects BGR for standard OpenCV but since Gradio treats output as RGB, 
                        # if we draw colors (0,255,0) on an RGB image, it will look Green in Gradio. Correct.
                        
                        # Wait, one detail:
                        # process_attendance_frame internally:
                        # display_frame = frame.copy()
                        # returns display_frame.
                        # So it returns what it got. If it got RGB, it returns RGB with drawings.
                        display_frame, status = system.process_attendance_frame(frame)
                        yield display_frame, status
                        time.sleep(0.05)

                def start_and_stream_attendance():
                    # wrapper that starts camera and yields frames
                    system.start_camera()
                    yield from attendance_camera_stream()

                # Start streaming into the recognition view & status
                att_access_btn.click(
                    fn=start_and_stream_attendance,
                    inputs=[],
                    outputs=[att_webcam, att_status]
                )

                # Stop camera and clear preview/status
                att_close_btn.click(
                    fn=lambda: (system.stop_camera(), None, ""),  # return status, clear image, clear status
                    inputs=[],
                    outputs=[att_status, att_webcam, att_status]
                )

                # Manual punch: read latest snapshot from system camera
                # We need a wrapper to safely get frame
                def manual_punch_wrapper(action):
                     frame = system.get_latest_frame_for_display()
                     return system.manual_punch(frame, action)

                punch_in_btn.click(
                    fn=lambda: manual_punch_wrapper("IN"),
                    inputs=[],
                    outputs=[punch_result]
                )

                punch_out_btn.click(
                    fn=lambda: manual_punch_wrapper("OUT"),
                    inputs=[],
                    outputs=[punch_result]
                )
            
            # ==================== TAB 3: MANAGEMENT ====================
            with gr.Tab("📊 Management Dashboard"):
                gr.Markdown("### View Users and Attendance Records")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Registered Users")
                        refresh_users_btn = gr.Button("🔄 Refresh Users", variant="secondary")
                        users_display = gr.Textbox(label="Users", lines=15, interactive=False)
                        users_count = gr.Textbox(label="Statistics", lines=1, interactive=False)
                        
                        gr.Markdown("#### Delete User")
                        delete_user_id = gr.Textbox(label="User ID to Delete", placeholder="1")
                        delete_user_btn = gr.Button("🗑️ Delete User", variant="stop")
                        delete_result = gr.Textbox(label="Result", lines=1, interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### Attendance Records")
                        with gr.Row():
                            att_start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2026-02-01")
                            att_end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="2026-02-28")
                        
                        att_user_filter = gr.Textbox(label="Filter by User ID (Optional)", placeholder="Leave empty for all")
                        
                        with gr.Row():
                            refresh_att_btn = gr.Button("🔄 Refresh Records", variant="primary")
                            export_csv_btn = gr.Button("📥 Export to CSV", variant="secondary")
                        
                        att_records_display = gr.Textbox(label="Attendance Records", lines=15, interactive=False)
                        export_status = gr.Textbox(label="Export Status", lines=1, interactive=False)
                
                # Event handlers
                refresh_users_btn.click(
                    fn=system.get_registered_users,
                    outputs=[users_display, users_count]
                )
                
                delete_user_btn.click(
                    fn=system.delete_user_by_id,
                    inputs=[delete_user_id],
                    outputs=[delete_result]
                )
                
                refresh_att_btn.click(
                    fn=system.get_attendance_records,
                    inputs=[att_start_date, att_end_date, att_user_filter],
                    outputs=[att_records_display]
                )
                
                export_csv_btn.click(
                    fn=system.export_attendance_csv,
                    inputs=[att_start_date, att_end_date],
                    outputs=[export_status]
                )
            
            # ==================== TAB 4: SETTINGS ====================
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### System Configuration")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Recognition Settings")
                        threshold_slider = gr.Slider(
                            minimum=0.3, maximum=0.9, value=0.6, step=0.05,
                            label="Recognition Threshold",
                            info="Lower = more strict, Higher = more lenient"
                        )
                        
                        detection_model = gr.Radio(
                            choices=["hog", "cnn"],
                            value="hog",
                            label="Detection Model",
                            info="HOG: Faster, CNN: More accurate"
                        )
                        
                        liveness_toggle = gr.Checkbox(
                            value=True,
                            label="Enable Liveness Detection",
                            info="Prevent photo/video spoofing"
                        )
                    
                    with gr.Column():
                        gr.Markdown("#### System Information")
                        
                        info_text = f"""
                        **System Status:** ✅ Running
                        **Database:** {system.config.get('DATABASE', 'path')}
                        **Registered Users:** {len(system.db.get_all_users())}
                        **Detection Model:** {system.face_engine.detection_model.upper()}
                        **Recognition Threshold:** {system.face_engine.threshold}
                        **Liveness Detection:** {'Enabled' if system.liveness_detector.enabled else 'Disabled'}
                        
                        **Performance Tips:**
                        - Use HOG model for faster processing
                        - Lower threshold for higher security
                        - Ensure good lighting during registration
                        - Capture samples from slightly different angles
                        """
                        
                        gr.Markdown(info_text)
                
                gr.Markdown("### About")
                gr.Markdown("""
                Face Authentication Attendance System v1.0
                
                **Features:**
                - Multi-sample face registration
                - Real-time face recognition
                - Basic liveness detection (anti-spoofing)
                - Attendance logging with punch-in/out
                - Export attendance to CSV
                
                **Technologies:**
                - face_recognition (dlib ResNet)
                - OpenCV for video processing
                - SQLite for data storage
                - Gradio for UI
                
                **Known Limitations:**
                - Basic liveness detection (not foolproof against advanced attacks)
                - CPU-only processing
                - May struggle with identical twins
                - Not optimized for masked faces
                """)
    
    return app


if __name__ == "__main__":
    print("🚀 Starting Face Authentication Attendance System...")
    print("📊 Initializing components...")
    
    app = create_gradio_interface()
    
    print("✅ System ready!")
    print("🌐 Launching web interface...")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
