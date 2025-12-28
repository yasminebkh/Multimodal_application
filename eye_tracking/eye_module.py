# eye_tracking/eye_module.py
# ==========================

import cv2
import mediapipe as mp
import threading
import time
import json
import os
from collections import deque

# ============================================================
# VARIABLE PARTAGÉE AVEC main.py
# ============================================================

_last_eye_command = None
_lock = threading.Lock()


def get_eye_command():
    global _last_eye_command
    with _lock:
        cmd = _last_eye_command
        _last_eye_command = None
    return cmd


def _set_eye_command(command: str):
    global _last_eye_command
    with _lock:
        _last_eye_command = command
# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ============================================================
# CALIBRATION SYSTEM (SIMPLIFIÉ – SILENCIEUX)
# ============================================================

class CalibrationSystem:
    def __init__(self):
        self.center_position = None
        self.thresholds = {"x": 25, "y": 20}

    def set_center(self, pos):
        self.center_position = pos

calibration = CalibrationSystem()

# ============================================================
# EYE + BLINK DETECTOR (STABILISÉ)
# ============================================================

class ImprovedEyeDetector:
    def __init__(self):
        self.LEFT_IRIS = [473]
        self.RIGHT_IRIS = [468]
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.position_history = deque(maxlen=10)

        self.blink_threshold = 0.22
        self.blink_frames = 0
        self.blink_counter = 0
        self.last_blink_time = 0.0
        self.min_blink_gap = 0.35

    def calculate_ear(self, landmarks, eye_points, w, h):
        pts = []
        for idx in eye_points:
            lm = landmarks[idx]
            pts.append((int(lm.x * w), int(lm.y * h)))
        pts = np.array(pts)
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hdist = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hdist)

    def detect_blink(self, landmarks, w, h):
        ear_l = self.calculate_ear(landmarks, self.LEFT_EYE, w, h)
        ear_r = self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)
        ear = (ear_l + ear_r) / 2.0
        t = time.time()

        if ear < self.blink_threshold:
            self.blink_frames += 1
        else:
            if self.blink_frames >= 2 and t - self.last_blink_time > self.min_blink_gap:
                self.blink_counter += 1
                self.last_blink_time = t
            self.blink_frames = 0

        if self.blink_counter >= 2:
            self.blink_counter = 0
            return True

        if t - self.last_blink_time > 2:
            self.blink_counter = 0

        return False

    def get_gaze(self, landmarks, w, h):
        ir = landmarks[self.RIGHT_IRIS[0]]
        il = landmarks[self.LEFT_IRIS[0]]
        x = int((ir.x + il.x) / 2 * w)
        y = int((ir.y + il.y) / 2 * h)

        self.position_history.append((x, y))
        if len(self.position_history) >= 6:
            x = int(sum(p[0] for p in self.position_history) / len(self.position_history))
            y = int(sum(p[1] for p in self.position_history) / len(self.position_history))

        return x, y

    def get_direction(self, pos):
        if calibration.center_position is None:
            calibration.set_center(pos)
            return None

        dx = pos[0] - calibration.center_position[0]
        dy = pos[1] - calibration.center_position[1]

        tx = calibration.thresholds["x"]
        ty = calibration.thresholds["y"]

        if abs(dx) < tx and abs(dy) < ty:
            return "centre"
        if abs(dx) > abs(dy):
            return "gauche" if dx < 0 else "droite"
        return "haut" if dy < 0 else "bas"

eye_detector = ImprovedEyeDetector()
