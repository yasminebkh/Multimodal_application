"""
SmartVision Care
================
Eye-Based Medical Communication System
Locked-In Syndrome – Clinical Version

Interaction:
- Gaze = pre-selection (yellow)
- Double blink = confirmation (green)
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import pygame
import json
import os
from collections import deque

# ============================================================
# AUDIO INITIALIZATION
# ============================================================

pygame.mixer.init()

sounds = {}
try:
    sounds = {
        "meal": pygame.mixer.Sound("meal.mp3"),
        "drink": pygame.mixer.Sound("drink.mp3"),
        "toilet": pygame.mixer.Sound("toilet.mp3"),
        "comfort": pygame.mixer.Sound("comfort.mp3"),
        "nothing": pygame.mixer.Sound("nothing.mp3"),
    }
except Exception as e:
    print("[WARNING] Audio files missing:", e)

print("=" * 70)
print("SMARTVISION CARE – Eye-Based Medical Communication System")
print("=" * 70)

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
# CALIBRATION SYSTEM
# ============================================================

class CalibrationSystem:
    def __init__(self):
        self.file = "calibration_data.json"
        self.center = None
        self.samples = {"center": [], "up": [], "down": [], "left": [], "right": []}
        self.thresholds = {"x": 10, "y": 5}
        self.current = None
        self.active = False
        self.load()

    def load(self):
        if os.path.exists(self.file):
            with open(self.file, "r") as f:
                data = json.load(f)
                self.center = tuple(data["center"])
                self.thresholds = data["thresholds"]

    def save(self):
        with open(self.file, "w") as f:
            json.dump({"center": self.center, "thresholds": self.thresholds}, f, indent=4)

    def start(self, direction):
        self.current = direction
        self.samples[direction] = []
        self.active = True

    def add(self, pos):
        self.samples[self.current].append(pos)
        return len(self.samples[self.current]) >= 30

    def finish(self):
        if self.current == "center":
            xs = [p[0] for p in self.samples["center"]]
            ys = [p[1] for p in self.samples["center"]]
            self.center = (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))
        self.active = False
        self.current = None

calibration = CalibrationSystem()

# ============================================================
# EYE DETECTOR
# ============================================================

class EyeDetector:
    def __init__(self):
        self.L_IRIS = 473
        self.R_IRIS = 468
        self.history = deque(maxlen=10)
        self.blink_frames = 0
        self.blinks = 0
        self.last_blink = 0

    def get_position(self, lms, w, h):
        x = int((lms[self.L_IRIS].x + lms[self.R_IRIS].x) * w / 2)
        y = int((lms[self.L_IRIS].y + lms[self.R_IRIS].y) * h / 2)
        self.history.append((x, y))
        ax = sum(p[0] for p in self.history) // len(self.history)
        ay = sum(p[1] for p in self.history) // len(self.history)
        return (ax, ay)

    def get_direction(self, pos):
        if not calibration.center:
            return None
        dx = pos[0] - calibration.center[0]
        dy = pos[1] - calibration.center[1]
        if abs(dx) < 1.6*calibration.thresholds["x"] and abs(dy) < 1.6*calibration.thresholds["y"]:
            return "center"
        return "left" if dx < 0 else "right" if abs(dx) > abs(dy) else "up" if dy < 0 else "down"

eye = EyeDetector()

# ============================================================
# GUI SCREEN MANAGER
# ============================================================

class ScreenManager:
    def __init__(self, root, buttons, log, status):
        self.root = root
        self.buttons = buttons
        self.log = log
        self.status = status
        self.mapping = {}
        self.selected = None
        self.start_time = 0

    def set(self, mapping, title):
        self.mapping = mapping
        self.status.config(text=title)
        for d, b in self.buttons.items():
            if d in mapping:
                b.config(text=mapping[d][0], state=tk.NORMAL)
                b.cb = mapping[d][1]
            else:
                b.config(text="", state=tk.DISABLED)

    def highlight(self, d):
        if d != self.selected:
            self.selected = d
            self.start_time = time.time()
        for k, b in self.buttons.items():
            b.config(bg="gold" if k == d else "#333")

    def validate(self):
        if self.selected in self.mapping and time.time() - self.start_time > 0.7:
            self.mapping[self.selected][1]()

# ============================================================
# GUI
# ============================================================

def build_gui():
    root = tk.Tk()
    root.title("SmartVision Care – Medical Eye-Based Communication")
    root.geometry("900x650")
    root.configure(bg="#1e1e1e")

    tk.Label(root, text="SmartVision Care", fg="#00ff88", bg="#1e1e1e",
             font=("Arial", 20, "bold")).pack(pady=10)

    status = tk.Label(root, text="Main Menu", fg="white", bg="#1e1e1e")
    status.pack()

    frame = tk.Frame(root, bg="#1e1e1e")
    frame.pack(expand=True)

    buttons = {}
    for name, r, c in [("up",0,1),("left",1,0),("center",1,1),("right",1,2),("down",2,1)]:
        b = tk.Button(frame, width=18, height=2, bg="#333", fg="white")
        b.grid(row=r, column=c, padx=10, pady=10)
        buttons[name] = b

    log = scrolledtext.ScrolledText(root, height=8)
    log.pack(fill=tk.X, padx=10)

    sm = ScreenManager(root, buttons, log, status)

    def play(k):
        if k in sounds: sounds[k].play()

    sm.set({
        "up": ("Medical Consultation", lambda: messagebox.showinfo("Consultation","Consultation requested")),
        "down": ("Medications", lambda: messagebox.showinfo("Medications","Medication info")),
        "left": ("Basic Needs", lambda: play("meal")),
        "right": ("Comfort", lambda: play("comfort")),
        "center": ("No Action", lambda: play("nothing")),
    }, "Main Menu")

    root.mainloop()

if __name__ == "__main__":
    build_gui()
