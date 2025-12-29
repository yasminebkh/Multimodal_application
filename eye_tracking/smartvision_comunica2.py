"""
SmartVision Syndrome d'Enfermement
==================================

Version stabilisée (regard + double clignement)

Améliorations par rapport à la version précédente :
- Lissage du regard plus fort (historique plus long)
- Zone centrale (neutre) élargie → moins de fausses directions
- Double clignement plus strict (moins de validations accidentelles)
- Temps de pré-sélection augmenté (le regard doit rester plus longtemps)
- Aucune validation automatique par simple maintien du regard
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
# INITIALISATION AUDIO
# ============================================================

pygame.mixer.init()

sounds = {}
try:
    sounds = {
        "repas": pygame.mixer.Sound("repas.mp3"),
        "soins": pygame.mixer.Sound("soins.mp3"),
        "wc": pygame.mixer.Sound("wc.mp3"),
        "confort": pygame.mixer.Sound("confort.mp3"),
        "rien": pygame.mixer.Sound("rien.mp3"),
    }
except Exception as e:
    print("[AVERTISSEMENT] Impossible de charger certains sons :", e)

print("\n" + "=" * 70)
print("SMARTVISION - Syndrome d'enfermement - Interface médicale oculaire")
print("=" * 70)

# ============================================================
# CONFIGURATION MEDIAPIPE (FACE MESH + IRIS)
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

print("[CONFIG] MediaPipe FaceMesh initialisé (haute précision)")


# ============================================================
# SYSTEME DE CALIBRATION DU REGARD
# ============================================================

class CalibrationSystem:
    """Gestion complète de la calibration du regard (en pixels)."""

    def __init__(self):
        self.calibration_file = "calibration_data.json"
        self.center_position = None
        self.positions = {
            "centre": [],
            "haut": [],
            "bas": [],
            "gauche": [],
            "droite": [],
        }
        self.is_calibrating = False
        self.current_calibration_step = None
        self.calibration_samples = 30
        self.thresholds = {"x": 10, "y": 5}
        self.load_calibration()

    def load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r") as f:
                    data = json.load(f)
                self.center_position = tuple(data.get("center_position", [0, 0]))
                self.thresholds = data.get("thresholds", {"x": 10, "y": 5})
                print(
                    f"[CALIBRATION] Chargée : centre={self.center_position}, "
                    f"seuils X={self.thresholds['x']} Y={self.thresholds['y']}"
                )
                return True
            except Exception as e:
                print("[ERREUR] Chargement calibration :", e)
        return False

    def save_calibration(self):
        try:
            data = {
                "center_position": list(self.center_position)
                if self.center_position
                else [0, 0],
                "thresholds": self.thresholds,
            }
            with open(self.calibration_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"[CALIBRATION] Sauvegardée dans {self.calibration_file}")
            return True
        except Exception as e:
            print("[ERREUR] Sauvegarde calibration :", e)
            return False

    def start_calibration(self, direction):
        self.is_calibrating = True
        self.current_calibration_step = direction
        self.positions[direction] = []
        print(f"[CALIBRATION] Début étape : {direction.upper()}")

    def add_calibration_sample(self, position):
        if self.is_calibrating and self.current_calibration_step:
            self.positions[self.current_calibration_step].append(position)
            if len(self.positions[self.current_calibration_step]) >= self.calibration_samples:
                return True
        return False

    def finish_calibration_step(self):
        if self.current_calibration_step == "centre":
            positions = self.positions["centre"]
            if positions:
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                self.center_position = (int(avg_x), int(avg_y))
                print(f"[CALIBRATION] Centre calibré : {self.center_position}")

        self.is_calibrating = False
        self.current_calibration_step = None

    def calculate_thresholds(self):
        if not self.center_position or not any(self.positions.values()):
            return

        distances_x, distances_y = [], []

        for direction, positions in self.positions.items():
            if direction == "centre" or not positions:
                continue

            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)

            dx = abs(avg_x - self.center_position[0])
            dy = abs(avg_y - self.center_position[1])

            if direction in ("gauche", "droite"):
                distances_x.append(dx)
            if direction in ("haut", "bas"):
                distances_y.append(dy)

        if distances_x:
            self.thresholds["x"] = int(sum(distances_x) / len(distances_x) * 0.6)
        if distances_y:
            self.thresholds["y"] = int(sum(distances_y) / len(distances_y) * 0.6)

        print(
            f"[CALIBRATION] Seuils calculés : X={self.thresholds['x']} "
            f"Y={self.thresholds['y']}"
        )
        self.save_calibration()


calibration = CalibrationSystem()


# ============================================================
# DETECTION DU REGARD / CLIGNEMENTS
# ============================================================

class ImprovedEyeDetector:
    """Extraction des positions d’iris + détection de clignements."""

    def __init__(self):
        self.LEFT_IRIS = [473]
        self.RIGHT_IRIS = [468]
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # ✅ LISSAGE PLUS FORT
        self.position_history = deque(maxlen=10)
        self.smoothed_position = None

        # ✅ CLIGNEMENT PLUS STRICT
        self.blink_threshold = 0.22
        self.blink_frames = 0
        self.blink_counter = 0
        self.last_blink_time = 0.0
        self.consecutive_blinks_needed = 2  # double clignement
        self.min_blink_gap = 0.35           # délai mini entre deux clignements comptés

    def calculate_eye_aspect_ratio(self, landmarks, eye_points, frame_w, frame_h):
        coords = []
        for idx in eye_points:
            lm = landmarks[idx]
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)
            coords.append((x, y))
        coords = np.array(coords)
        v1 = np.linalg.norm(coords[1] - coords[5])
        v2 = np.linalg.norm(coords[2] - coords[4])
        h = np.linalg.norm(coords[0] - coords[3])
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_blink(self, landmarks, frame_w, frame_h):
        left_ear = self.calculate_eye_aspect_ratio(
            landmarks, self.LEFT_EYE, frame_w, frame_h
        )
        right_ear = self.calculate_eye_aspect_ratio(
            landmarks, self.RIGHT_EYE, frame_w, frame_h
        )
        avg_ear = (left_ear + right_ear) / 2.0
        t = time.time()

        if avg_ear < self.blink_threshold:
            self.blink_frames += 1
        else:
            if self.blink_frames >= 2:
                # ✅ on vérifie le délai minimal entre 2 clignements comptés
                if t - self.last_blink_time > self.min_blink_gap:
                    self.blink_counter += 1
                    self.last_blink_time = t
                    print(f"[BLINK] Clignement #{self.blink_counter}")
            self.blink_frames = 0

        # Remise à zéro si plus de 2 s sans cligner
        if t - self.last_blink_time > 2.0:
            self.blink_counter = 0

        if self.blink_counter >= self.consecutive_blinks_needed:
            self.blink_counter = 0
            return True

        return False

    def get_smoothed_iris_position(self, landmarks, frame_w, frame_h):
        if not landmarks:
            return None

        iris_r = landmarks[self.RIGHT_IRIS[0]]
        iris_l = landmarks[self.LEFT_IRIS[0]]

        xr, yr = int(iris_r.x * frame_w), int(iris_r.y * frame_h)
        xl, yl = int(iris_l.x * frame_w), int(iris_l.y * frame_h)
        x_avg = (xr + xl) // 2
        y_avg = (yr + yl) // 2

        self.position_history.append((x_avg, y_avg))

        # ✅ on attend un peu plus d'échantillons avant de lisser
        if len(self.position_history) >= 6:
            ax = sum(p[0] for p in self.position_history) / len(self.position_history)
            ay = sum(p[1] for p in self.position_history) / len(self.position_history)
            self.smoothed_position = (int(ax), int(ay))
        else:
            self.smoothed_position = (x_avg, y_avg)

        return self.smoothed_position

    def get_direction_from_position(self, current_pos):
        """Retourne 'haut', 'bas', 'gauche', 'droite', 'centre' ou None."""
        if not calibration.center_position or not current_pos:
            return None

        dx = current_pos[0] - calibration.center_position[0]
        dy = current_pos[1] - calibration.center_position[1]

        # ✅ zone centrale volontairement élargie
        tx = calibration.thresholds["x"] * 1.6
        ty = calibration.thresholds["y"] * 1.6

        if abs(dx) < tx and abs(dy) < ty:
            return "centre"
        elif abs(dx) > abs(dy):
            if dx < -tx:
                return "gauche"
            elif dx > tx:
                return "droite"
        else:
            if dy < -ty:
                return "haut"
            elif dy > ty:
                return "bas"
        return None

    def reset_blink_counter(self):
        self.blink_counter = 0
        self.blink_frames = 0


eye_detector = ImprovedEyeDetector()


# ============================================================
# VARIABLES GLOBALES / FPS / VALIDATION 2 ETAPES
# ============================================================

pause_until = 0.0
pause_duration = 0.6

fps_counter = 0
fps_start_time = time.time()
current_fps = 0

# ✅ Validation en 2 étapes : pré-sélection + double clignement
PRESELECT_TIME_REQUIRED = 0.7  # temps min de regard stable avant confirmation

preselected_direction = None
preselect_start_time = 0.0

screen_manager = None
root = None


# ============================================================
# GESTIONNAIRE D'ECRANS AVEC 5 BOUTONS (CROIX + CENTRE)
# ============================================================

class ScreenManager:
    """
    mapping : direction -> (texte_du_bouton, callback)
    direction ∈ {"haut", "bas", "gauche", "droite", "centre"}

    - Le regard sert à pointer (mise en JAUNE).
    - La validation se fait uniquement par DOUBLE CLIGNEMENT,
      si le regard est resté suffisamment longtemps.
    """

    def __init__(self, root, buttons_dict, log_widget, status_label):
        self.root = root
        self.buttons = buttons_dict   # dict direction -> tk.Button
        self.log_widget = log_widget
        self.status_label = status_label

        self.current_screen = "menu_principal"
        self.current_mapping = {}
        self.last_direction = None

        # Consultation
        self.consultation_data = {}
        self.consultation_questions = [
            (
                "Intensité de la douleur",
                [
                    ("Légère", "légère"),
                    ("Modérée", "modérée"),
                    ("Forte", "forte"),
                    ("Insupportable", "insupportable"),
                ],
                "intensite",
            ),
            (
                "Type de douleur",
                [
                    ("Brûlure", "brûlure"),
                    ("Coup de couteau", "coup de couteau"),
                    ("Pression / étau", "pression / étau"),
                    ("Crampes", "crampes"),
                    ("Décharge électrique", "décharge électrique"),
                ],
                "type",
            ),
            (
                "Début de la douleur",
                [
                    ("Brutal", "brutal"),
                    ("Progressif", "progressif"),
                    ("Hier", "hier"),
                    ("Il y a plusieurs jours", "il y a plusieurs jours"),
                ],
                "debut",
            ),
            (
                "Évolution",
                [
                    ("Ça s'aggrave", "qui s'aggrave"),
                    ("Ça s'améliore", "qui s'améliore"),
                    ("Stable", "stable"),
                ],
                "evolution",
            ),
        ]
        self.current_question_idx = 0
        self.confort_page = 1

    # ------------------------ Utilitaires ------------------------

    def log(self, text):
        ts = time.strftime("%H:%M:%S")
        msg = f"[{ts}] {text}\n"
        try:
            self.log_widget.insert(tk.END, msg)
            self.log_widget.see(tk.END)
        except Exception:
            pass
        print(msg.strip())

    def set_mapping(self, mapping, screen_name, status_text):
        self.current_screen = screen_name
        self.status_label.config(text=status_text)
        self.current_mapping = {}
        for direction, btn in self.buttons.items():
            data = mapping.get(direction)
            if data is None:
                btn.config(text="", state=tk.DISABLED, bg="#333333", fg="white")
                btn._callback = None
            else:
                label, callback = data
                btn.config(
                    text=label,
                    state=tk.NORMAL,
                    bg="#333333",
                    fg="white",
                )
                btn._callback = callback
                self.current_mapping[direction] = (label, callback)
        self.highlight_direction(None)

    def highlight_direction(self, direction):
        """Pré-sélection visuelle (JAUNE) du bouton regardé."""
        global preselected_direction, preselect_start_time

        now = time.time()

        if direction != preselected_direction:
            preselected_direction = direction
            preselect_start_time = now

        for d, btn in self.buttons.items():
            if (
                d == preselected_direction
                and btn.cget("state") == tk.NORMAL
                and btn.cget("text")
            ):
                btn.config(bg="gold", fg="black")
            else:
                btn.config(bg="#333333", fg="white")

        self.last_direction = preselected_direction

    def handle_direction(self, direction):
        """
        Appelé en temps réel par l'eye-tracker.
        Si direction None ou hors mapping → aucune sélection (repos).
        """
        if direction in self.current_mapping:
            self.highlight_direction(direction)
        else:
            self.highlight_direction(None)

    def activate_direction(self, direction):
        """Exécute le callback associé à une direction."""
        if direction not in self.current_mapping:
            return
        label, callback = self.current_mapping[direction]
        if not label or callback is None:
            return
        self.log(f"[VALIDE] {label}")
        callback()

    def validate_current_choice(self):
        """
        Double clignement : on valide uniquement si
        - un bouton est pré-sélectionné
        - le regard est resté assez longtemps dessus
        """
        global preselected_direction, preselect_start_time

        now = time.time()

        if preselected_direction in self.current_mapping:
            dwell_time = now - preselect_start_time

            if dwell_time >= PRESELECT_TIME_REQUIRED:
                btn = self.buttons[preselected_direction]
                btn.config(bg="lime", fg="black")
                self.activate_direction(preselected_direction)

                preselected_direction = None
                preselect_start_time = 0.0
                self.highlight_direction(None)
            else:
                self.log(
                    "[SECURITE] Double clignement trop rapide "
                    "(regard pas assez stable) → confirmation refusée"
                )
        else:
            self.log("[INFO] Double clignement sans pré-sélection → aucune action")

    # ============================================================
    # MENU PRINCIPAL
    # ============================================================

    def show_main_menu(self):
        def do_nothing():
            self.log("[INFO] Rien demandé pour le moment.")
            try:
                if "rien" in sounds:
                    sounds["rien"].play()
            except Exception:
                pass
            self.show_main_menu()

        mapping = {
            "haut": ("🩺 Consultation", self.goto_consultation),
            "bas": ("💊 Médicaments", self.show_medicaments),
            "gauche": ("🤲 Besoins", self.show_besoins),
            "droite": ("🛌 Confort", self.show_confort_page1),
            "centre": ("Rien / Ne rien faire", do_nothing),
        }
        self.set_mapping(mapping, "menu_principal", "Menu principal")

    # ============================================================
    # CONSULTATION
    # ============================================================

    def goto_consultation(self):
        self.consultation_data = {}
        self.show_consultation_zone()

    def show_consultation_zone(self):
        mapping = {
            "haut": (
                "Tête / visage",
                lambda: self._set_zone_and_questions("tête / visage"),
            ),
            "bas": (
                "Ventre / abdomen / autre",
                lambda: self._set_zone_and_questions("ventre / abdomen / autre"),
            ),
            "gauche": (
                "Dos / membres\n(bras, mains, jambes, pieds)",
                lambda: self._set_zone_and_questions("dos / membres"),
            ),
            "droite": (
                "Poitrine / thorax / cœur\n+ respiration",
                lambda: self._set_zone_and_questions(
                    "poitrine / thorax / cœur / respiration"
                ),
            ),
            "centre": ("Retour menu principal", self.show_main_menu),
        }
        self.set_mapping(
            mapping, "consultation_zone", "Consultation – Localisation de la douleur"
        )

    def _set_zone_and_questions(self, zone_label):
        self.consultation_data = {}
        self.consultation_data["zone"] = zone_label
        self.log(f"[CONSULTATION] Zone : {zone_label}")
        self.current_question_idx = 0
        self.show_consultation_question()

    def show_consultation_question(self):
        if (
            self.current_question_idx < 0
            or self.current_question_idx >= len(self.consultation_questions)
        ):
            self.finish_consultation()
            return

        question_text, answers, key = self.consultation_questions[
            self.current_question_idx
        ]
        status = f"Consultation – {question_text}"

        ans = answers[:4]
        mapping = {}

        def make_cb(val):
            return lambda: self.answer_consultation_question(key, val)

        dirs = ["haut", "bas", "gauche", "droite"]
        for d, (label, value) in zip(dirs, ans):
            mapping[d] = (label, make_cb(value))

        mapping["centre"] = ("Annuler / Retour menu principal", self.show_main_menu)

        self.set_mapping(mapping, "consultation_questionnaire", status)

    def answer_consultation_question(self, key, value):
        self.consultation_data[key] = value
        self.log(f"[CONSULTATION] {key} = {value}")
        self.current_question_idx += 1
        if self.current_question_idx >= len(self.consultation_questions):
            self.finish_consultation()
        else:
            self.show_consultation_question()

    def finish_consultation(self):
        zone = self.consultation_data.get("zone", "zone inconnue")
        intensite = self.consultation_data.get("intensite", "")
        type_d = self.consultation_data.get("type", "")
        debut = self.consultation_data.get("debut", "")
        evolution = self.consultation_data.get("evolution", "")

        summary = f"Consultation demandée : douleur {zone}"
        if intensite:
            summary += f" {intensite}"
        if type_d:
            summary += f", type {type_d}"
        if debut:
            summary += f", début {debut}"
        if evolution:
            summary += f", {evolution}"
        summary += "."

        self.log(f"[CONSULTATION] {summary}")
        messagebox.showinfo("Résumé consultation", summary)
        self.show_main_menu()

    # ============================================================
    # MEDICAMENTS
    # ============================================================

    def show_medicaments(self):
        mapping = {
            "haut": (
                "J'ai pris mon médicament",
                lambda: self.finish_medic("pris", None),
            ),
            "bas": (
                "J'ai oublié de le prendre",
                lambda: self.finish_medic("oublié", None),
            ),
            "gauche": (
                "Je pense que la dose\nest trop forte",
                lambda: self.finish_medic("dose trop forte", None),
            ),
            "droite": ("Effets secondaires", self.show_medicaments_side_effects),
            "centre": ("Retour menu principal", self.show_main_menu),
        }
        self.set_mapping(mapping, "medicaments", "Médicaments")

    def show_medicaments_side_effects(self):
        mapping = {
            "haut": (
                "Nausées",
                lambda: self.finish_medic("effets secondaires", "nausées"),
            ),
            "bas": (
                "Vertiges",
                lambda: self.finish_medic("effets secondaires", "vertiges"),
            ),
            "gauche": (
                "Somnolence",
                lambda: self.finish_medic("effets secondaires", "somnolence"),
            ),
            "droite": (
                "Autre / palpitations",
                lambda: self.finish_medic(
                    "effets secondaires", "autre effet / palpitations"
                ),
            ),
            "centre": ("Retour médicaments", self.show_medicaments),
        }
        self.set_mapping(
            mapping, "medicaments_effets", "Médicaments – Effets secondaires"
        )

    def finish_medic(self, main_info, effect):
        if main_info == "effets secondaires" and effect:
            summary = (
                f"Médicaments : patient signale des effets secondaires ({effect})."
            )
        elif main_info == "pris":
            summary = "Médicaments : le patient a pris son médicament."
        elif main_info == "oublié":
            summary = "Médicaments : le patient a oublié de prendre son médicament."
        elif main_info == "dose trop forte":
            summary = (
                "Médicaments : le patient pense que la dose est trop forte."
            )
        else:
            summary = "Médicaments : information non spécifiée."

        self.log(f"[MEDICAMENTS] {summary}")
        messagebox.showinfo("Médicaments", summary)
        self.show_main_menu()

    # ============================================================
    # BESOINS
    # ============================================================

    def show_besoins(self):
        def besoin(label, sound_key=None):
            def _cb():
                self.log(f"[BESOIN] {label}")
                if sound_key and sound_key in sounds:
                    try:
                        sounds[sound_key].play()
                    except Exception as e:
                        print("[AUDIO] Erreur son :", e)

            return _cb

        mapping = {
            "haut": ("Faim / repas", besoin("Faim / repas", "repas")),
            "bas": ("Soif", besoin("Soif", "repas")),
            "gauche": (
                "Besoin d'aller aux toilettes",
                besoin("Besoin WC", "wc"),
            ),
            "droite": ("Autres besoins", self.show_besoins_autres),
            "centre": ("Retour menu principal", self.show_main_menu),
        }
        self.set_mapping(mapping, "besoins", "Besoins immédiats")

    def show_besoins_autres(self):
        def besoin(label, sound_key=None):
            def _cb():
                self.log(f"[BESOIN] {label}")
                if sound_key and sound_key in sounds:
                    try:
                        sounds[sound_key].play()
                    except Exception:
                        pass
                self.show_besoins()

            return _cb

        mapping = {
            "haut": (
                "Besoin d'être repositionné",
                besoin("Repositionnement demandé", "confort"),
            ),
            "bas": (
                "Besoin d'aspiration (trachéo)",
                besoin("Besoin d'aspiration (trachéo)", "soins"),
            ),
            "gauche": (
                "Besoin d'aide respiratoire",
                besoin("Besoin d'aide respiratoire", "soins"),
            ),
            "droite": ("Autre besoin", besoin("Autre besoin", None)),
            "centre": ("Retour besoins", self.show_besoins),
        }
        self.set_mapping(mapping, "besoins_autres", "Besoins – Autres")

    # ============================================================
    # CONFORT (2 pages)
    # ============================================================

    def show_confort_page1(self):
        self.confort_page = 1
        mapping = {
            "haut": (
                "Trop de lumière",
                lambda: self._log_confort_and_stay("Trop de lumière"),
            ),
            "bas": (
                "Trop de bruit",
                lambda: self._log_confort_and_stay("Trop de bruit"),
            ),
            "gauche": (
                "Trop froid",
                lambda: self._log_confort_and_stay("Trop froid"),
            ),
            "droite": ("Plus de choix\n(Confort)", self.show_confort_page2),
            "centre": ("Retour menu principal", self.show_main_menu),
        }
        self.set_mapping(mapping, "confort1", "Confort (1/2)")

    def show_confort_page2(self):
        self.confort_page = 2
        mapping = {
            "haut": (
                "Trop chaud",
                lambda: self._log_confort_and_stay("Trop chaud"),
            ),
            "bas": (
                "Fatigue",
                lambda: self._log_confort_and_stay("Fatigue"),
            ),
            "gauche": (
                "Anxiété / stress",
                lambda: self._log_confort_and_stay("Anxiété / stress"),
            ),
            "droite": (
                "Ça va bien (Retour)",
                lambda: self._log_confort_and_menu("Ça va bien"),
            ),
            "centre": ("Retour page précédente", self.show_confort_page1),
        }
        self.set_mapping(mapping, "confort2", "Confort (2/2)")

    def _log_confort_and_stay(self, label):
        self.log(f"[CONFORT] {label}")
        messagebox.showinfo("Confort", label)
        if self.confort_page == 1:
            self.show_confort_page1()
        else:
            self.show_confort_page2()

    def _log_confort_and_menu(self, label):
        self.log(f"[CONFORT] {label}")
        messagebox.showinfo("Confort", label)
        self.show_main_menu()


# ============================================================
# CALIBRATION
# ============================================================

def start_calibration_sequence():
    print("\n" + "=" * 70)
    print("DEMARRAGE DE LA CALIBRATION")
    print("=" * 70)
    print("\nInstructions patient :")
    print("1. CENTRE   : regardez le milieu de l'écran")
    print("2. HAUT     : regardez en haut")
    print("3. BAS      : regardez en bas")
    print("4. GAUCHE   : regardez à gauche")
    print("5. DROITE   : regardez à droite")
    print("\nChaque étape ≈ 30 échantillons (~2 secondes)\n")

    sequence = ["centre", "haut", "bas", "gauche", "droite"]

    def next_step(idx=0):
        if idx < len(sequence):
            direction = sequence[idx]
            calibration.start_calibration(direction)

            def check():
                if not calibration.is_calibrating:
                    root.after(800, lambda: next_step(idx + 1))
                else:
                    root.after(100, check)

            check()
        else:
            calibration.calculate_thresholds()
            print("=" * 70)
            print("CALIBRATION TERMINEE")
            print("=" * 70)
            messagebox.showinfo(
                "Calibration terminée",
                f"Centre : {calibration.center_position}\n"
                f"Seuil X : {calibration.thresholds['x']}\n"
                f"Seuil Y : {calibration.thresholds['y']}",
            )

    next_step()


# ============================================================
# THREAD VIDEO : EYE-TRACKING
# ============================================================

def process_video():
    global pause_until, fps_counter, fps_start_time, current_fps

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # FPS plus stables

    print(
        "\n[WEBCAM] Résolution :",
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "x",
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print("[WEBCAM] FPS cible :", int(cap.get(cv2.CAP_PROP_FPS)))
    print("[INFO] Validation uniquement par DOUBLE CLIGNEMENT")
    print("[INFO] Etape 1 : regard (JAUNE) → Etape 2 : clignement (VERT)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            t = time.time()

            gaze = eye_detector.get_smoothed_iris_position(lms, w, h)

            # MODE CALIBRATION
            if calibration.is_calibrating and gaze:
                n = len(calibration.positions[calibration.current_calibration_step])
                progress = n / calibration.calibration_samples * 100
                cv2.putText(
                    frame,
                    f"CALIBRATION: {calibration.current_calibration_step.upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3,
                )
                cv2.putText(
                    frame,
                    f"Progression: {int(progress)}% ({n}/{calibration.calibration_samples})",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                done = calibration.add_calibration_sample(gaze)
                if done:
                    calibration.finish_calibration_step()

            # MODE NORMAL
            elif gaze:
                if calibration.center_position:
                    cv2.circle(
                        frame,
                        calibration.center_position,
                        6,
                        (255, 0, 255),
                        2,
                    )
                    cv2.line(
                        frame,
                        calibration.center_position,
                        gaze,
                        (255, 255, 0),
                        2,
                    )

                cv2.circle(frame, gaze, 10, (0, 255, 0), -1)
                cv2.circle(frame, gaze, 14, (255, 255, 255), 2)

                direction = eye_detector.get_direction_from_position(gaze)

                # Surbrillance temps réel (pré-sélection JAUNE)
                if screen_manager is not None:
                    root.after(0, lambda d=direction: screen_manager.handle_direction(d))

                # Validation par double clignement
                blink_valid = eye_detector.detect_blink(lms, w, h)
                if blink_valid and t >= pause_until and screen_manager is not None:
                    root.after(0, screen_manager.validate_current_choice)
                    pause_until = t + pause_duration
                    eye_detector.reset_blink_counter()
                    cv2.putText(
                        frame,
                        "VALIDATION (double clignement)",
                        (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                status = f"FPS: {current_fps}"
                if not calibration.center_position:
                    status = "NON CALIBRÉ – Cliquez sur 'CALIBRER LE REGARD'"
                cv2.putText(
                    frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Regard = pré-sélection (JAUNE) | Double clignement = confirmer (VERT)",
                    (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        cv2.imshow("SmartVision - Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# INTERFACE TKINTER
# ============================================================

def build_gui():
    global root, screen_manager

    root = tk.Tk()
    root.title(
        "SmartVision - Interface médicale oculaire (Syndrome d'enfermement)"
    )
    root.geometry("950x700")
    root.configure(bg="#1E1E1E")

    title = tk.Label(
        root,
        text="SmartVision - Communication médicale par le regard",
        font=("Arial", 18, "bold"),
        bg="#1E1E1E",
        fg="#00FF00",
    )
    title.pack(pady=10)

    subtitle = tk.Label(
        root,
        text=(
            "Regard = pré-sélection (JAUNE) | "
            "Double clignement après maintien = validation (VERT)"
        ),
        font=("Arial", 10),
        bg="#1E1E1E",
        fg="#AAAAAA",
    )
    subtitle.pack(pady=2)

    top_frame = tk.Frame(root, bg="#1E1E1E")
    top_frame.pack(pady=10)

    calib_btn = tk.Button(
        top_frame,
        text="🎯 CALIBRER LE REGARD",
        command=start_calibration_sequence,
        width=25,
        height=2,
        bg="#FF6B35",
        fg="white",
        font=("Arial", 12, "bold"),
    )
    calib_btn.pack(side=tk.LEFT, padx=10)

    status_label = tk.Label(
        top_frame,
        text="Menu principal",
        font=("Arial", 12, "bold"),
        bg="#1E1E1E",
        fg="#00FF00",
    )
    status_label.pack(side=tk.LEFT, padx=10)

    # Cadre central : 5 boutons en croix + centre
    content = tk.Frame(root, bg="#1E1E1E")
    content.pack(pady=10, fill=tk.BOTH, expand=True)

    for c in range(3):
        content.grid_columnconfigure(c, weight=1)
    for r in range(3):
        content.grid_rowconfigure(r, weight=1)

    btn_haut = tk.Button(
        content,
        text="",
        bg="#333333",
        fg="white",
        font=("Arial", 14, "bold"),
        height=2,
        width=18,
        wraplength=260,
        justify="center",
    )
    btn_haut.grid(row=0, column=1, pady=15, padx=15, sticky="nsew")

    btn_gauche = tk.Button(
        content,
        text="",
        bg="#333333",
        fg="white",
        font=("Arial", 14, "bold"),
        height=2,
        width=18,
        wraplength=260,
        justify="center",
    )
    btn_gauche.grid(row=1, column=0, pady=15, padx=15, sticky="nsew")

    btn_centre = tk.Button(
        content,
        text="",
        bg="#333333",
        fg="white",
        font=("Arial", 14, "bold"),
        height=2,
        width=18,
        wraplength=260,
        justify="center",
    )
    btn_centre.grid(row=1, column=1, pady=15, padx=15, sticky="nsew")

    btn_droite = tk.Button(
        content,
        text="",
        bg="#333333",
        fg="white",
        font=("Arial", 14, "bold"),
        height=2,
        width=18,
        wraplength=260,
        justify="center",
    )
    btn_droite.grid(row=1, column=2, pady=15, padx=15, sticky="nsew")

    btn_bas = tk.Button(
        content,
        text="",
        bg="#333333",
        fg="white",
        font=("Arial", 14, "bold"),
        height=2,
        width=18,
        wraplength=260,
        justify="center",
    )
    btn_bas.grid(row=2, column=1, pady=15, padx=15, sticky="nsew")

    buttons_dict = {
        "haut": btn_haut,
        "bas": btn_bas,
        "gauche": btn_gauche,
        "droite": btn_droite,
        "centre": btn_centre,
    }

    log_label = tk.Label(
        root,
        text="Journal des messages / résumés",
        font=("Arial", 11, "bold"),
        bg="#1E1E1E",
        fg="#FFFFFF",
    )
    log_label.pack()

    log_widget = scrolledtext.ScrolledText(
        root,
        height=10,
        bg="#111111",
        fg="#DDDDDD",
        insertbackground="white",
        font=("Consolas", 10),
    )
    log_widget.pack(fill=tk.BOTH, padx=10, pady=5)

    global screen_manager
    screen_manager = ScreenManager(root, buttons_dict, log_widget, status_label)
    screen_manager.show_main_menu()

    threading.Thread(target=process_video, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    build_gui()
