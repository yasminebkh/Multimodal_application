# main.py
# =======
# Point d'entrée global du système multimodal SmartVision

from mode_manager.mode_manager import ModeManager
from shared.protocol import InputMode
from eye_tracking.eye_module import start_eye_tracking, get_eye_command
from voice_transcription.voice_module import (
    start_voice_recognition,
    is_voice_active,
    get_voice_text
)


# --------------------------------------------------
# STUBS TEMPORAIRES (seront remplacés par les vrais)
# --------------------------------------------------



def is_gesture_active():
    # À remplacer par le module gesture
    return False


def get_gesture_command():
    return ""


def avatar_react(intent):
    # À remplacer par le vrai module avatar
    print(f"[AVATAR] Mode={intent.mode.value} | Message={intent.content}")


# --------------------------------------------------
# BOUCLE PRINCIPALE
# --------------------------------------------------

def main_loop():
    manager = ModeManager()

    # Démarrage du module eye-tracking (thread interne)
    start_eye_tracking()
    start_voice_recognition()

    print("SmartVision Multimodal System started")

    # 🔒 Mémoire de la dernière commande EYE (anti-répétition)
    last_eye_content = None

    while True:
        voice_active = is_voice_active()
        gesture_active = is_gesture_active()

        mode = manager.decide_mode(
            voice_active=voice_active,
            gesture_active=gesture_active
        )

        content = ""

        if mode == InputMode.VOICE:
            content = get_voice_text()

        elif mode == InputMode.GESTURE:
            content = get_gesture_command()

        elif mode == InputMode.EYE:
            content = get_eye_command()

            # 🔐 Sécurité médicale :
            # empêcher l'envoi répété de la même commande oculaire
            if content == last_eye_content:
                content = ""

            if content:
                last_eye_content = content

        # Envoi vers l’avatar uniquement si une intention valide existe
        if content:
            intent = manager.build_intent(
                mode=mode,
                content=content,
                confidence=1.0
            )
            avatar_react(intent)

        # ⏱️ Fréquence volontairement lente (sécurité médicale)
        import time
        time.sleep(0.5)


if __name__ == "__main__":
    main_loop()
