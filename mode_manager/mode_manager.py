# mode_manager/mode_manager.py
# =============================
# Gestionnaire central des modes de communication

from shared.protocol import InputMode, UserIntent
import time


class ModeManager:
    """
    Décide automatiquement quel mode de communication utiliser
    selon l'activité détectée chez le patient.
    """

    def __init__(self):
        self.current_mode = InputMode.NONE

    def decide_mode(self, voice_active: bool, gesture_active: bool) -> InputMode:
        """
        Priorité médicale :
        1) Voix
        2) Geste
        3) Regard
        """
        if voice_active:
            self.current_mode = InputMode.VOICE
        elif gesture_active:
            self.current_mode = InputMode.GESTURE
        else:
            self.current_mode = InputMode.EYE

        return self.current_mode

    def build_intent(
        self,
        mode: InputMode,
        content: str,
        confidence: float = 1.0
    ) -> UserIntent:
        """
        Construit un message standardisé à destination de l’avatar
        ou de l’interface clinique.
        """
        return UserIntent(
            mode=mode,
            content=content,
            confidence=confidence,
            timestamp=time.time()
        )
