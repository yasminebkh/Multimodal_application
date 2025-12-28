# shared/protocol.py
# ===================
# Langage commun entre tous les modules

from enum import Enum
from dataclasses import dataclass
import time


class InputMode(Enum):
    VOICE = "voice"
    GESTURE = "gesture"
    EYE = "eye"
    NONE = "none"


@dataclass
class UserIntent:
    mode: InputMode
    content: str
    confidence: float
    timestamp: float = time.time()
