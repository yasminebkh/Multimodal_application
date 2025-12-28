# voice_transcription/voice_module.py
# ==================================
# Moteur de transcription vocale silencieux (API multimodale)

import threading
import time
import speech_recognition as sr

_recognizer = sr.Recognizer()
_microphone = sr.Microphone()

_last_voice_text = None
_voice_active = False
_lock = threading.Lock()


def _voice_loop():
    global _last_voice_text, _voice_active

    with _microphone as source:
        _recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with _microphone as source:
                _voice_active = False
                audio = _recognizer.listen(
                    source,
                    timeout=1,
                    phrase_time_limit=4
                )

            text = _recognizer.recognize_google(audio, language="fr-FR")

            with _lock:
                _last_voice_text = text
                _voice_active = True

        except sr.WaitTimeoutError:
            _voice_active = False
        except sr.UnknownValueError:
            _voice_active = False
        except Exception:
            _voice_active = False

        time.sleep(0.1)


def start_voice_recognition():
    thread = threading.Thread(target=_voice_loop, daemon=True)
    thread.start()


def is_voice_active():
    return _voice_active


def get_voice_text():
    global _last_voice_text
    with _lock:
        text = _last_voice_text
        _last_voice_text = None
    return text
