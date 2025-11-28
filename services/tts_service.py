# services/tts_service.py
import os
import subprocess
import sys


class TTSSpeaker:
    def __init__(self, speaker_script=None):
        if speaker_script is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))  # ia_hackathon/
            speaker_script = os.path.join(base_dir, "speaker.py")
            speaker_script = os.path.abspath(speaker_script)
        self.speaker_script = speaker_script

    def speak(self, text):
        if not text:
            return

        try:
            print(f"[TTS-EXTERNO] {text}")
            subprocess.Popen([sys.executable, self.speaker_script, text])
        except Exception as e:
            print(f"Erro ao chamar speaker externo: {e}")
