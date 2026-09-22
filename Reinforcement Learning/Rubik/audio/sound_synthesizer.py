"""
Astra-DeepCube: Audio & Futuristic SFX Synthesizer
Generates real-time cybernetic sound effects (whooshes, neural scan pulses, solve fanfare).
"""

import math
import struct
import wave
import io
import threading
from typing import Optional

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


class SoundSynthesizer:
    """
    Pure Python Audio Synthesizer producing Sci-Fi sound effects dynamically.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.enabled = True

    def _play_wave_bytes(self, wave_data: bytes):
        """Plays PCM audio asynchronously."""
        if not self.enabled or not WINSOUND_AVAILABLE or not wave_data:
            return

        def _worker():
            try:
                winsound.PlaySound(wave_data, winsound.SND_MEMORY | winsound.SND_ASYNC)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _create_wav(self, samples: list) -> bytes:
        """Converts raw float samples (-1.0 to 1.0) into WAV binary with NaN/inf guards."""
        buf = io.BytesIO()
        try:
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                sanitized = []
                for s in samples:
                    if math.isnan(s) or math.isinf(s):
                        val = 0
                    else:
                        val = int(max(-1.0, min(1.0, s)) * 32767)
                    sanitized.append(struct.pack("<h", val))
                wf.writeframes(b"".join(sanitized))
            return buf.getvalue()
        except Exception:
            return b""

    def play_whoosh(self):
        """Generates a smooth futuristic slice rotation whoosh sound."""
        duration = 0.08
        num_samples = int(self.sample_rate * duration)
        samples = []
        for i in range(num_samples):
            t = i / self.sample_rate
            freq = 480.0 * (1.0 - t / duration) + 120.0
            env = math.sin(math.pi * (t / duration))
            s = math.sin(2.0 * math.pi * freq * t) * env * 0.3
            samples.append(s)
        self._play_wave_bytes(self._create_wav(samples))

    def play_scan_blip(self):
        """Generates a high-tech neural search node pulse."""
        duration = 0.04
        num_samples = int(self.sample_rate * duration)
        samples = []
        for i in range(num_samples):
            t = i / self.sample_rate
            freq = 900.0 + 300.0 * math.sin(2.0 * math.pi * 50.0 * t)
            env = math.exp(-t * 80.0)
            s = math.sin(2.0 * math.pi * freq * t) * env * 0.2
            samples.append(s)
        self._play_wave_bytes(self._create_wav(samples))

    def play_solve_fanfare(self):
        """Generates a triumphal cybernetic ascending chord."""
        duration = 0.45
        num_samples = int(self.sample_rate * duration)
        samples = []
        chord_freqs = [523.25, 659.25, 783.99, 1046.50]  # C5, E5, G5, C6
        for i in range(num_samples):
            t = i / self.sample_rate
            s = 0.0
            env = math.sin(math.pi * min(1.0, t * 4.0)) * math.exp(-t * 2.5)
            for f in chord_freqs:
                s += math.sin(2.0 * math.pi * f * t)
            samples.append((s / len(chord_freqs)) * env * 0.4)
        self._play_wave_bytes(self._create_wav(samples))
