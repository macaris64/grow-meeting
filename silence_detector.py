import numpy as np


class SilenceDetector:
    """
    Detects silence based on RMS energy of the audio signal.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.01,
        silence_duration_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms

        self.silence_samples = int(
            (silence_duration_ms / 1000) * sample_rate
        )

        print(
            f"[SilenceDetector] Initialized | "
            f"threshold={self.silence_threshold}, "
            f"tail={self.silence_duration_ms}ms"
        )

    def _rms(self, signal: np.ndarray) -> float:
        """
        Compute RMS energy of a signal.
        """
        return np.sqrt(np.mean(np.square(signal)))

    def detect(self, audio_window: np.ndarray) -> dict:
        """
        Analyze the tail of an audio window and detect silence.

        Returns:
            {
                "is_silent": bool,
                "rms": float
            }
        """
        if audio_window.ndim != 1:
            raise ValueError("audio_window must be mono (1D array)")

        if len(audio_window) < self.silence_samples:
            raise ValueError("audio_window shorter than silence tail")

        tail = audio_window[-self.silence_samples :]

        rms = self._rms(tail)

        is_silent = rms < self.silence_threshold

        return {
            "is_silent": is_silent,
            "rms": rms,
        }
