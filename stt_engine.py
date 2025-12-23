# stt_engine.py

from typing import List
import numpy as np
from faster_whisper import WhisperModel


class STTEngine:
    """
    Stateless Speech-to-Text engine wrapper.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        self.language = language

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print(
            f"[STTEngine] Loaded model={model_size}, "
            f"device={device}, compute_type={compute_type}"
        )

    def transcribe(self, audio: np.ndarray) -> List[str]:
        """
        Transcribe a mono 16kHz audio window.

        Returns:
            List of text segments.
        """
        if audio.ndim != 1:
            raise ValueError("Audio must be mono (1D numpy array)")

        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )

        texts: List[str] = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)

        return texts
