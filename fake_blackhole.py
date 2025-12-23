import time
from typing import Callable

import numpy as np
import soundfile as sf


class FakeBlackHole:
    """
    Simulates BlackHole virtual audio device by streaming
    a WAV file as realtime PCM audio chunks.
    """

    def __init__(
        self,
        wav_path: str,
        frame_size: int = 1024,
        realtime: bool = True,
    ):
        self.wav_path = wav_path
        self.frame_size = frame_size
        self.realtime = realtime

        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.channels: int | None = None

    def load(self) -> None:
        """Load WAV file into memory."""
        audio, sr = sf.read(self.wav_path, dtype="float32")

        # Ensure shape: (frames, channels)
        if audio.ndim == 1:
            audio = audio[:, None]

        self.audio = audio
        self.sample_rate = sr
        self.channels = audio.shape[1]

        print(
            f"[FakeBlackHole] Loaded WAV: {self.wav_path} | "
            f"sr={self.sample_rate}, channels={self.channels}, "
            f"frames={len(self.audio)}"
        )

    def stream(self, on_audio_chunk: Callable[[np.ndarray], None]) -> None:
        """
        Start streaming audio chunks.

        on_audio_chunk receives numpy.ndarray:
        shape = (frame_size, channels)
        dtype = float32
        """
        if self.audio is None:
            raise RuntimeError("Audio not loaded. Call load() first.")

        total_frames = len(self.audio)
        idx = 0

        print("[FakeBlackHole] Streaming started")

        while idx + self.frame_size <= total_frames:
            chunk = self.audio[idx : idx + self.frame_size]

            on_audio_chunk(chunk)

            idx += self.frame_size

            if self.realtime:
                time.sleep(self.frame_size / self.sample_rate)

        print("[FakeBlackHole] Streaming finished")


if __name__ == "__main__":
    def debug_callback(chunk: np.ndarray):
        print(f"Chunk: shape={chunk.shape}, dtype={chunk.dtype}")

    bh = FakeBlackHole(
        wav_path="audio/test01_20s.wav",
        frame_size=1024,
        realtime=True,
    )

    bh.load()
    bh.stream(debug_callback)
