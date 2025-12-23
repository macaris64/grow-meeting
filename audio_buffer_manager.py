import numpy as np
from scipy.signal import resample_poly


class AudioBufferManager:
    """
    Collects small PCM audio chunks and produces sliding windows
    suitable for speech-to-text processing.
    """

    def __init__(
        self,
        input_sample_rate: int,
        target_sample_rate: int = 16000,
        window_size_sec: float = 2.0,
        step_size_sec: float = 1.0,
    ):
        self.input_sr = input_sample_rate
        self.target_sr = target_sample_rate

        self.window_size_samples = int(window_size_sec * target_sample_rate)
        self.step_size_samples = int(step_size_sec * target_sample_rate)

        self.buffer = np.zeros((0,), dtype=np.float32)

        self._last_emitted_sample = 0

        print(
            f"[AudioBufferManager] Initialized | "
            f"input_sr={self.input_sr}, target_sr={self.target_sr}, "
            f"window={self.window_size_samples} samples, "
            f"step={self.step_size_samples} samples"
        )

    def _to_mono(self, chunk: np.ndarray) -> np.ndarray:
        """
        Convert (frames, channels) → mono (frames,)
        """
        if chunk.ndim == 1:
            return chunk

        return np.mean(chunk, axis=1)

    def _resample(self, mono_chunk: np.ndarray) -> np.ndarray:
        """
        Resample audio to target sample rate.
        """
        if self.input_sr == self.target_sr:
            return mono_chunk

        gcd = np.gcd(self.input_sr, self.target_sr)
        up = self.target_sr // gcd
        down = self.input_sr // gcd

        return resample_poly(mono_chunk, up, down)

    def add_chunk(self, chunk: np.ndarray) -> list[np.ndarray]:
        """
        Add a new PCM chunk and return zero or more audio windows.

        Returns:
            List of numpy arrays:
            Each array shape = (window_size_samples,)
        """
        mono = self._to_mono(chunk)
        resampled = self._resample(mono)

        self.buffer = np.concatenate([self.buffer, resampled])

        windows: list[np.ndarray] = []

        while (
            len(self.buffer) - self._last_emitted_sample
            >= self.window_size_samples
        ):
            start = self._last_emitted_sample
            end = start + self.window_size_samples

            window = self.buffer[start:end]
            windows.append(window.copy())

            self._last_emitted_sample += self.step_size_samples

        # Optional memory cleanup
        if self._last_emitted_sample > self.window_size_samples:
            self.buffer = self.buffer[self._last_emitted_sample :]
            self._last_emitted_sample = 0

        return windows
