from app.fake_blackhole import FakeBlackHole
from app.audio_buffer_manager import AudioBufferManager


def main():
    bh = FakeBlackHole(
        wav_path="audio/test01_20s.wav",
        frame_size=1024,
        realtime=False,
    )
    bh.load()

    buffer_manager = AudioBufferManager(
        input_sample_rate=bh.sample_rate,
        target_sample_rate=16000,
        window_size_sec=2.0,
        step_size_sec=1.0,
    )

    def on_audio_chunk(chunk):
        windows = buffer_manager.add_chunk(chunk)
        for w in windows:
            print(f"Window produced: shape={w.shape}")

    bh.stream(on_audio_chunk)


if __name__ == "__main__":
    main()
