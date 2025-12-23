from app.fake_blackhole import FakeBlackHole
from app.audio_buffer_manager import AudioBufferManager
from app.stt_engine import STTEngine


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

    stt = STTEngine(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="en",
    )

    window_index = 0

    def on_audio_chunk(chunk):
        nonlocal window_index

        windows = buffer_manager.add_chunk(chunk)
        for w in windows:
            window_index += 1
            texts = stt.transcribe(w)

            print(f"\n[STT] Window #{window_index}")
            for t in texts:
                print(f"  → {t}")

    bh.stream(on_audio_chunk)


if __name__ == "__main__":
    main()
