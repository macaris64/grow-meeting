from fake_blackhole import FakeBlackHole
from audio_buffer_manager import AudioBufferManager
from silence_detector import SilenceDetector
from stt_engine import STTEngine
from sentence_builder import SentenceBuilder


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

    silence_detector = SilenceDetector(
        sample_rate=16000,
        silence_threshold=0.01,
        silence_duration_ms=500,
    )

    stt = STTEngine(
        model_size="small",
        device="cpu",
        compute_type="int8",
        language="en",
    )

    builder = SentenceBuilder()

    accumulated_silence_ms = 0
    STEP_MS = 1000

    def on_audio_chunk(chunk):
        nonlocal accumulated_silence_ms
        windows = buffer_manager.add_chunk(chunk)
        for w in windows:
            silence = silence_detector.detect(w)
            if silence["is_silent"]:
                accumulated_silence_ms += STEP_MS
            else:
                accumulated_silence_ms = 0

            segments = stt.transcribe(w)
            final_sentence = builder.add_segments(
                segments=segments,
                silence_ms=accumulated_silence_ms,
            )
            if final_sentence:
                print(f"\n🎯 FINAL SENTENCE → {final_sentence}")

    bh.stream(on_audio_chunk)


if __name__ == "__main__":
    main()
