import asyncio

from fake_blackhole import FakeBlackHole
from audio_buffer_manager import AudioBufferManager
from silence_detector import SilenceDetector
from stt_engine import STTEngine
from sentence_builder import SentenceBuilder
from llm_client import LLMClient
from llm_commit_queue import OrderedCommitQueue
from output_manager import OutputManager


async def main():
    # -------------------------
    # Output
    # -------------------------
    output = OutputManager()

    # -------------------------
    # Core pipeline
    # -------------------------
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

    # -------------------------
    # LLM layer
    # -------------------------
    llm = LLMClient()
    commit_queue = OrderedCommitQueue()

    sentence_id = 0
    accumulated_silence_ms = 0
    STEP_MS = 1000

    # 🔑 PENDING TASKS (EN KRİTİK EK)
    pending_tasks: set[asyncio.Task] = set()

    # -------------------------
    # Async LLM worker
    # -------------------------
    async def process_llm(sentence_id: int, sentence: str):
        result = await llm.refine_and_translate(sentence)
        if result:
            commit_queue.add_result(sentence_id, result)

        # Commit in order
        while True:
            ready = commit_queue.pop_ready()
            if not ready:
                break

            ready_id, ready_result = ready

            print(f"🧠 LLM OUTPUT #{ready_id} → {ready_result}")

            output.write_llm(
                sentence_id=ready_id,
                refined=ready_result["refined_en"],
                translated=ready_result["translated"],
            )

    # -------------------------
    # Audio callback
    # -------------------------
    def on_audio_chunk(chunk):
        nonlocal accumulated_silence_ms, sentence_id

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
                sentence_id += 1

                # 1️⃣ RAW hemen göster
                print(f"\n🎯 RAW #{sentence_id} → {final_sentence}")
                output.write_raw(sentence_id, final_sentence)

                # 2️⃣ LLM async fire-and-forget (AMA TRACK EDİLİYOR)
                task = asyncio.create_task(
                    process_llm(sentence_id, final_sentence)
                )
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)

    # -------------------------
    # Run
    # -------------------------
    bh.stream(on_audio_chunk)

    if pending_tasks:
        print(f"[Main] Waiting for {len(pending_tasks)} LLM tasks...")
        await asyncio.wait(pending_tasks, timeout=15)

    output.close()


if __name__ == "__main__":
    asyncio.run(main())
