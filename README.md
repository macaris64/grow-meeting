# 🎙️ Realtime AI Meeting Transcript Engine

This project is a **realtime audio-to-text meeting transcription engine** with **async AI enrichment** (refinement + translation).
It is designed as an MVP-quality core for products like **AI meeting assistants**, **live captioning tools**, or **cross-language collaboration apps**.

The system is built to be:

* 🔁 Realtime-safe (non-blocking)
* 🧠 AI-enhanced (LLM-based refinement & translation)
* 🧩 Modular & extensible
* ⚡ Low-latency by design

---

## ✨ Key Features

* **Realtime audio ingestion**

  * Simulated via `FakeBlackHole` (WAV-based streaming)
  * Ready for system audio tools like BlackHole / Zoom

* **Streaming Speech-to-Text**

  * Sliding windows (2s window / 1s step)
  * Silence-aware segmentation
  * Word-level deduplication
  * Punctuation-aware sentence splitting

* **Async LLM Enrichment**

  * Grammar & clarity refinement
  * Translation to target language
  * Fire-and-forget async calls
  * Order-guaranteed output (no race conditions)

* **Pluggable Output Layer**

  * File-based output (`JSONL`) ✔️
  * SQLite (planned)
  * WebSocket (planned)

* **Production-grade async lifecycle**

  * No retries (latency-first)
  * Timeout-safe
  * Graceful shutdown with pending task tracking

---

## 🧠 Architecture Overview

```text
Audio Stream
    ↓
AudioBufferManager
    ↓
SilenceDetector
    ↓
STT Engine (Whisper)
    ↓
SentenceBuilder
    ↓
RAW Transcript  ──────────────▶ Output (immediate)
       │
       └── Async LLM Enrichment ─▶ OrderedCommitQueue ─▶ Output (delayed)
```

**Key principle:**

> Realtime pipeline is never blocked by AI calls.

---

## 📂 Project Structure

```text
.
├── main.py                  # Application entrypoint
├── fake_blackhole.py        # WAV-based audio stream simulator
├── audio_buffer_manager.py  # Sliding window audio buffer
├── silence_detector.py      # Silence detection logic
├── stt_engine.py            # Speech-to-text (Whisper)
├── sentence_builder.py      # Sentence segmentation & cleanup
├── llm_client.py            # Async LLM client
├── llm_commit_queue.py      # Order-guaranteed async commit
├── output_manager.py        # Pluggable output abstraction
├── output/                 # Generated transcripts (ignored by git)
├── audio/                  # Test audio (optional)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment

Copy example config:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# LLM
LLM_MODEL=gpt-4o-mini
LLM_TARGET_LANG=tr
LLM_TIMEOUT_SEC=3.0

# Output
OUTPUT_FORMAT=FORMAT_FILE
OUTPUT_PATH=output/transcript.jsonl
```

---

## ▶️ Run

```bash
python main.py
```

Expected behavior:

* RAW transcript sentences are printed immediately
* LLM-refined & translated sentences appear shortly after
* All output is written to `output/transcript.jsonl`

Example output:

```json
{"type":"raw","sentence_id":3,"text":"Who will I be today?","timestamp":...}
{"type":"llm","sentence_id":3,"refined_en":"Who will I be today?","translated":"Bugün kim olacağım?","timestamp":...}
```

---

## 📤 Output Formats

Currently supported:

* ✅ `FORMAT_FILE` (JSON Lines)

Planned:

* ⏳ `FORMAT_SQLITE`
* ⏳ `FORMAT_WEBSOCKET`

Switching formats does **not** require changing core logic.

---

## 🧪 Design Decisions

* **No retries for LLM calls**
  Latency is prioritized over completeness in realtime scenarios.

* **Order-guaranteed async processing**
  LLM responses may arrive out-of-order; output is always consistent.

* **SentenceBuilder before LLM**
  LLM is used for quality, not structure.

---

## 🚀 Possible Extensions

* Live WebSocket UI
* Zoom / system audio integration
* Speaker diarization
* Meeting summaries & action items
* Multi-language input detection
* Cost & latency analytics

---

## 📌 Status

✅ Core MVP complete
🔧 Actively extensible
🎯 Ready for productization or portfolio use

---

## 📄 License

MIT (or your preferred license)
