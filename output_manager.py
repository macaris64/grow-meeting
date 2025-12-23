import os
import json
import time
from dotenv import load_dotenv

load_dotenv()


class OutputManager:
    def __init__(self):
        self.format = os.getenv("OUTPUT_FORMAT", "FORMAT_FILE")
        self.path = os.getenv("OUTPUT_PATH", "output/transcript.jsonl")

        print(f"[OutputManager] Initialized | format={self.format}")

        if self.format == "FORMAT_FILE":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")
        else:
            raise NotImplementedError(
                f"Output format not supported yet: {self.format}"
            )

    def write_raw(self, sentence_id: int, text: str):
        record = {
            "type": "raw",
            "sentence_id": sentence_id,
            "text": text,
            "timestamp": time.time(),
        }
        self._write(record)

    def write_llm(self, sentence_id: int, refined: str, translated: str):
        record = {
            "type": "llm",
            "sentence_id": sentence_id,
            "refined_en": refined,
            "translated": translated,
            "timestamp": time.time(),
        }
        self._write(record)

    def _write(self, record: dict):
        if self.format == "FORMAT_FILE":
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()

    def close(self):
        if hasattr(self, "_file"):
            self._file.close()
