# sentence_builder.py

import re
from typing import List, Optional


class SentenceBuilder:
    """
    Builds FINAL sentences from streaming STT segments using silence signals.
    Includes word-level overlap deduplication and punctuation-aware soft splitting.
    """

    def __init__(
        self,
        min_words: int = 3,
        max_words: int = 20,
        silence_finalize_ms: int = 1200,
        silence_candidate_ms: int = 800,
        max_punctuations: int = 2,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.silence_finalize_ms = silence_finalize_ms
        self.silence_candidate_ms = silence_candidate_ms
        self.max_punctuations = max_punctuations

        self._buffer: List[str] = []
        self._last_final_sentence: Optional[str] = None

        print(
            f"[SentenceBuilder] Initialized | "
            f"min_words={self.min_words}, "
            f"max_words={self.max_words}, "
            f"max_punctuations={self.max_punctuations}"
        )

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def add_segments(
        self,
        segments: List[str],
        silence_ms: int,
    ) -> Optional[str]:
        """
        Add new STT text segments and possibly produce a FINAL sentence.
        """
        for segment in segments:
            cleaned = self._clean(segment)
            if not cleaned:
                continue

            words = self._dedup_and_split_words(cleaned)
            if not words:
                continue

            self._buffer.extend(words)

        if self._should_finalize(silence_ms):
            return self._finalize()

        return None

    # -------------------------------------------------
    # FINALIZATION LOGIC
    # -------------------------------------------------

    def _should_finalize(self, silence_ms: int) -> bool:
        word_count = len(self._buffer)

        if word_count == 0:
            return False

        # 1️⃣ Hard silence
        if silence_ms >= self.silence_finalize_ms:
            return True

        # 2️⃣ Soft silence
        if (
            silence_ms >= self.silence_candidate_ms
            and word_count >= self.min_words
        ):
            return True

        # 3️⃣ Fast speaker guard
        if word_count >= self.max_words:
            return True

        # 4️⃣ Punctuation-aware soft split
        if (
            self._punctuation_count() >= self.max_punctuations
            and word_count >= self.min_words
        ):
            return True

        return False

    def _finalize(self) -> Optional[str]:
        if len(self._buffer) < self.min_words:
            self._buffer.clear()
            return None

        sentence = " ".join(self._buffer)
        sentence = self._normalize(sentence)

        self._last_final_sentence = sentence
        self._buffer.clear()

        return sentence

    # -------------------------------------------------
    # WORD-LEVEL DEDUP
    # -------------------------------------------------

    def _dedup_and_split_words(self, new_text: str) -> List[str]:
        """
        Remove word-level overlap between existing buffer and new text.
        Only removes suffix/prefix overlaps (safe for realtime).
        """
        new_words = new_text.split()

        if not self._buffer:
            return new_words

        max_overlap = min(5, len(self._buffer), len(new_words))

        for k in range(max_overlap, 0, -1):
            if self._buffer[-k:] == new_words[:k]:
                return new_words[k:]

        return new_words

    # -------------------------------------------------
    # CLEAN / NORMALIZE
    # -------------------------------------------------

    def _clean(self, text: str) -> str:
        text = text.strip()

        # Drop punctuation-only or junk tokens
        if not re.search(r"[a-zA-Z0-9]", text):
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text

    def _normalize(self, text: str) -> str:
        text = text.strip()

        if not text:
            return text

        # Capitalize first letter
        text = text[0].upper() + text[1:]

        # Ensure sentence-ending punctuation
        if not text.endswith((".", "!", "?")):
            text += "."

        return text

    # -------------------------------------------------
    # PUNCTUATION
    # -------------------------------------------------

    def _punctuation_count(self) -> int:
        """
        Count sentence-ending punctuations in buffer.
        """
        text = " ".join(self._buffer)
        return len(re.findall(r"[.!?]", text))
