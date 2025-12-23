# llm_client.py

import asyncio
import os
from typing import Optional, Dict

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env once
load_dotenv()


class LLMClient:
    """
    Async LLM client for sentence refinement and translation.
    Config is loaded from environment variables.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        target_lang: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.target_lang = target_lang or os.getenv("LLM_TARGET_LANG", "tr")
        self.timeout_sec = (
            timeout_sec
            if timeout_sec is not None
            else float(os.getenv("LLM_TIMEOUT_SEC", "1.0"))
        )

        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        print(
            "[LLMClient] Initialized | "
            f"model={self.model}, "
            f"target_lang={self.target_lang}, "
            f"timeout={self.timeout_sec}s"
        )

    async def refine_and_translate(
        self,
        sentence: str,
    ) -> Optional[Dict[str, str]]:
        """
        Returns:
        {
            "refined_en": "...",
            "translated": "..."
        }
        """
        prompt = f"""
You are a professional meeting transcript editor.

Task:
1. Rewrite the sentence in clean, natural English.
2. Remove repetitions or filler words.
3. Translate it into {self.target_lang}.

Return STRICT JSON:
{{
  "refined_en": "...",
  "translated": "..."
}}

Sentence:
\"{sentence}\"
"""

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You edit meeting transcripts.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                ),
                timeout=self.timeout_sec,
            )

            content = response.choices[0].message.content
            return eval(content)  # MVP-safe, trusted JSON

        except Exception as e:
            print("[LLMClient] LLM failed")
            print("Exception type:", type(e))
            print("Exception repr:", repr(e))
            return None
