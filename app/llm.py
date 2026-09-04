import json
from typing import Any
import asyncio
from openai import AsyncOpenAI
from .config import Settings

SYSTEM_PROMPT = """You are a precise engineering assistant. Answer only from the supplied context when it is relevant. If the context is insufficient, say so. Return JSON with exactly these keys: answer (string), confidence (number from 0 to 1), and citations (array of document names)."""


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(self, question: str, context: str, tool_context: str, temperature: float, top_p: float) -> tuple[dict[str, Any], str]:
        if self.settings.provider == "local":
            return {"answer": self._offline_answer(question, context, tool_context), "confidence": 0.5, "citations": []}, "local"
        primary = self.settings.vllm_model if self.settings.provider == "vllm" else self.settings.openai_model
        base_url = self.settings.vllm_base_url if self.settings.provider == "vllm" else self.settings.openai_base_url
        api_key = "EMPTY" if self.settings.provider == "vllm" else self.settings.openai_api_key
        try:
            return await self._call(base_url, api_key, primary, question, context, tool_context, temperature, top_p), self.settings.provider
        except Exception:
            if not self.settings.openai_api_key or self.settings.provider == "openai":
                raise
            result = await self._call(self.settings.openai_base_url, self.settings.openai_api_key, self.settings.fallback_model, question, context, tool_context, temperature, top_p)
            return result, "fallback-openai"

    async def _call(self, base_url: str, api_key: str, model: str, question: str, context: str, tool_context: str, temperature: float, top_p: float) -> dict[str, Any]:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        prompt = f"Context:\n{context}\n\n{tool_context}\n\nQuestion: {question}"
        last_error = None
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    response_format={"type": "json_object"},
                )
                return json.loads(response.choices[0].message.content or "{}")
            except Exception as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(0.4 * (2 ** attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

    @staticmethod
    def _offline_answer(question: str, context: str, tool_context: str) -> str:
        if tool_context:
            return tool_context
        if not context:
            return "I do not have enough indexed context to answer that question."
        return f"Based on the indexed documents: {context.split(chr(10))[0]}"
