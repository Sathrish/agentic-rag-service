"""Thin LLM wrapper over any OpenAI-compatible chat API (Groq by default), using the OpenAI SDK.

The agent only needs two calls, ``text`` and ``json``, so tests can swap in a fake.
"""
import json
import re

from . import settings

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(raw: str) -> dict:
    """Parse the first JSON object in ``raw`` (models sometimes wrap it in prose or fences)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = _JSON_BLOCK.search(raw or "")
        if not match:
            raise ValueError(f"No JSON object in model output: {raw!r:.200}")
        return json.loads(match.group(0))


class ChatLLM:
    def __init__(self, model=settings.LLM_MODEL, base_url=settings.LLM_BASE_URL,
                 api_key=settings.LLM_API_KEY, temperature=settings.LLM_TEMPERATURE,
                 reasoning_effort=settings.LLM_REASONING_EFFORT):
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY (or LLM_API_KEY) to call the LLM. See .env.example.")
        # The plain OpenAI SDK (not langchain-openai) keeps the service ~30 MB lighter on 512 MB hosts.
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key, max_retries=4, timeout=60)
        self._extra = {}
        # gpt-oss models "think" before answering; low effort cuts hidden reasoning tokens ~20x.
        if reasoning_effort and "gpt-oss" in model:
            self._extra["reasoning_effort"] = reasoning_effort
        self.calls = 0
        self.tokens = 0

    def _invoke(self, system, user, json_mode=False):
        self.calls += 1
        kwargs = dict(self._extra)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        if resp.usage is not None:
            self.tokens += resp.usage.total_tokens or 0
        return resp.choices[0].message.content or ""

    def text(self, system: str, user: str) -> str:
        return self._invoke(system, user).strip()

    def json(self, system: str, user: str) -> dict:
        raw = self._invoke(system + "\nRespond with a JSON object only.", user, json_mode=True)
        try:
            return parse_json(raw)
        except ValueError:   # one repair attempt
            raw = self._invoke(system + "\nYour previous reply was not valid JSON. Respond with a JSON object only.", user)
            return parse_json(raw)
