"""One LLM client for every provider, with three things production pipelines need:

1. Retries with exponential backoff on rate limits and transient errors
   (free tiers return HTTP 429 often).
2. `structured()`: get JSON that validates against a Pydantic schema. If the
   first answer fails validation, send the validation errors back to the model
   and ask for a fix ("repair"), up to `max_repair_attempts` times.
3. `run_tools()`: a tool-calling loop. The model may call Python functions
   (e.g. look up a character in the story bible) before giving its answer.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .config import PROVIDERS, Settings
from .tracing import LLMEvent, Tracer, cost_of


class StageError(RuntimeError):
    """Raised when a stage cannot produce valid output even after repair."""


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


# --------------------------------------------------------------------- helpers
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull a JSON object out of a model reply (handles ```json fences and chatter)."""
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text.strip()


def schema_hint(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(), ensure_ascii=False)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    } or getattr(exc, "status_code", None) in {429, 500, 502, 503, 504}


# --------------------------------------------------------------------- client
class LLMClient:
    def __init__(self, settings: Settings, tracer: Tracer, provider: str | None = None):
        self.settings = settings
        self.tracer = tracer
        self.provider = provider or settings.provider
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {self.provider!r}; choose from {list(PROVIDERS)}")
        if self.provider == "mock":
            from .mock_llm import MockChat

            self._backend: Any = MockChat()
        else:
            import os

            from openai import OpenAI

            cfg = PROVIDERS[self.provider]
            key = os.getenv(cfg["key_env"], "")
            if not key and self.provider != "selfhosted":
                raise RuntimeError(
                    f"Set {cfg['key_env']} in your environment or .env file "
                    f"(or run with LLM_PROVIDER=mock for an offline demo)."
                )
            self._backend = OpenAI(base_url=cfg["base_url"], api_key=key or "none", max_retries=0)

    # ------------------------------------------------------------- raw chat
    def chat(
        self,
        messages: list[dict[str, Any]],
        stage: str,
        tools: list[dict[str, Any]] | None = None,
        kind: str = "llm",
        temperature: float | None = None,
    ) -> ChatResult:
        model = self.settings.model(stage, self.provider)
        temp = self.settings.temperature if temperature is None else temperature

        @retry(
            retry=retry_if_exception(_is_transient),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            stop=stop_after_attempt(self.settings.max_retries),
            reraise=True,
        )
        def _call() -> ChatResult:
            if self.provider == "mock":
                return self._backend.complete(messages, tools=tools)
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temp}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = self._backend.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ]
            usage = resp.usage
            return ChatResult(
                content=msg.content or "",
                tool_calls=calls,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )

        t0 = time.perf_counter()
        ok, result = True, None
        try:
            result = _call()
            return result
        except Exception:
            ok = False
            raise
        finally:
            r = result or ChatResult("")
            self.tracer.record(
                LLMEvent(
                    stage=stage,
                    model=model,
                    provider=self.provider,
                    input_tokens=r.input_tokens,
                    output_tokens=r.output_tokens,
                    latency_s=round(time.perf_counter() - t0, 3),
                    cost_usd=cost_of(model, r.input_tokens, r.output_tokens),
                    kind=kind,
                    ok=ok,
                ),
                messages=messages,
                output=r.content or r.tool_calls,
            )

    # ------------------------------------------------------------- structured
    def _parse(
        self, text: str, schema: type[BaseModel], validate: Callable[[BaseModel], str | None] | None
    ) -> tuple[BaseModel | None, str]:
        raw = extract_json(text)
        try:
            obj = schema.model_validate_json(raw)
        except ValidationError as e:
            errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:12])
            return None, f"Schema validation failed: {errs}"
        except ValueError as e:
            return None, f"Invalid JSON: {e}"
        if validate:
            problem = validate(obj)
            if problem:
                return None, f"Content check failed: {problem}"
        return obj, ""

    def structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        stage: str,
        validate: Callable[[BaseModel], str | None] | None = None,
        first_reply: ChatResult | None = None,
        temperature: float | None = None,
    ) -> BaseModel:
        """Return a validated `schema` instance, repairing the output if needed."""
        m = self.tracer.metrics
        m.structured_calls[stage] += 1
        reply = first_reply or self.chat(messages, stage, temperature=temperature)
        obj, err = self._parse(reply.content, schema, validate)
        if obj is not None:
            m.first_pass_valid[stage] += 1
            return obj

        convo = list(messages)
        for _ in range(self.settings.max_repair_attempts):
            convo += [
                {"role": "assistant", "content": reply.content or "(empty)"},
                {
                    "role": "user",
                    "content": (
                        f"TASK: repair\nYour previous answer was rejected. {err}\n"
                        f"Return ONLY the corrected JSON object, no prose, matching this JSON schema:\n"
                        f"{schema_hint(schema)}"
                    ),
                },
            ]
            reply = self.chat(convo, stage, kind="repair", temperature=0.2)
            obj, err = self._parse(reply.content, schema, validate)
            if obj is not None:
                m.valid_after_repair[stage] += 1
                return obj
        m.failed[stage] += 1
        raise StageError(f"[{stage}] could not produce valid {schema.__name__}: {err}")

    # ------------------------------------------------------------- tool loop
    def run_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        impls: dict[str, Callable[..., str]],
        schema: type[BaseModel],
        stage: str,
        validate: Callable[[BaseModel], str | None] | None = None,
    ) -> BaseModel:
        """Let the model call tools for up to `max_tool_rounds`, then parse its final answer."""
        convo = list(messages)
        for _ in range(self.settings.max_tool_rounds):
            try:
                reply = self.chat(convo, stage, tools=tools)
            except Exception as e:
                # Some providers reject a malformed tool call with HTTP 400
                # (Groq: "tool_use_failed"). Degrade to answering without tools.
                if getattr(e, "status_code", None) != 400:
                    raise
                self.tracer.metrics.tool_calls["_failed_tool_round"] += 1
                break
            if not reply.tool_calls:
                return self.structured(convo, schema, stage, validate, first_reply=reply)
            convo.append(
                {
                    "role": "assistant",
                    "content": reply.content or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                self.tracer.metrics.tool_calls[call["name"]] += 1
                fn = impls.get(call["name"])
                try:
                    args = json.loads(call["arguments"] or "{}")
                    out = fn(**args) if fn else f"Unknown tool {call['name']}"
                except Exception as e:  # tool errors go back to the model, not up the stack
                    out = f"Tool error: {e}"
                convo.append({"role": "tool", "tool_call_id": call["id"], "content": str(out)})
        # Tool budget used up: ask for the final answer without tools.
        convo.append({"role": "user", "content": "Tool budget used up. Now return ONLY the final JSON object."})
        return self.structured(convo, schema, stage, validate)
