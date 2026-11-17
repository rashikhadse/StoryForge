import json

import pytest

from storyforge.config import get_settings
from storyforge.llm import ChatResult, LLMClient, StageError, extract_json
from storyforge.mock_llm import BIBLE
from storyforge.schemas import StoryBible
from storyforge.tracing import Tracer


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nHope that helps') == '{"a": 1}'
    assert extract_json('prefix {"a": {"b": 2}} suffix') == '{"a": {"b": 2}}'


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages, tools=None):
        self.seen.append(messages)
        return ChatResult(content=self.replies.pop(0), input_tokens=10, output_tokens=10)


def _client(replies):
    tracer = Tracer("t")
    c = LLMClient(get_settings(provider="mock"), tracer)
    c._backend = ScriptedBackend(replies)
    return c, tracer


def test_repair_fixes_invalid_output():
    c, tracer = _client(["not json at all", json.dumps(BIBLE)])
    out = c.structured([{"role": "user", "content": "x"}], StoryBible, "extract")
    assert out.title == BIBLE["title"]
    assert tracer.metrics.valid_after_repair["extract"] == 1
    # the repair prompt must tell the model what went wrong
    assert "rejected" in c._backend.seen[1][-1]["content"]


def test_custom_validator_triggers_repair():
    bad = dict(BIBLE, beats=[dict(b, order=b["order"] + 1) for b in BIBLE["beats"]])
    c, tracer = _client([json.dumps(bad), json.dumps(BIBLE)])

    def check(b):
        return None if b.beats[0].order == 1 else "beats must start at 1"

    c.structured([{"role": "user", "content": "x"}], StoryBible, "extract", validate=check)
    assert tracer.metrics.valid_after_repair["extract"] == 1


def test_gives_up_after_max_repairs():
    c, tracer = _client(["{}", "{}", "{}"])
    with pytest.raises(StageError):
        c.structured([{"role": "user", "content": "x"}], StoryBible, "extract")
    assert tracer.metrics.failed["extract"] == 1


def test_openai_compatible_path_parses_tool_calls(monkeypatch):
    """The real-provider code path, with the HTTP client faked out."""
    from types import SimpleNamespace as NS

    monkeypatch.setenv("GROQ_API_KEY", "test")
    c = LLMClient(get_settings(provider="groq"), Tracer("t"))
    captured = {}

    def create(**kw):
        captured.update(kw)
        tc = NS(id="c1", function=NS(name="get_character", arguments='{"name": "Meera"}'))
        msg = NS(content=None, tool_calls=[tc])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=100, completion_tokens=20))

    c._backend = NS(chat=NS(completions=NS(create=create)))
    r = c.chat([{"role": "user", "content": "x"}], "write", tools=[{"type": "function"}])
    assert r.tool_calls == [{"id": "c1", "name": "get_character", "arguments": '{"name": "Meera"}'}]
    assert captured["model"] == "llama-3.3-70b-versatile" and captured["tool_choice"] == "auto"
    ev = c.tracer.events[-1]
    assert (ev.input_tokens, ev.output_tokens) == (100, 20) and ev.cost_usd > 0


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        LLMClient(get_settings(provider="gemini"), Tracer("t"))


def test_tool_400_falls_back_to_plain_answer():
    """A provider rejecting a malformed tool call must not crash the writer."""

    class Bad400(Exception):
        status_code = 400

    class Backend:
        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None):
            self.n += 1
            if tools:
                raise Bad400("tool_use_failed")
            return ChatResult(content=json.dumps(BIBLE))

    c = LLMClient(get_settings(provider="mock"), Tracer("t"))
    c._backend = Backend()
    out = c.run_tools([{"role": "user", "content": "x"}], [{"type": "function"}], {}, StoryBible, "write")
    assert out.title == BIBLE["title"]
    assert c.tracer.metrics.tool_calls["_failed_tool_round"] == 1


def test_critic_provider_does_not_inherit_main_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.delenv("CRITIQUE_MODEL", raising=False)
    s = get_settings(provider="groq", critic_provider="gemini")
    assert s.model("write") == "llama-3.3-70b-versatile"
    assert s.model("critique", "gemini") == "gemini-2.5-flash"
