"""Deterministic offline LLM used by the tests and `--mock` demo runs.

It returns canned, schema-shaped answers for the bundled sample story so the
whole graph (tool calls, repair, critic loop, dedup, localization) can run with
no API key. On purpose it:
  * returns an invalid story bible first, to exercise the repair path;
  * calls tools before writing, to exercise the tool loop;
  * scores episode 1 low once, to exercise the critic -> writer revision loop;
  * repeats a hook and a dialogue line, to exercise deduplication.
It tests the plumbing, not the writing quality.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

from .llm import ChatResult

BIBLE = {
    "title": "The Potter's Daughter",
    "logline": "A potter's daughter outwits a greedy moneylender who plans to seize her father's wheel, using the one thing he cannot count: her patience.",
    "genre": "folk tale / drama",
    "tone": "warm, tense, quietly triumphant",
    "setting": "A river village in central India, a hundred years ago",
    "characters": [
        {
            "name": "Meera",
            "name_hi": "मीरा",
            "role": "protagonist",
            "description": "Sixteen, sharp-eyed, clay always under her nails; calm when others panic.",
            "relationships": ["daughter of Ramdas"],
        },
        {
            "name": "Ramdas",
            "name_hi": "रामदास",
            "role": "supporting",
            "description": "An ageing potter with failing eyesight, proud and too trusting.",
            "relationships": ["father of Meera", "debtor of Seth Lalchand"],
        },
        {
            "name": "Seth Lalchand",
            "name_hi": "सेठ लालचंद",
            "role": "antagonist",
            "description": "The village moneylender; soft voice, ledger always open, counts everything.",
            "relationships": ["creditor of Ramdas"],
        },
    ],
    "locations": [
        {"name": "Ramdas's courtyard", "description": "Mud courtyard with the potter's wheel under a neem tree."},
        {"name": "Lalchand's shop", "description": "Dark room of brass scales and ledgers by the market."},
        {"name": "Village square", "description": "Open ground by the well where the panchayat meets."},
    ],
    "beats": [
        {
            "order": 1,
            "summary": "Lalchand arrives demanding Ramdas repay his loan by the full moon or lose his wheel.",
            "characters": ["Seth Lalchand", "Ramdas"],
            "emotional_tone": "dread",
        },
        {
            "order": 2,
            "summary": "Meera learns her father signed a paper he could not read.",
            "characters": ["Meera", "Ramdas"],
            "emotional_tone": "betrayal",
        },
        {
            "order": 3,
            "summary": "Meera fires pots through the night to raise money, but rain ruins the kiln.",
            "characters": ["Meera"],
            "emotional_tone": "despair",
        },
        {
            "order": 4,
            "summary": "Meera visits Lalchand's shop and reads the contract aloud, spotting a strange clause.",
            "characters": ["Meera", "Seth Lalchand"],
            "emotional_tone": "suspicion",
        },
        {
            "order": 5,
            "summary": "The clause says the debt is paid if the wheel produces a thousand pots before the full moon.",
            "characters": ["Meera"],
            "emotional_tone": "hope",
        },
        {
            "order": 6,
            "summary": "The village secretly helps Meera shape pots at night.",
            "characters": ["Meera", "Ramdas"],
            "emotional_tone": "solidarity",
        },
        {
            "order": 7,
            "summary": "At the panchayat, Lalchand insists on counting every pot himself.",
            "characters": ["Seth Lalchand", "Meera"],
            "emotional_tone": "tension",
        },
        {
            "order": 8,
            "summary": "The count reaches one thousand and one; the panchayat cancels the debt and Ramdas keeps his wheel.",
            "characters": ["Meera", "Ramdas", "Seth Lalchand"],
            "emotional_tone": "triumph",
        },
    ],
    "themes": ["literacy as power", "patience", "community"],
}


def _usage(messages: list[dict[str, Any]], out: str) -> tuple[int, int]:
    n_in = sum(len(str(m.get("content") or "")) for m in messages) // 4
    return n_in, max(len(out) // 4, 1)


def _task(messages: list[dict[str, Any]]) -> str:
    sys = next((m["content"] for m in messages if m["role"] == "system"), "")
    m = re.search(r"TASK:\s*(\w+)", sys or "")
    return m.group(1) if m else "unknown"


def _num(text: str, key: str, default: int = 1) -> int:
    m = re.search(rf"{key}:\s*(\d+)", text)
    return int(m.group(1)) if m else default


def _user(messages: list[dict[str, Any]]) -> str:
    return next((m["content"] for m in messages if m["role"] == "user"), "")


class MockChat:
    def __init__(self) -> None:
        self.calls: dict[str, int] = defaultdict(int)
        self.critiques: dict[int, int] = defaultdict(int)

    def complete(self, messages: list[dict[str, Any]], tools: list | None = None) -> ChatResult:
        task = _task(messages)
        self.calls[task] += 1
        is_repair = "TASK: repair" in str(messages[-1].get("content", ""))
        fn = getattr(self, f"_{task}", None)
        if fn is None:
            raise ValueError(f"MockChat has no handler for task {task!r}")
        result = fn(messages, tools, is_repair)
        if isinstance(result, ChatResult):
            return result
        out = json.dumps(result, ensure_ascii=False)
        i, o = _usage(messages, out)
        return ChatResult(content=out, input_tokens=i, output_tokens=o)

    # ------------------------------------------------------------- handlers
    def _extract(self, messages, tools, is_repair):
        if not is_repair:  # first answer is deliberately invalid (too few beats)
            bad = dict(BIBLE, beats=BIBLE["beats"][:2])
            return ChatResult(
                content="Here is the bible:\n```json\n" + json.dumps(bad) + "\n```", input_tokens=900, output_tokens=300
            )
        return BIBLE

    def _plan(self, messages, tools, is_repair):
        n = _num(_user(messages), "NUM_EPISODES", 5)
        beats = list(range(1, len(BIBLE["beats"]) + 1))
        size, rem = divmod(len(beats), n)
        eps, start = [], 0
        for e in range(1, n + 1):
            take = size + (1 if e <= rem else 0)
            chunk = beats[start : start + take]
            start += take
            hook = (
                "A fist pounds on the courtyard door at dawn."
                if e in (2, 3)
                else f"Episode {e} opens on a cracked clay pot spinning on the wheel."
            )
            eps.append(
                {
                    "episode": e,
                    "title": f"The Full Moon, Part {e}",
                    "beat_orders": chunk,
                    "hook": hook,
                    "cliffhanger": "Will the count reach a thousand?" if e < n else "The wheel turns again.",
                    "summary": " ".join(BIBLE["beats"][o - 1]["summary"] for o in chunk),
                }
            )
        return {"series_title": "Meera and the Moneylender", "episodes": eps}

    def _write(self, messages, tools, is_repair):
        user = _user(messages)
        m = re.search(r"EPISODE_NUMBER: (\d+) of (\d+)", user)
        n, total = (int(m.group(1)), int(m.group(2))) if m else (1, 5)
        if tools and not any(x["role"] == "tool" for x in messages):
            return ChatResult(
                content="",
                tool_calls=[
                    {"id": f"call_{n}_1", "name": "get_character", "arguments": json.dumps({"name": "Meera"})},
                    {"id": f"call_{n}_2", "name": "get_prior_events", "arguments": json.dumps({"episode": n})},
                ],
                input_tokens=1200,
                output_tokens=40,
            )
        outline = self._plan([{"role": "user", "content": f"NUM_EPISODES: {total}"}], None, False)["episodes"][n - 1]
        scenes = []
        for k, order in enumerate(outline["beat_orders"], start=1):
            beat = BIBLE["beats"][order - 1]
            who = beat["characters"]
            lines = [
                {
                    "id": f"e{n}s{k}l1",
                    "speaker": who[0],
                    "line": f"I can't believe it. {beat['summary']}",
                    "direction": f"({beat['emotional_tone']})",
                },
                {
                    "id": f"e{n}s{k}l2",
                    "speaker": who[-1] if len(who) > 1 else "NARRATOR",
                    "line": f"Then we have until the full moon — night {order}.",
                    "direction": "(quietly)",
                },
            ]
            if n == 1 and k == 1:  # a near-duplicate line for the dedup stage to catch
                lines.append(
                    {
                        "id": f"e{n}s{k}l3",
                        "speaker": who[0],
                        "line": f"I can't believe it! {beat['summary']}",
                        "direction": "",
                    }
                )
            scenes.append(
                {
                    "scene": k,
                    "location": BIBLE["locations"][k % 3]["name"],
                    "time_of_day": "NIGHT" if order % 2 else "DAY",
                    "characters": who,
                    "action": beat["summary"],
                    "dialogue": lines,
                    "visual_prompt": f"Vertical 9:16 cinematic frame, village India 1920s, {beat['summary'].lower()} warm lamplight, film grain",
                }
            )
        return {
            "episode": n,
            "title": outline["title"],
            "hook": outline["hook"],
            "scenes": scenes,
            "cliffhanger": outline["cliffhanger"],
            "est_duration_sec": 75,
        }

    def _critique(self, messages, tools, is_repair):
        n = _num(_user(messages), "EPISODE_NUMBER", 1)
        self.critiques[n] += 1
        low = n == 1 and self.critiques[n] == 1
        crits = ["hook", "pacing", "character_consistency", "dialogue", "cliffhanger", "faithfulness"]
        return {
            "scores": [
                {"criterion": c, "score": 3 if low else 4 + (i % 2), "reason": "mock judgement"}
                for i, c in enumerate(crits)
            ],
            "continuity_issues": ["Ramdas is called Meera's uncle; the bible says father."] if low else [],
            "revision_notes": "Open on the moneylender's knock; cut the exposition in scene 1."
            if low
            else "Tighten scene 2.",
        }

    def _localize(self, messages, tools, is_repair):
        user = _user(messages)
        lang = re.search(r"TARGET_LANGUAGE:\s*(\S+)", user).group(1)
        ep = json.loads(user.split("EPISODE:\n", 1)[1])
        tag = "हिंदी" if lang == "hi" else lang
        return {
            "episode": ep["episode"],
            "language": lang,
            "title": f"[{tag}] {ep['title']}",
            "hook": f"[{tag}] {ep['hook']}",
            "cliffhanger": f"[{tag}] {ep['cliffhanger']}",
            "lines": [{"id": x["id"], "line": f"[{tag}] {x['line']}"} for x in ep["lines"]],
        }

    def _rewrite_hook(self, messages, tools, is_repair):
        m = re.search(r"\(episode (\d+)\)", _user(messages))
        n = int(m.group(1)) if m else 0
        return {"hook": f"The kiln hisses as rain hits it; episode {n} opens in steam and darkness."}
