"""The StoryForge pipeline as a LangGraph state machine.

    ingest -> extract -> index_bible -> plan -> write <-> critique -> accept --+
                                                 ^                          |
                                                 +---- next episode --------+
                                                                            |
                                         finalize <- localize <- dedup <----+

* write:     tool-calling agent; queries the Qdrant story bible before writing.
* critique:  LLM-as-judge on a 6-criterion rubric. Below the threshold, the
             episode goes back to the writer with notes (max MAX_REVISIONS).
* dedup:     embedding similarity catches repeated hooks / dialogue lines.
* localize:  Hindi / Hinglish adaptation with a name glossary.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from string import Template
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .config import Settings, get_settings
from .ingest import clean_text
from .llm import LLMClient
from .memory import Embedder, StoryBibleStore, make_embedder, near_duplicates
from .render import render_localized_markdown, render_markdown
from .schemas import (
    Critique,
    EpisodeScript,
    LocalizedEpisode,
    SeriesPlan,
    StoryBible,
)
from .tracing import Tracer

PROMPTS = Path(__file__).parent / "prompts"
CRITERIA = {"hook", "pacing", "character_consistency", "dialogue", "cliffhanger", "faithfulness"}


def prompt(name: str, **kw: Any) -> str:
    vals = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in kw.items()}
    return Template((PROMPTS / f"{name}.md").read_text(encoding="utf-8")).safe_substitute(**vals)


def dump(model: BaseModel | list | dict) -> str:
    if isinstance(model, BaseModel):
        return model.model_dump_json(indent=1)
    return json.dumps(model, ensure_ascii=False, indent=1, default=lambda o: o.model_dump())


def known_speaker(speaker: str, names: set[str]) -> bool:
    """True if `speaker` is a bible character or NARRATOR.

    Tolerates short forms ("Lalchand" for "Seth Lalchand") and parentheticals ("NARRATOR (V.O.)").
    """
    sp = re.sub(r"\(.*?\)", "", speaker).strip().lower()
    return bool(sp) and any(sp == n or sp in n.split() or n in sp for n in names)


def renumber(ep: EpisodeScript) -> EpisodeScript:
    """Assign scene numbers and line ids in code: deterministic fixes beat a repair call."""
    for k, sc in enumerate(ep.scenes, start=1):
        sc.scene = k
        for i, d in enumerate(sc.dialogue, start=1):
            d.id = f"e{ep.episode}s{k}l{i}"
    return ep


class HookOut(BaseModel):
    hook: str


class StoryState(TypedDict, total=False):
    title: str
    source_text: str
    clean_text: str
    bible: StoryBible
    plan: SeriesPlan
    current_ep: int
    draft: EpisodeScript
    critique: Critique
    revision: int
    decision: Literal["revise", "accept"]
    episodes: list[EpisodeScript]
    localized: dict[str, list[LocalizedEpisode]]
    outputs: dict[str, str]
    metrics: dict[str, Any]


# Writer tools, in OpenAI function-calling format.
WRITER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_character",
            "description": "Look up a character's canonical description and relationships in the story bible.",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prior_events",
            "description": "List plot beats already shown in episodes before the given episode.",
            "parameters": {"type": "object", "properties": {"episode": {"type": "integer"}}, "required": ["episode"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_story_bible",
            "description": "Semantic search over characters, locations and plot beats.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        },
    },
]


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        run_id: str | None = None,
        out_dir: str | Path | None = None,
        embedder: Embedder | None = None,
    ):
        self.settings = settings or get_settings()
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.out_dir = Path(out_dir or Path(self.settings.output_dir) / self.run_id)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tracer = Tracer(self.run_id, self.out_dir)
        self.llm = LLMClient(self.settings, self.tracer)
        self.critic_llm = (
            self.llm
            if self.settings.critic_provider == self.settings.provider
            else LLMClient(self.settings, self.tracer, provider=self.settings.critic_provider)
        )
        if embedder is None and self.settings.provider == "mock":
            embedder = make_embedder("hash")  # offline mode stays fully offline
        self.embedder = embedder or make_embedder(self.settings.embed_model)
        self.store = StoryBibleStore(
            self.embedder,
            f"storyforge_{self.run_id}",
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
        )
        self.graph = self.build_graph()

    # ================================================================ nodes
    def ingest(self, s: StoryState) -> StoryState:
        with self.tracer.node("ingest"):
            return {"clean_text": clean_text(s["source_text"])}

    def extract(self, s: StoryState) -> StoryState:
        with self.tracer.node("extract"):
            msgs = [
                {"role": "system", "content": prompt("extractor", schema=StoryBible.model_json_schema())},
                {
                    "role": "user",
                    "content": f"TITLE HINT: {s.get('title') or 'unknown'}\n\nSOURCE STORY:\n{s['clean_text']}",
                },
            ]

            def check(b: StoryBible) -> str | None:
                orders = [x.order for x in b.beats]
                if orders != list(range(1, len(orders) + 1)):
                    return f"beat orders must be 1..{len(orders)} in sequence, got {orders}"
                return None

            bible = self.llm.structured(msgs, StoryBible, "extract", validate=check)
            return {"bible": bible}

    def index_bible(self, s: StoryState) -> StoryState:
        with self.tracer.node("index_bible"):
            self.store.index(s["bible"])
            return {}

    def plan(self, s: StoryState) -> StoryState:
        with self.tracer.node("plan"):
            bible = s["bible"]
            n = min(self.settings.num_episodes, len(bible.beats))
            all_beats = [b.order for b in bible.beats]

            def check(p: SeriesPlan) -> str | None:
                if [e.episode for e in p.episodes] != list(range(1, n + 1)):
                    return f"need exactly {n} episodes numbered 1..{n}"
                used = [o for e in p.episodes for o in e.beat_orders]
                if sorted(used) != all_beats:
                    return f"every beat {all_beats} must be used exactly once; you used {used}"
                if any(not e.beat_orders for e in p.episodes):
                    return "every episode needs at least one beat"
                return None

            msgs = [
                {"role": "system", "content": prompt("planner", num_episodes=n, schema=SeriesPlan.model_json_schema())},
                {"role": "user", "content": f"NUM_EPISODES: {n}\n\nSTORY BIBLE:\n{dump(bible)}"},
            ]
            plan = self.llm.structured(msgs, SeriesPlan, "plan", validate=check)
            self.store.set_plan(plan)
            return {"plan": plan, "current_ep": 1, "revision": 0, "episodes": []}

    def write(self, s: StoryState) -> StoryState:
        with self.tracer.node("write"):
            bible, plan, n = s["bible"], s["plan"], s["current_ep"]
            outline = plan.episodes[n - 1]
            beats = [b for b in bible.beats if b.order in outline.beat_orders]
            prev = s["episodes"][-1] if s.get("episodes") else None
            names = {c.name.lower() for c in bible.characters} | {"narrator"}

            parts = [
                f"EPISODE_NUMBER: {n} of {len(plan.episodes)}",
                f"SERIES: {plan.series_title}",
                f"EPISODE OUTLINE:\n{dump(outline)}",
                f"BEATS TO COVER:\n{dump(beats)}",
                f"PREVIOUS EPISODE ENDED ON: {prev.cliffhanger if prev else '(series opener)'}",
                f"CHARACTER NAMES (exact spelling): {', '.join(bible.character_names())}",
            ]
            if s.get("revision", 0) > 0 and s.get("critique") is not None:
                c = s["critique"]
                parts.append(
                    f"REVISION REQUEST (attempt {s['revision']}): the editor scored your draft {c.overall}/5.\n"
                    f"Fix these: {c.revision_notes}\n"
                    f"Continuity issues: {c.continuity_issues or 'none'}\n"
                    f"YOUR PREVIOUS DRAFT:\n{dump(s['draft'])}"
                )

            def check(ep: EpisodeScript) -> str | None:
                if ep.episode != n:
                    return f"episode must be {n}"

                bad = sorted(
                    {d.speaker for sc in ep.scenes for d in sc.dialogue if not known_speaker(d.speaker, names)}
                )
                if bad:
                    return f"unknown speakers {bad}; use only {sorted(names)}"
                return None

            msgs = [
                {"role": "system", "content": prompt("writer", schema=EpisodeScript.model_json_schema())},
                {"role": "user", "content": "\n\n".join(parts)},
            ]
            if self.settings.use_rag:
                impls = {
                    "get_character": self.store.get_character,
                    "get_prior_events": self.store.get_prior_events,
                    "search_story_bible": self.store.search_story_bible,
                }
                draft = self.llm.run_tools(msgs, WRITER_TOOLS, impls, EpisodeScript, "write", validate=check)
            else:  # ablation: no retrieval, the writer only sees the outline and names
                draft = self.llm.structured(msgs, EpisodeScript, "write", validate=check)
            return {"draft": renumber(draft)}

    def critique(self, s: StoryState) -> StoryState:
        with self.tracer.node("critique"):
            bible, plan, n = s["bible"], s["plan"], s["current_ep"]
            is_final = n == len(plan.episodes)

            def check(c: Critique) -> str | None:
                got = [x.criterion for x in c.scores]
                if sorted(got) != sorted(CRITERIA):
                    return f"score each of {sorted(CRITERIA)} exactly once"
                return None

            msgs = [
                {"role": "system", "content": prompt("critic", schema=Critique.model_json_schema())},
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"EPISODE_NUMBER: {n} of {len(plan.episodes)} (final episode: {is_final})",
                            f"STORY BIBLE:\n{dump(bible)}",
                            f"EPISODE OUTLINE:\n{dump(plan.episodes[n - 1])}",
                            f"WHAT THE AUDIENCE ALREADY KNOWS:\n{self.store.get_prior_events(n)}",
                            f"SCRIPT TO JUDGE:\n{dump(s['draft'])}",
                        ]
                    ),
                },
            ]
            # Low temperature: a judge should give the same score to the same script.
            crit: Critique = self.critic_llm.structured(msgs, Critique, "critique", validate=check, temperature=0.1)
            m = self.tracer.metrics
            rev = s.get("revision", 0)
            if rev == 0:
                m.first_draft_scores[n] = crit.overall
                m.first_draft_continuity[n] = len(crit.continuity_issues)
            m.final_scores[n] = crit.overall
            m.final_continuity[n] = len(crit.continuity_issues)

            needs_work = crit.overall < self.settings.critic_threshold or crit.continuity_issues
            if needs_work and rev < self.settings.max_revisions:
                return {"critique": crit, "decision": "revise", "revision": rev + 1}
            return {"critique": crit, "decision": "accept"}

    def accept(self, s: StoryState) -> StoryState:
        with self.tracer.node("accept"):
            n = s["current_ep"]
            self.tracer.metrics.revisions[n] = s.get("revision", 0)
            return {
                "episodes": [*s.get("episodes", []), s["draft"]],
                "current_ep": n + 1,
                "revision": 0,
                "critique": None,
            }

    def dedup(self, s: StoryState) -> StoryState:
        with self.tracer.node("dedup"):
            eps = [e.model_copy(deep=True) for e in s["episodes"]]
            th, m = self.settings.dedup_threshold, self.tracer.metrics

            # 1) Repeated hooks across episodes: rewrite the later one.
            for i, j, sim in near_duplicates([e.hook for e in eps], self.embedder, th):
                msgs = [
                    {"role": "system", "content": prompt("rewrite_hook")},
                    {
                        "role": "user",
                        "content": (
                            f"EARLIER HOOK: {eps[i].hook}\nREPEATED HOOK (episode {eps[j].episode}): {eps[j].hook}\n"
                            f"SIMILARITY: {sim}\nEPISODE CONTENT:\n{dump(eps[j])}"
                        ),
                    },
                ]
                eps[j].hook = self.llm.structured(msgs, HookOut, "dedup").hook
                m.dedup_hooks_rewritten += 1

            # 2) Near-identical dialogue lines inside an episode: drop the repeat.
            for ep in eps:
                lines = [(sc, d) for sc in ep.scenes for d in sc.dialogue]
                pairs = near_duplicates([d.line for _, d in lines], self.embedder, th)
                # Only the same speaker repeating themselves counts; an echo by
                # another character can be a deliberate dramatic choice.
                dupes = {j for i, j, _s in pairs if lines[i][1].speaker == lines[j][1].speaker}
                for idx in sorted(dupes, reverse=True):
                    sc, d = lines[idx]
                    sc.dialogue.remove(d)
                    m.dedup_lines_removed += 1
            return {"episodes": eps}

    def localize(self, s: StoryState) -> StoryState:
        with self.tracer.node("localize"):
            glossary = {c.name: c.name_hi or c.name for c in s["bible"].characters}
            out: dict[str, list[LocalizedEpisode]] = {}
            for lang in self.settings.target_languages:
                out[lang] = []
                for ep in s["episodes"]:
                    src = {
                        "episode": ep.episode,
                        "title": ep.title,
                        "hook": ep.hook,
                        "cliffhanger": ep.cliffhanger,
                        "lines": [
                            {"id": d.id, "speaker": d.speaker, "line": d.line} for sc in ep.scenes for d in sc.dialogue
                        ],
                    }
                    ids = [x["id"] for x in src["lines"]]

                    def check(loc: LocalizedEpisode, ids: list[str] = ids) -> str | None:
                        got = [x.id for x in loc.lines]
                        if got != ids:
                            missing = sorted(set(ids) - set(got))
                            return f"line ids must be exactly {ids} in order; missing {missing}"
                        return None

                    msgs = [
                        {
                            "role": "system",
                            "content": prompt(
                                "localizer",
                                language=lang,
                                glossary=json.dumps(glossary, ensure_ascii=False),
                                schema=LocalizedEpisode.model_json_schema(),
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"TARGET_LANGUAGE: {lang}\n\nEPISODE:\n{json.dumps(src, ensure_ascii=False, indent=1)}",
                        },
                    ]
                    out[lang].append(self.llm.structured(msgs, LocalizedEpisode, "localize", validate=check))
            return {"localized": out}

    def finalize(self, s: StoryState) -> StoryState:
        with self.tracer.node("finalize"):
            d = self.out_dir
            (d / "story_bible.json").write_text(s["bible"].model_dump_json(indent=2), encoding="utf-8")
            (d / "series_plan.json").write_text(s["plan"].model_dump_json(indent=2), encoding="utf-8")
            (d / "episodes.json").write_text(dump(s["episodes"]), encoding="utf-8")
            outputs = {"story_bible": str(d / "story_bible.json"), "episodes": str(d / "episodes.json")}
            md = render_markdown(s["bible"], s["plan"], s["episodes"])
            (d / "script_en.md").write_text(md, encoding="utf-8")
            outputs["script_en"] = str(d / "script_en.md")
            for lang, eps in s.get("localized", {}).items():
                p = d / f"script_{lang}.md"
                p.write_text(render_localized_markdown(s["plan"], s["episodes"], eps), encoding="utf-8")
                outputs[f"script_{lang}"] = str(p)
            metrics = self.tracer.summary()
            metrics["run"] = {
                "run_id": self.run_id,
                "provider": self.settings.provider,
                "critic_provider": self.settings.critic_provider,
                "model": self.settings.model("write"),
                "use_rag": self.settings.use_rag,
                "episodes": len(s["episodes"]),
                "languages": self.settings.target_languages,
            }
            (d / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
            outputs["metrics"] = str(d / "metrics.json")
            self.tracer.flush()
            return {"outputs": outputs, "metrics": metrics}

    # ================================================================ routing
    @staticmethod
    def after_critique(s: StoryState) -> str:
        return "write" if s.get("decision") == "revise" else "accept"

    @staticmethod
    def after_accept(s: StoryState) -> str:
        return "write" if s["current_ep"] <= len(s["plan"].episodes) else "dedup"

    def build_graph(self):
        g = StateGraph(StoryState)
        for name in [
            "ingest",
            "extract",
            "index_bible",
            "plan",
            "write",
            "critique",
            "accept",
            "dedup",
            "localize",
            "finalize",
        ]:
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "ingest")
        g.add_edge("ingest", "extract")
        g.add_edge("extract", "index_bible")
        g.add_edge("index_bible", "plan")
        g.add_edge("plan", "write")
        g.add_edge("write", "critique")
        g.add_conditional_edges("critique", self.after_critique, {"write": "write", "accept": "accept"})
        g.add_conditional_edges("accept", self.after_accept, {"write": "write", "dedup": "dedup"})
        g.add_edge("dedup", "localize")
        g.add_edge("localize", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, text: str, title: str = "") -> StoryState:
        # Each episode takes up to (2 + 2*max_revisions) steps; leave generous headroom.
        limit = 20 + self.settings.num_episodes * (3 + 2 * self.settings.max_revisions)
        return self.graph.invoke({"source_text": text, "title": title}, {"recursion_limit": limit})
