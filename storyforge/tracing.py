"""Lightweight tracing + metrics.

Every LLM call and every graph node is recorded to `trace.jsonl` in the run's
output folder: stage, model, tokens, latency and cost. If LANGFUSE_* keys are
set, the same events are also sent to Langfuse. Tracing must never break the
pipeline, so every Langfuse call is wrapped defensively.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import PRICING


@dataclass
class LLMEvent:
    stage: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    kind: str = "llm"  # llm | tool | repair
    ok: bool = True
    note: str = ""


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return round(input_tokens / 1e6 * inp + output_tokens / 1e6 * out, 6)


@dataclass
class Metrics:
    """Numbers that end up in metrics.json (and on your resume)."""

    structured_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_pass_valid: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    valid_after_repair: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tool_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_draft_scores: dict[int, float] = field(default_factory=dict)
    final_scores: dict[int, float] = field(default_factory=dict)
    first_draft_continuity: dict[int, int] = field(default_factory=dict)
    final_continuity: dict[int, int] = field(default_factory=dict)
    revisions: dict[int, int] = field(default_factory=dict)
    dedup_hooks_rewritten: int = 0
    dedup_lines_removed: int = 0
    node_latency_s: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def summary(self, events: list[LLMEvent]) -> dict[str, Any]:
        total_calls = sum(self.structured_calls.values())
        fp = sum(self.first_pass_valid.values())
        rep = sum(self.valid_after_repair.values())

        def avg(d: dict) -> float | None:
            return round(sum(d.values()) / len(d), 2) if d else None

        n_eps = max(len(self.final_scores), 1)
        return {
            "structured_outputs": {
                "calls": total_calls,
                "first_pass_valid_pct": round(100 * fp / total_calls, 1) if total_calls else None,
                "valid_after_repair_pct": round(100 * (fp + rep) / total_calls, 1) if total_calls else None,
                "failed": sum(self.failed.values()),
                "by_stage": {
                    k: {
                        "calls": self.structured_calls[k],
                        "first_pass_valid": self.first_pass_valid.get(k, 0),
                        "repaired": self.valid_after_repair.get(k, 0),
                    }
                    for k in self.structured_calls
                },
            },
            "quality": {
                "avg_first_draft_score": avg(self.first_draft_scores),
                "avg_final_score": avg(self.final_scores),
                "avg_continuity_issues_first_draft": avg(self.first_draft_continuity),
                "avg_continuity_issues_final": avg(self.final_continuity),
                "total_revisions": sum(self.revisions.values()),
                "per_episode": {
                    ep: {
                        "first_draft": self.first_draft_scores.get(ep),
                        "final": self.final_scores.get(ep),
                        "revisions": self.revisions.get(ep, 0),
                        "continuity_first": self.first_draft_continuity.get(ep),
                        "continuity_final": self.final_continuity.get(ep),
                    }
                    for ep in sorted(self.final_scores)
                },
            },
            "dedup": {
                "hooks_rewritten": self.dedup_hooks_rewritten,
                "dialogue_lines_removed": self.dedup_lines_removed,
            },
            "tools": dict(self.tool_calls),
            "cost_latency": {
                "llm_calls": len(events),
                "input_tokens": sum(e.input_tokens for e in events),
                "output_tokens": sum(e.output_tokens for e in events),
                "total_cost_usd": round(sum(e.cost_usd for e in events), 5),
                "cost_per_episode_usd": round(sum(e.cost_usd for e in events) / n_eps, 5),
                "total_llm_latency_s": round(sum(e.latency_s for e in events), 2),
                "node_latency_s": {k: round(v, 2) for k, v in self.node_latency_s.items()},
            },
        }


class Tracer:
    def __init__(self, run_id: str, out_dir: str | Path | None = None):
        self.run_id = run_id
        self.events: list[LLMEvent] = []
        self.metrics = Metrics()
        self.out_path = Path(out_dir) / "trace.jsonl" if out_dir else None
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._lf = self._init_langfuse()

    # ---------------------------------------------------------------- langfuse
    @staticmethod
    def _init_langfuse():
        if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
            return None
        try:
            from langfuse import get_client  # langfuse >= 3

            return get_client()
        except Exception:  # pragma: no cover - optional dependency
            return None

    def _lf_generation(self, ev: LLMEvent, messages: Any, output: Any) -> None:
        if not self._lf:
            return
        try:  # pragma: no cover - network
            with self._lf.start_as_current_generation(
                name=f"{ev.stage}:{ev.kind}",
                model=ev.model,
                input=messages,
                metadata={"run_id": self.run_id, "provider": ev.provider},
            ) as gen:
                gen.update(
                    output=output,
                    usage_details={"input": ev.input_tokens, "output": ev.output_tokens},
                    cost_details={"total": ev.cost_usd},
                )
        except Exception:
            pass

    # ---------------------------------------------------------------- records
    def record(self, ev: LLMEvent, messages: Any = None, output: Any = None) -> None:
        self.events.append(ev)
        if self.out_path:
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": self.run_id, "ts": time.time(), **asdict(ev)}, ensure_ascii=False) + "\n")
        self._lf_generation(ev, messages, output)

    @contextmanager
    def node(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.metrics.node_latency_s[name] += dt
            if self.out_path:
                with self.out_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "run_id": self.run_id,
                                "ts": time.time(),
                                "kind": "node",
                                "node": name,
                                "latency_s": round(dt, 3),
                            }
                        )
                        + "\n"
                    )

    def flush(self) -> None:
        if self._lf:
            try:  # pragma: no cover
                self._lf.flush()
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        return self.metrics.summary(self.events)
