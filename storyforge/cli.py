"""Command-line interface.

python -m storyforge run data/stories/the_potters_daughter.txt --episodes 5 --languages hi,hi-Latn
python -m storyforge run data/stories/the_potters_daughter.txt --mock      # offline demo, no API key
python -m storyforge ablation data/stories/the_potters_daughter.txt       # RAG on vs off
python -m storyforge fetch <url> data/stories/idgah.txt                   # download a public-domain story
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .ingest import fetch_story, load_story
from .llm import StageError


def _settings(args: argparse.Namespace, **extra):
    kw = dict(
        num_episodes=args.episodes,
        target_languages=[x.strip() for x in args.languages.split(",") if x.strip()] if args.languages else None,
        output_dir=args.out,
        **extra,
    )
    if args.mock:
        kw.update(provider="mock", critic_provider="mock", embed_model="hash")
    return get_settings(**kw)


def _print_summary(metrics: dict, outputs: dict) -> None:
    so, q, cl = metrics["structured_outputs"], metrics["quality"], metrics["cost_latency"]
    rows = [
        ("Episodes", metrics["run"]["episodes"]),
        ("Schema-valid on first pass", f"{so['first_pass_valid_pct']}%"),
        ("Schema-valid after repair", f"{so['valid_after_repair_pct']}%"),
        ("Avg judge score: first draft -> final", f"{q['avg_first_draft_score']} -> {q['avg_final_score']} / 5"),
        (
            "Continuity issues/ep: first -> final",
            f"{q['avg_continuity_issues_first_draft']} -> {q['avg_continuity_issues_final']}",
        ),
        ("Critic-driven revisions", q["total_revisions"]),
        (
            "Hooks rewritten / lines deduped",
            f"{metrics['dedup']['hooks_rewritten']} / {metrics['dedup']['dialogue_lines_removed']}",
        ),
        ("Tool calls", sum(metrics["tools"].values())),
        ("LLM calls / tokens in+out", f"{cl['llm_calls']} / {cl['input_tokens']}+{cl['output_tokens']}"),
        ("Cost per episode (list price)", f"${cl['cost_per_episode_usd']}"),
        ("Total LLM latency", f"{cl['total_llm_latency_s']} s"),
    ]
    w = max(len(r[0]) for r in rows)
    print("\n" + "\n".join(f"  {k:<{w}}  {v}" for k, v in rows))
    print("\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k:<14} {v}")


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline

    settings = _settings(args, use_rag=not args.no_rag)
    text = load_story(args.story)
    title = args.title or Path(args.story).stem.replace("_", " ").title()
    pipe = Pipeline(settings)
    print(
        f"StoryForge run {pipe.run_id}: provider={settings.provider} model={settings.model('write')} "
        f"critic={settings.critic_provider} rag={settings.use_rag} -> {pipe.out_dir}"
    )
    try:
        state = pipe.run(text, title)
    except StageError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    _print_summary(state["metrics"], state["outputs"])
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Same story, RAG on vs off: shows what the story-bible retrieval buys you."""
    from .pipeline import Pipeline

    text = load_story(args.story)
    results = {}
    for rag in (True, False):
        pipe = Pipeline(_settings(args, use_rag=rag))
        print(f"Running with RAG={'on' if rag else 'off'} ({pipe.run_id}) ...")
        results["on" if rag else "off"] = pipe.run(text, args.title or "")["metrics"]["quality"]
    print("\n                           RAG on    RAG off")
    for key, label in [
        ("avg_continuity_issues_first_draft", "Continuity issues/ep (1st)"),
        ("avg_first_draft_score", "Judge score (1st draft)"),
        ("avg_final_score", "Judge score (final)"),
        ("total_revisions", "Revisions needed"),
    ]:
        print(f"  {label:<26} {results['on'][key]!s:<9} {results['off'][key]}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "ablation_rag.json").write_text(json.dumps(results, indent=2))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    out = fetch_story(args.url, args.out_path)
    print(f"Saved {out} ({out.stat().st_size // 1024} KB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="storyforge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("story", help="Path to a .txt story")
        sp.add_argument("--title", default="")
        sp.add_argument("--episodes", type=int, default=None, help="Number of episodes (default 5)")
        sp.add_argument("--languages", default=None, help="Comma-separated, e.g. hi,hi-Latn")
        sp.add_argument("--out", default="outputs", help="Output directory")
        sp.add_argument("--mock", action="store_true", help="Offline demo with a fake LLM (no API key)")

    r = sub.add_parser("run", help="Generate a microdrama series from a story")
    common(r)
    r.add_argument("--no-rag", action="store_true", help="Disable story-bible retrieval (ablation)")
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("ablation", help="Compare RAG on vs off on the same story")
    common(a)
    a.set_defaults(fn=cmd_ablation)

    f = sub.add_parser("fetch", help="Download and clean a public-domain story")
    f.add_argument("url")
    f.add_argument("out_path")
    f.set_defaults(fn=cmd_fetch)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
