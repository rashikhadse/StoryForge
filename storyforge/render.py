"""Render scripts as readable Markdown (what a producer or voice actor reads)."""

from __future__ import annotations

from .schemas import EpisodeScript, LocalizedEpisode, SeriesPlan, StoryBible


def render_markdown(bible: StoryBible, plan: SeriesPlan, episodes: list[EpisodeScript]) -> str:
    out = [f"# {plan.series_title}", "", f'*Adapted from "{bible.title}"* — {bible.logline}', ""]
    out += ["## Characters", ""]
    out += [f"- **{c.name}** ({c.name_hi}) — {c.role}: {c.description}" for c in bible.characters]
    for ep in episodes:
        out += [
            "",
            "---",
            "",
            f"## Episode {ep.episode}: {ep.title}  (~{ep.est_duration_sec}s)",
            "",
            f"**Hook:** {ep.hook}",
            "",
        ]
        for sc in ep.scenes:
            out += [
                f"### Scene {sc.scene} — {sc.location.upper()} — {sc.time_of_day.upper()}",
                "",
                f"*{sc.action}*",
                "",
            ]
            for d in sc.dialogue:
                direction = f" {d.direction}" if d.direction else ""
                out.append(f"**{d.speaker.upper()}**{direction}: {d.line}  ")
            out += ["", f"> Storyboard prompt: {sc.visual_prompt}", ""]
        out += [f"**Cliffhanger:** {ep.cliffhanger}"]
    return "\n".join(out) + "\n"


def render_localized_markdown(
    plan: SeriesPlan, episodes: list[EpisodeScript], localized: list[LocalizedEpisode]
) -> str:
    lang = localized[0].language if localized else ""
    out = [f"# {plan.series_title} [{lang}]", ""]
    for ep, loc in zip(episodes, localized, strict=True):
        lines = {x.id: x.line for x in loc.lines}
        out += ["---", "", f"## Episode {ep.episode}: {loc.title}", "", f"**Hook:** {loc.hook}", ""]
        for sc in ep.scenes:
            out += [f"### Scene {sc.scene} — {sc.location.upper()}", ""]
            for d in sc.dialogue:
                out.append(f"**{d.speaker.upper()}**: {lines.get(d.id, d.line)}  ")
            out.append("")
        out += [f"**Cliffhanger:** {loc.cliffhanger}", ""]
    return "\n".join(out) + "\n"
