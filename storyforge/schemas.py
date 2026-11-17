"""Pydantic schemas: the contract between every stage of the pipeline.

Every LLM output is parsed into one of these models. If parsing fails, the
LLM client runs a repair prompt (see llm.py), so downstream nodes can always
trust the shape of the data they receive.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------- story bible
class Character(BaseModel):
    name: str = Field(description="Canonical name as it appears in the source")
    name_hi: str = Field(default="", description="Name in Devanagari, for the localizer glossary")
    role: str = Field(description="protagonist / antagonist / supporting / minor")
    description: str = Field(description="Look, personality, motivation in 1-2 sentences")
    relationships: list[str] = Field(default_factory=list, description="e.g. 'mother of Hamid'")


class Location(BaseModel):
    name: str
    description: str


class PlotBeat(BaseModel):
    order: int = Field(ge=1)
    summary: str = Field(description="What happens, in one or two sentences")
    characters: list[str] = Field(default_factory=list)
    emotional_tone: str = Field(default="", description="e.g. hopeful, tense, bittersweet")


class StoryBible(BaseModel):
    title: str
    logline: str = Field(description="The whole story in one sentence")
    genre: str
    tone: str
    setting: str = Field(description="Time and place")
    characters: list[Character] = Field(min_length=1)
    locations: list[Location] = Field(default_factory=list)
    beats: list[PlotBeat] = Field(min_length=3)
    themes: list[str] = Field(default_factory=list)

    def character_names(self) -> list[str]:
        return [c.name for c in self.characters]


# ---------------------------------------------------------------- planning
class EpisodeOutline(BaseModel):
    episode: int = Field(ge=1)
    title: str
    beat_orders: list[int] = Field(description="Which PlotBeat.order values this episode covers")
    hook: str = Field(description="What grabs the viewer in the first 5 seconds")
    cliffhanger: str = Field(description="The unresolved question that ends the episode")
    summary: str


class SeriesPlan(BaseModel):
    series_title: str
    episodes: list[EpisodeOutline] = Field(min_length=1)


# ---------------------------------------------------------------- script
class DialogueLine(BaseModel):
    id: str = Field(description="Stable id like 'e1s2l3' so the localizer can map lines")
    speaker: str = Field(description="Character name, or NARRATOR")
    line: str
    direction: str = Field(default="", description="Delivery note, e.g. '(whispering)'")


class Scene(BaseModel):
    scene: int = Field(ge=1)
    location: str
    time_of_day: str = Field(default="DAY")
    characters: list[str] = Field(default_factory=list)
    action: str = Field(description="What we see, present tense, 1-3 sentences")
    dialogue: list[DialogueLine] = Field(default_factory=list)
    visual_prompt: str = Field(description="A text-to-image prompt for a storyboard key frame of this scene")


class EpisodeScript(BaseModel):
    episode: int = Field(ge=1)
    title: str
    hook: str
    scenes: list[Scene] = Field(min_length=1)
    cliffhanger: str
    est_duration_sec: int = Field(default=75, ge=15, le=300)


# ---------------------------------------------------------------- critique
class CriterionScore(BaseModel):
    criterion: Literal["hook", "pacing", "character_consistency", "dialogue", "cliffhanger", "faithfulness"]
    score: int = Field(ge=1, le=5)
    reason: str


class Critique(BaseModel):
    scores: list[CriterionScore]
    continuity_issues: list[str] = Field(
        default_factory=list,
        description="Concrete contradictions with the story bible; empty if none",
    )
    revision_notes: str = Field(description="Specific, actionable instructions for the writer")

    @property
    def overall(self) -> float:
        return round(sum(s.score for s in self.scores) / max(len(self.scores), 1), 2)


# ---------------------------------------------------------------- localization
class LocalizedLine(BaseModel):
    id: str
    line: str


class LocalizedEpisode(BaseModel):
    episode: int
    language: str = Field(description="BCP-47 tag, e.g. 'hi' or 'hi-Latn' for Hinglish")
    title: str
    hook: str
    cliffhanger: str
    lines: list[LocalizedLine]

    @field_validator("lines")
    @classmethod
    def no_empty_lines(cls, v: list[LocalizedLine]) -> list[LocalizedLine]:
        if any(not item.line.strip() for item in v):
            raise ValueError("localized lines must not be empty")
        return v
