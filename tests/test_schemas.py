import pytest
from pydantic import ValidationError

from storyforge.mock_llm import BIBLE
from storyforge.schemas import Critique, LocalizedEpisode, StoryBible


def test_bible_parses():
    b = StoryBible.model_validate(BIBLE)
    assert b.character_names() == ["Meera", "Ramdas", "Seth Lalchand"]


def test_bible_needs_three_beats():
    with pytest.raises(ValidationError):
        StoryBible.model_validate(dict(BIBLE, beats=BIBLE["beats"][:2]))


def test_critique_overall():
    c = Critique.model_validate(
        {
            "scores": [
                {"criterion": "hook", "score": 5, "reason": ""},
                {"criterion": "pacing", "score": 2, "reason": ""},
            ],
            "revision_notes": "x",
        }
    )
    assert c.overall == 3.5


def test_score_range_enforced():
    with pytest.raises(ValidationError):
        Critique.model_validate({"scores": [{"criterion": "hook", "score": 7, "reason": ""}], "revision_notes": ""})


def test_localized_lines_not_empty():
    with pytest.raises(ValidationError):
        LocalizedEpisode.model_validate(
            {
                "episode": 1,
                "language": "hi",
                "title": "t",
                "hook": "h",
                "cliffhanger": "c",
                "lines": [{"id": "e1s1l1", "line": "  "}],
            }
        )
