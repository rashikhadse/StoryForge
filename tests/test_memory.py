from storyforge.memory import HashEmbedder, StoryBibleStore, near_duplicates
from storyforge.mock_llm import BIBLE
from storyforge.schemas import SeriesPlan, StoryBible


def _store():
    s = StoryBibleStore(HashEmbedder(), "test")
    s.index(StoryBible.model_validate(BIBLE))
    return s


def test_get_character_exact_and_partial():
    s = _store()
    assert "father of Meera" in s.get_character("Ramdas")
    assert "moneylender" in s.get_character("lalchand")


def test_prior_events_respect_plan():
    s = _store()
    s.set_plan(
        SeriesPlan.model_validate(
            {
                "series_title": "x",
                "episodes": [
                    {
                        "episode": 1,
                        "title": "a",
                        "beat_orders": [1, 2],
                        "hook": "h",
                        "cliffhanger": "c",
                        "summary": "s",
                    },
                    {"episode": 2, "title": "b", "beat_orders": [3], "hook": "h", "cliffhanger": "c", "summary": "s"},
                ],
            }
        )
    )
    prior = s.get_prior_events(2)
    assert "Beat 1" in prior and "Beat 2" in prior and "Beat 3" not in prior
    assert "first episode" in s.get_prior_events(1)


def test_search_returns_hits():
    assert len(_store().search("potter's wheel", k=3)) == 3


def test_near_duplicates():
    texts = ["A fist pounds on the door at dawn.", "A fist pounds on the door at dawn!", "Rain hits the kiln."]
    pairs = near_duplicates(texts, HashEmbedder(), 0.9)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
