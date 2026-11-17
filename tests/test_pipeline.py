import json
from pathlib import Path

from storyforge.pipeline import Pipeline


def test_end_to_end_mock_run(mock_settings, sample_text):
    pipe = Pipeline(mock_settings)
    state = pipe.run(sample_text, "The Potter's Daughter")

    # four episodes, in order, each with scenes
    eps = state["episodes"]
    assert [e.episode for e in eps] == [1, 2, 3, 4]
    assert all(e.scenes for e in eps)

    m = state["metrics"]
    # the mock returns a broken bible first -> repaired
    assert m["structured_outputs"]["by_stage"]["extract"]["repaired"] == 1
    assert m["structured_outputs"]["valid_after_repair_pct"] == 100.0
    # the critic sent episode 1 back once and the score went up
    assert m["quality"]["per_episode"][1]["revisions"] == 1
    assert m["quality"]["avg_final_score"] > m["quality"]["avg_first_draft_score"]
    # the writer used its tools
    assert m["tools"]["get_character"] >= 4
    # dedup caught the repeated hook and line
    assert m["dedup"]["hooks_rewritten"] >= 1 and m["dedup"]["dialogue_lines_removed"] >= 1

    # localization kept every line id
    for lang in ("hi", "hi-Latn"):
        for ep, loc in zip(eps, state["localized"][lang], strict=True):
            assert [d.id for sc in ep.scenes for d in sc.dialogue] == [x.id for x in loc.lines]

    for key in ("story_bible", "episodes", "script_en", "script_hi", "script_hi-Latn", "metrics"):
        assert Path(state["outputs"][key]).exists(), key
    assert json.loads(Path(state["outputs"]["metrics"]).read_text())["run"]["episodes"] == 4
    assert (pipe.out_dir / "trace.jsonl").exists()


def test_no_rag_ablation_skips_tools(mock_settings, sample_text):
    mock_settings.use_rag = False
    state = Pipeline(mock_settings).run(sample_text)
    assert state["metrics"]["tools"] == {}


def test_known_speaker():
    from storyforge.pipeline import known_speaker

    names = {"meera", "ramdas", "seth lalchand", "narrator"}
    assert known_speaker("Lalchand", names)
    assert known_speaker("NARRATOR (V.O.)", names)
    assert known_speaker("MEERA", names)
    assert not known_speaker("Inspector Rao", names)
    assert not known_speaker("", names)


def test_renumber_assigns_ids():
    from storyforge.pipeline import renumber
    from storyforge.schemas import EpisodeScript

    ep = EpisodeScript.model_validate(
        {
            "episode": 2,
            "title": "t",
            "hook": "h",
            "cliffhanger": "c",
            "scenes": [
                {
                    "scene": 7,
                    "location": "x",
                    "action": "a",
                    "visual_prompt": "v",
                    "dialogue": [
                        {"id": "x", "speaker": "Meera", "line": "a"},
                        {"id": "x", "speaker": "Meera", "line": "b"},
                    ],
                },
                {
                    "scene": 9,
                    "location": "y",
                    "action": "a",
                    "visual_prompt": "v",
                    "dialogue": [{"id": "?", "speaker": "Ramdas", "line": "c"}],
                },
            ],
        }
    )
    ep = renumber(ep)
    assert [sc.scene for sc in ep.scenes] == [1, 2]
    assert [d.id for sc in ep.scenes for d in sc.dialogue] == ["e2s1l1", "e2s1l2", "e2s2l1"]
