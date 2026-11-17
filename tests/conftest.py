import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("CRITIC_PROVIDER", "mock")
os.environ.setdefault("EMBED_MODEL", "hash")

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "stories" / "the_potters_daughter.txt"


@pytest.fixture
def sample_text() -> str:
    return SAMPLE.read_text(encoding="utf-8")


@pytest.fixture
def mock_settings(tmp_path):
    from storyforge.config import get_settings

    return get_settings(
        provider="mock",
        critic_provider="mock",
        embed_model="hash",
        num_episodes=4,
        target_languages=["hi", "hi-Latn"],
        output_dir=str(tmp_path),
    )
