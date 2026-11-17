"""Story bible memory: embeddings + Qdrant.

Every character, location and plot beat is stored as a vector in Qdrant. The
writer agent queries it through tools, so long series stay consistent (a
character's description in episode 5 matches episode 1).

Embedders
- FastEmbedEmbedder: real multilingual sentence embeddings (ONNX, CPU-friendly).
- HashEmbedder: dependency-free character n-gram hashing. Used in tests and
  offline demos; good enough for near-duplicate detection, not for semantics.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
import warnings
from typing import Protocol

import numpy as np

from .schemas import SeriesPlan, StoryBible


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = re.sub(r"\s+", " ", text.lower().strip())
        grams = [t[i : i + 3] for i in range(max(len(t) - 2, 1))] + t.split()
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 1) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))


class FastEmbedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self.model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        arr = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9, None)


def make_embedder(model_name: str) -> Embedder:
    if model_name in {"hash", "mock"}:
        return HashEmbedder()
    try:
        return FastEmbedEmbedder(model_name)
    except Exception as e:  # e.g. no internet to download the model
        warnings.warn(
            f"Could not load embedding model {model_name!r} ({type(e).__name__}: {e}). "
            "Falling back to HashEmbedder: dedup still works, semantic search is weaker.",
            stacklevel=2,
        )
        return HashEmbedder()


# ----------------------------------------------------------------- Qdrant store
class StoryBibleStore:
    def __init__(self, embedder: Embedder, collection: str, url: str = "", api_key: str = ""):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.embedder = embedder
        self.collection = re.sub(r"[^a-zA-Z0-9_-]", "_", collection)[:60]
        self.client = QdrantClient(url=url, api_key=api_key or None) if url else QdrantClient(":memory:")
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        self.bible: StoryBible | None = None
        self.plan: SeriesPlan | None = None

    # ------------------------------------------------------------- indexing
    def index(self, bible: StoryBible) -> int:
        self.bible = bible
        docs: list[tuple[str, dict]] = []
        for c in bible.characters:
            rel = f" Relationships: {', '.join(c.relationships)}." if c.relationships else ""
            docs.append(
                (
                    f"Character {c.name} ({c.role}): {c.description}{rel}",
                    {"type": "character", "name": c.name, "name_lc": c.name.lower()},
                )
            )
        for loc in bible.locations:
            docs.append((f"Location {loc.name}: {loc.description}", {"type": "location", "name": loc.name}))
        for b in bible.beats:
            docs.append(
                (
                    f"Beat {b.order}: {b.summary} (characters: {', '.join(b.characters)})",
                    {"type": "beat", "order": b.order},
                )
            )
        vecs = self.embedder.embed([d[0] for d in docs])
        points = [
            self.models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection}-{i}")),
                vector=vecs[i].tolist(),
                payload={"text": docs[i][0], **docs[i][1]},
            )
            for i in range(len(docs))
        ]
        self.client.upsert(self.collection, points=points)
        return len(points)

    def set_plan(self, plan: SeriesPlan) -> None:
        self.plan = plan

    # ------------------------------------------------------------- queries (agent tools)
    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[str]:
        flt = None
        if kind:
            m = self.models
            flt = m.Filter(must=[m.FieldCondition(key="type", match=m.MatchValue(value=kind))])
        vec = self.embedder.embed([query])[0].tolist()
        res = self.client.query_points(self.collection, query=vec, limit=k, query_filter=flt)
        return [p.payload["text"] for p in res.points]

    def get_character(self, name: str) -> str:
        assert self.bible is not None
        wanted = name.strip().lower()
        for c in self.bible.characters:
            if c.name.lower() == wanted or wanted in c.name.lower():
                rel = f" Relationships: {', '.join(c.relationships)}." if c.relationships else ""
                return f"{c.name} ({c.role}): {c.description}{rel}"
        hits = self.search(f"Character {name}", k=2, kind="character")
        return "No exact match. Closest characters:\n" + "\n".join(hits) if hits else "Not found."

    def get_prior_events(self, episode: int) -> str:
        """Beats already shown in episodes before `episode` (what the audience knows)."""
        assert self.bible is not None
        if not self.plan or episode <= 1:
            return "This is the first episode; nothing has happened yet."
        orders = {o for e in self.plan.episodes if e.episode < episode for o in e.beat_orders}
        beats = [f"Beat {b.order}: {b.summary}" for b in self.bible.beats if b.order in orders]
        return "\n".join(beats) or "No prior beats found."

    def search_story_bible(self, query: str) -> str:
        return "\n".join(self.search(query, k=4)) or "Nothing relevant found."


# ----------------------------------------------------------------- similarity
def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    if len(vecs) == 0:
        return np.zeros((0, 0))
    norm = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
    return norm @ norm.T


def near_duplicates(texts: list[str], embedder: Embedder, threshold: float) -> list[tuple[int, int, float]]:
    """Pairs (i, j, similarity) with j > i whose cosine similarity >= threshold."""
    if len(texts) < 2:
        return []
    sims = cosine_matrix(embedder.embed(texts))
    out = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sims[i, j] >= threshold and not math.isnan(sims[i, j]):
                out.append((i, j, float(round(sims[i, j], 4))))
    return out
