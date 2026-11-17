"""FastAPI service.

    uvicorn storyforge.api:app --reload --port 7860

POST /generate      start a job (returns job_id immediately; generation takes minutes)
GET  /jobs/{id}     status, metrics, and which scripts are ready
GET  /jobs/{id}/script?lang=en|hi|hi-Latn   the script as Markdown
GET  /health
GET  /              a minimal web demo
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PROVIDERS, get_settings

app = FastAPI(
    title="StoryForge",
    version=__version__,
    description="Multi-agent pipeline: public-domain story -> microdrama series script.",
)

STATIC = Path(__file__).parent / "static"
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class GenerateRequest(BaseModel):
    text: str = Field(min_length=200, max_length=60_000, description="The source story")
    title: str = ""
    episodes: int = Field(default=5, ge=1, le=10)
    languages: list[str] = Field(default_factory=lambda: ["hi"], max_length=3)
    mock: bool = Field(default=False, description="Use the offline fake LLM (no API key needed)")


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    created_at: float
    finished_at: float | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None
    scripts: list[str] = []


def _has_key() -> bool:
    provider = os.getenv("LLM_PROVIDER", "groq")
    env = PROVIDERS.get(provider, {}).get("key_env", "")
    return provider == "mock" or bool(env and os.getenv(env))


def _run_job(job_id: str, req: GenerateRequest) -> None:
    from .pipeline import Pipeline

    with _lock:
        _jobs[job_id]["status"] = "running"
    try:
        force_mock = req.mock or os.getenv("DEMO_FORCE_MOCK", "").lower() == "true" or not _has_key()
        overrides: dict[str, Any] = {"num_episodes": req.episodes, "target_languages": req.languages}
        if force_mock:
            overrides.update(provider="mock", critic_provider="mock", embed_model="hash")
        pipe = Pipeline(get_settings(**overrides), run_id=job_id)
        state = pipe.run(req.text, req.title)
        with _lock:
            _jobs[job_id].update(
                status="done",
                finished_at=time.time(),
                metrics=state["metrics"],
                outputs=state["outputs"],
                mocked=force_mock,
            )
    except Exception as e:  # surface the reason to the client, keep the server alive
        traceback.print_exc()
        with _lock:
            _jobs[job_id].update(status="failed", finished_at=time.time(), error=f"{type(e).__name__}: {e}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "llm_key_configured": _has_key()}


@app.post("/generate", response_model=JobStatus, status_code=202)
def generate(req: GenerateRequest, background: BackgroundTasks) -> JobStatus:
    job_id = uuid.uuid4().hex[:10]
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "status": "queued", "created_at": time.time()}
    background.add_task(_run_job, job_id, req)
    return JobStatus(**_jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job(job_id: str) -> JobStatus:
    with _lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    scripts = [k.removeprefix("script_") for k in (j.get("outputs") or {}) if k.startswith("script_")]
    return JobStatus(**{k: v for k, v in j.items() if k in JobStatus.model_fields}, scripts=scripts)


@app.get("/jobs/{job_id}/script", response_class=PlainTextResponse)
def script(job_id: str, lang: str = "en") -> str:
    with _lock:
        j = _jobs.get(job_id)
    if not j or j.get("status") != "done":
        raise HTTPException(404, "job not found or not finished")
    path = (j.get("outputs") or {}).get(f"script_{lang}")
    if not path:
        raise HTTPException(404, f"no script for language {lang!r}")
    return Path(path).read_text(encoding="utf-8")


@app.get("/sample", response_class=PlainTextResponse)
def sample() -> str:
    p = Path(__file__).resolve().parent.parent / "data" / "stories" / "the_potters_daughter.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
