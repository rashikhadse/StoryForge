import time

from fastapi.testclient import TestClient

from storyforge.api import app


def test_generate_job_lifecycle(sample_text, tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/generate", json={"text": sample_text, "episodes": 3, "languages": ["hi"], "mock": True})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(50):
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "done", j.get("error")
    assert set(j["scripts"]) == {"en", "hi"}
    md = client.get(f"/jobs/{job_id}/script", params={"lang": "hi"}).text
    assert "Episode 1" in md


def test_rejects_tiny_story():
    assert TestClient(app).post("/generate", json={"text": "too short"}).status_code == 422


def test_index_and_sample():
    c = TestClient(app)
    assert "StoryForge" in c.get("/").text
    assert "Meera" in c.get("/sample").text
