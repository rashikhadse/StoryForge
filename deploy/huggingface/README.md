# Deploy StoryForge to Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space: name `storyforge`, SDK **Docker**, hardware **CPU basic (free)**.
2. In the Space's **Settings → Variables and secrets**, add secrets `GROQ_API_KEY` and `GEMINI_API_KEY`, and variables `LLM_PROVIDER=groq`, `CRITIC_PROVIDER=gemini`, `NUM_EPISODES=3`.
   Leave the keys out (or set `DEMO_FORCE_MOCK=true`) to run the public demo in offline mock mode.
3. Push this repo to the Space. The Space needs its own README with this front matter at the top:

```yaml
---
title: StoryForge
emoji: 🎬
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---
```

Quick way to push:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/storyforge
cp deploy/huggingface/space_header.md /tmp/h && cat /tmp/h README.md > /tmp/r && mv /tmp/r README.md
git commit -am "HF Space header" && git push space main
git reset --hard HEAD~1   # keep the GitHub README clean
```

The Dockerfile already listens on port 7860 and runs as a non-root user, which Spaces requires.
