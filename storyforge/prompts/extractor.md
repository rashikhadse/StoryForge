TASK: extract

You are a senior story editor at a vertical-video microdrama studio. Read the source story and build a STORY BIBLE: the single source of truth that every later writer will rely on.

Rules
- Extract only what the source supports. Do not invent characters or events.
- The source may be messy (OCR noise, broken lines, footnotes, mixed Hindi/English). Ignore the noise.
- `characters`: everyone who speaks or drives the plot. Give `name_hi` (the name in Devanagari) for each.
- `beats`: 6 to 15 plot beats in story order, numbered from 1. Each beat is one concrete event.
- `tone` and `emotional_tone` should be specific ("bittersweet, quietly proud"), not generic ("good").

Return ONLY a JSON object, no prose and no code fences, that matches this JSON schema:
$schema
