TASK: plan

You are a showrunner adapting a story into a vertical microdrama series for Indian audiences (Kuku TV style): episodes of 60 to 90 seconds, watched on a phone, with a drop-off risk in every second.

Split the story into exactly $num_episodes episodes.
- Use every beat exactly once, in order. `beat_orders` lists the beat numbers each episode covers.
- `hook`: a concrete image, line or conflict in the first 5 seconds. Never start with exposition.
- `cliffhanger`: end every episode on an open question, except the final one, which resolves the story.
- Every hook must be distinct; do not reuse the same device twice.

Return ONLY a JSON object, no prose and no code fences, that matches this JSON schema:
$schema
