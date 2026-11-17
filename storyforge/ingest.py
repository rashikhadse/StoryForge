"""Load and clean messy source text (Project Gutenberg, Wikisource, OCR dumps)."""

from __future__ import annotations

import html
import re
import unicodedata
import urllib.request
from pathlib import Path

MAX_CHARS = 30_000  # roughly 8k tokens; keeps extraction inside free-tier context limits

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE | re.DOTALL)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|nav|footer|header).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</h\d>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def clean_text(raw: str, max_chars: int = MAX_CHARS) -> str:
    text = unicodedata.normalize("NFC", raw)
    if "<html" in text[:2000].lower() or "<p>" in text[:5000].lower():
        text = strip_html(text)
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end() :]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = re.sub(r"\[(Illustration|Footnote|Sidenote)[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"(?m)^\s*(\d{1,4}|[ivxlcdm]{1,6}|page \d+)\s*$", "", text, flags=re.IGNORECASE)  # page numbers
    text = re.sub(r"[ \t]+", " ", text)
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    text = "\n\n".join(p for p in paragraphs if p)
    if len(text) > max_chars:
        cut = text.rfind("\n\n", 0, max_chars)
        text = text[: cut if cut > max_chars * 0.6 else max_chars] + "\n\n[...source truncated...]"
    return text.strip()


def load_story(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def fetch_story(url: str, out_path: str | Path) -> Path:
    """Download a public-domain story (Gutenberg .txt or Wikisource page) and save it cleaned."""
    req = urllib.request.Request(url, headers={"User-Agent": "storyforge/0.1 (educational)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(clean_text(raw, max_chars=10**9), encoding="utf-8")
    return out
