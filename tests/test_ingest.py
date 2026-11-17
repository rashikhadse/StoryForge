from storyforge.ingest import clean_text


def test_strips_gutenberg_boilerplate_and_noise():
    raw = (
        "Licence blah\n*** START OF THE PROJECT GUTENBERG EBOOK TALES ***\n\n"
        "The potter lived by the river. He was very hap-\npy.\n\n12\n\n"
        "[Illustration: a pot]\nThe end.\n*** END OF THE PROJECT GUTENBERG EBOOK TALES ***\nmore licence"
    )
    out = clean_text(raw)
    assert "Licence" not in out and "licence" not in out
    assert "happy" in out
    assert "Illustration" not in out
    assert "\n12\n" not in out


def test_strips_html():
    out = clean_text("<html><body><nav>menu</nav><p>Once upon a time.</p><p>The end.</p></body></html>")
    assert "menu" not in out and "Once upon a time." in out


def test_truncates_long_text():
    out = clean_text(("word " * 50 + "\n\n") * 500, max_chars=2000)
    assert len(out) < 2100 and out.endswith("[...source truncated...]")
