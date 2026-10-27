"""Text checks (spec 4.2: explicit sexual language in text overlays; section 7: memes are fine
unless the text itself violates). Shared by OCR on images / video frames and by speech.

Whole-word (or whole-phrase) matching against the word lists in settings.gate2.text, minus
the pet-platform allow list ("bitch" = female dog, "cock" = rooster...).
"""
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.core.config import settings


def read_list(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {" ".join(tokens(line)) for line in lines if line.strip() and not line.startswith("#")} - {""}


def tokens(text: str) -> list[str]:
    """Lower-case words. Splits on spaces and strips punctuation at the ends, so Devanagari
    vowel signs stay inside their word (a regex \\w split would cut them off)."""
    words = []
    for raw in unicodedata.normalize("NFKC", text).lower().split():
        word = "".join(c for c in raw if not unicodedata.category(c).startswith(("P", "S")))
        if word:
            words.append(word)
    return words


@lru_cache(maxsize=1)
def word_lists() -> tuple[frozenset[str], frozenset[str]]:
    """(single words, multi-word phrases) to look for."""
    cfg = settings.gate2.text
    blocked = set().union(*(read_list(Path(p)) for p in cfg.blocklists))
    blocked -= set().union(*(read_list(Path(p)) for p in cfg.allow))
    return frozenset(w for w in blocked if " " not in w), frozenset(w for w in blocked if " " in w)


def text_hits(text: str) -> list[str]:
    """The listed words and phrases that appear in the text, in order of first appearance."""
    words, phrases = word_lists()
    found = tokens(text)
    joined = f" {' '.join(found)} "
    hits = [w for w in found if w in words]
    hits += [p for p in phrases if f" {p} " in joined]
    return list(dict.fromkeys(hits))


def text_signals(text: str) -> set[str]:
    return {settings.gate2.text.signal} if text_hits(text) else set()
