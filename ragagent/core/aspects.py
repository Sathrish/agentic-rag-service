"""Rule-based extraction of query aspects (sub-topics)."""
import re

from . import config

QUESTION_PREFIX = re.compile(
    r"^(what|how|why|when|where|who|which|do|does|can|is|are)\s+", re.IGNORECASE
)
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "about", "into", "through",
    "what", "how", "why", "when", "where", "who", "which", "your",
    "their", "its", "this", "that", "these", "those",
})
# Keep letters, digits and inner hyphens/apostrophes ("covid-19", "omega-3").
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _clean(text: str) -> str:
    """Lowercase and drop punctuation, keeping words separated by single spaces."""
    return " ".join(_WORD.findall(text.lower()))


def extract_aspects(query: str, max_aspects: int = config.MAX_ASPECTS, min_word_len: int = 4):
    """Split a query into at most ``max_aspects`` short aspect phrases.

    1. Strip a leading question word.
    2. If the query contains "and", each side of it is an aspect.
    3. Otherwise drop stopwords and words shorter than ``min_word_len``, then
       pair the remaining content words into bigrams.
    """
    cleaned = QUESTION_PREFIX.sub("", query.strip())

    if " and " in cleaned.lower():
        parts = [_clean(p) for p in re.split(r"\band\b", cleaned, flags=re.IGNORECASE)]
        parts = [p for p in parts if len(p) > 3]
        if len(parts) >= 2:
            return parts[:max_aspects]

    words = [
        w for w in _WORD.findall(cleaned.lower())
        if w not in STOPWORDS and len(w) >= min_word_len
    ]
    aspects = [f"{words[i]} {words[i + 1]}" for i in range(0, len(words) - 1, 2)]
    if len(words) % 2 == 1:
        aspects.append(words[-1])

    if not aspects:
        fallback = _clean(cleaned)
        aspects = [fallback] if fallback else []
    return aspects[:max_aspects]
