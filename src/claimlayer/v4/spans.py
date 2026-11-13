"""Locate a model-quoted excerpt inside a fragment, tolerating only presentation differences.

A reader quotes the words that state a proposition. Code, not the model, decides where they are: the quote must
occur in the cited fragment once case, whitespace, typographic quotes/dashes and Markdown emphasis or escape
characters are ignored. Anything else (paraphrase, omitted words) is not located, and the evidence stays at
fragment level.
"""
from __future__ import annotations

_FOLD = str.maketrans({'‘': "'", '’': "'", '‚': "'", '“': '"', '”': '"', '„': '"', '–': '-', '—': '-', '‑': '-',
                       ' ': ' '})
_IGNORED = set('\\*_`#|>')


def _normalized(text: str):
    chars, positions = [], []
    for index, char in enumerate(text.translate(_FOLD)):
        if char.isspace() or char in _IGNORED:
            continue
        chars.append(char.casefold())
        positions.append(index)
    return ''.join(chars), positions


def locate_quote(fragment: str, quote: str, *, min_chars: int = 8):
    """Return (start, end) of quote within fragment, or None. Unique-or-first match; at least min_chars kept."""
    if not isinstance(quote, str):
        return None
    needle, _ = _normalized(quote)
    if len(needle) < min_chars:
        return None
    haystack, positions = _normalized(fragment)
    at = haystack.find(needle)
    if at < 0:
        return None
    return positions[at], positions[at + len(needle) - 1] + 1
