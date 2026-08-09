
"""Parser registry and auto-detection protocol.

Each parser module exposes a `parse(data_log, source, **opts)` function plus an
optional `detect(head, filename)` callable.  The registry maps format ids to
(parse_fn, detect_fn) pairs so the CLI can resolve AUTO input.
"""
from __future__ import annotations

from typing import Callable, Optional

# format_id -> (parse_fn, detect_fn_or_None)
PARSER_REGISTRY = {}


def register_parser(format_id: str, parse_fn: Callable, detect_fn: Optional[Callable] = None):
    """Register a parser under a format id (e.g. 'CSV', 'RCZ', 'VBO')."""
    PARSER_REGISTRY[format_id] = (parse_fn, detect_fn)


def resolve_parser(format_id: str):
    """Return the parse function for a format id, raising KeyError if unknown."""
    return PARSER_REGISTRY[format_id][0]


def detect_format(filename: str, head: str) -> str:
    """Return the format id whose detect_fn matches, or 'CSV' as fallback."""
    for format_id, (_parse_fn, detect_fn) in PARSER_REGISTRY.items():
        if detect_fn is not None and detect_fn(filename, head):
            return format_id
    return "CSV"
