import json
import re
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "safety_config.json"

_cache = None
_cache_mtime_ns = None


def _config_mtime_ns():
    try:
        return _CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        return 0


def load_safety_config(force: bool = False) -> dict:
    """Load safety_config.json, re-reading it whenever the file changes on disk.

    Editing the JSON (adding terms/patterns, refusal messages, org aliases)
    takes effect on the next request without a server restart.
    """
    global _cache, _cache_mtime_ns
    mtime = _config_mtime_ns()
    if force or _cache is None or mtime != _cache_mtime_ns:
        _cache = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        _cache_mtime_ns = mtime
    return _cache


def get_safety_config() -> dict:
    return load_safety_config()


def config_signature() -> int:
    """mtime-based signature; changes whenever safety_config.json is edited."""
    return _config_mtime_ns()


def escape_terms(terms):
    return [re.escape(str(term)) for term in terms or []]


def compile_rule_pattern(patterns) -> re.Pattern:
    """Compile a rule's `patterns` list into one combined regex.

    Each entry may be:
      {"terms": [...], "boundary": true} -> \b(?:term1|term2)\b
      {"pattern": "..."}                -> used verbatim
    All alternatives are OR-joined and compiled case-insensitively.
    """
    parts = []
    for entry in patterns or []:
        if isinstance(entry, str):
            parts.append(entry)
            continue
        if "terms" in entry:
            alternatives = "|".join(escape_terms(entry["terms"]))
            if not alternatives:
                continue
            if entry.get("boundary", True):
                parts.append(r"\b(?:" + alternatives + r")\b")
            else:
                parts.append("(?:" + alternatives + ")")
        elif "pattern" in entry:
            parts.append(entry["pattern"])
    return re.compile("|".join(parts), re.IGNORECASE)
