"""Stable hashing of config dicts, used as cache keys for preprocessing variants and runs."""
import hashlib
import json
from typing import Any


def stable_hash(obj: Any, length: int = 16) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
