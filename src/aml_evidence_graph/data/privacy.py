"""Data-minimizing tokenization helpers for internal and demo boundaries."""

from __future__ import annotations

import hashlib
import hmac


def tokenise_identifier(identifier: str, *, secret: str, namespace: str) -> str:
    """Create a deterministic non-reversible token for an internal identifier."""
    if not identifier:
        raise ValueError("identifier must be non-empty")
    if not secret:
        raise ValueError("secret must be non-empty")
    if not namespace:
        raise ValueError("namespace must be non-empty")

    payload = f"{namespace}:{identifier}".encode()
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{namespace}_{digest[:20]}"

