"""Truncates oversized trace fields before they are written to disk (M2-T2)."""

DEFAULT_MAX_BYTES = 4096


def truncate_value(value: str, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[str, bool, int]:
    """Cuts `value` to at most `max_bytes` UTF-8 bytes.

    Returns (possibly truncated value, whether truncation happened, original byte size).
    """
    original_size = len(value.encode("utf-8"))
    if original_size <= max_bytes:
        return value, False, original_size

    encoded = value.encode("utf-8")
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True, original_size
