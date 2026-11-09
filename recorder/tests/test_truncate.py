"""Unit tests for recorder/truncate.py (M2-T2). Pure function, no mocking, no live calls."""

from recorder.truncate import truncate_value, DEFAULT_MAX_BYTES


def test_short_value_not_truncated():
    value = "hello world"
    result, truncated, original_size = truncate_value(value, max_bytes=4096)
    assert result == value
    assert truncated is False
    assert original_size == len(value.encode("utf-8"))


def test_long_value_is_truncated():
    value = "x" * 5000
    result, truncated, original_size = truncate_value(value, max_bytes=100)
    assert truncated is True
    assert original_size == 5000
    assert len(result.encode("utf-8")) <= 100


def test_truncation_respects_multibyte_boundaries():
    value = "é" * 100  # 2 bytes each in UTF-8
    result, truncated, original_size = truncate_value(value, max_bytes=5)
    assert truncated is True
    assert original_size == 200
    # must decode cleanly, no partial multi-byte sequence left dangling
    result.encode("utf-8")


def test_default_max_bytes_is_4kb():
    assert DEFAULT_MAX_BYTES == 4096


def test_exact_boundary_not_truncated():
    value = "x" * 10
    result, truncated, original_size = truncate_value(value, max_bytes=10)
    assert truncated is False
    assert result == value
