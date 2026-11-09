"""Unit tests for recorder/redact.py (M2-T1). Pure function, no mocking, no live calls."""

from recorder.redact import redact_text


def test_redacts_groq_api_key():
    text = "my key is gsk_abcdefghijklmnopqrstuvwxyz1234"
    redacted, count = redact_text(text)
    assert "[REDACTED:api_key]" in redacted
    assert "gsk_" not in redacted
    assert count == 1


def test_redacts_openai_api_key():
    text = "set OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    redacted, count = redact_text(text)
    assert "[REDACTED:api_key]" in redacted
    assert count == 1


def test_redacts_email():
    text = "contact support at press@apple.example for help"
    redacted, count = redact_text(text)
    assert redacted == "contact support at [REDACTED:email] for help"
    assert count == 1


def test_redacts_account_number():
    text = "card number 4111 1111 1111 1111 on file"
    redacted, count = redact_text(text)
    assert "[REDACTED:account]" in redacted
    assert "4111" not in redacted
    assert count == 1


def test_redacts_multiple_occurrences():
    text = "emails: a@example.com and b@example.com"
    redacted, count = redact_text(text)
    assert count == 2
    assert "@" not in redacted


def test_no_redaction_needed():
    text = "Apple Inc. was founded by Steve Jobs."
    redacted, count = redact_text(text)
    assert redacted == text
    assert count == 0
