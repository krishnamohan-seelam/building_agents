"""Prompt definitions and prompt_helper integrations for Smart Query Router."""

from prompt_helper import PromptBuilder

EMAIL_AGENT_SYSTEM_PROMPT = """You are a professional email writing assistant.
    
Write a clear, professional email based on the user's request.
Include an appropriate subject line, greeting, body, and sign-off.
Keep the tone professional but friendly.

Format the output as:
Subject: [subject line]

[email body]
"""
HUMAN_EMAIL_REQUEST_PROMPT = """
Write an email for: {request}
"""

def build_email_draft_messages(request: str):
    """Build messages list for email draft using prompt_helper PromptBuilder."""
    return (
        PromptBuilder("intent_detection")
        .system(EMAIL_AGENT_SYSTEM_PROMPT)
        .human(HUMAN_EMAIL_REQUEST_PROMPT)
        .build(request=request)
    )
