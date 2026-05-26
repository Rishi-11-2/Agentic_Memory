"""Prompt injection filtering for untrusted tool output text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class InjectionGuard:
    """Redact common prompt-injection patterns before tool text reaches the Actor."""

    PATTERNS: list[str] = [
        r"ignore\s+(previous|above|prior)\s+instructions?",
        r"disregard\s+.{0,30}\s+instructions?",
        r"you\s+are\s+now\s+",
        r"new\s+persona",
        r"jailbreak",
        r"system\s*prompt\s*:",
        r"<\s*/?system\s*>",
    ]

    def sanitise(self, text: str) -> str:
        """Replace matched patterns with [REDACTED] and log a warning."""
        sanitized = text
        for pattern in self.PATTERNS:
            sanitized, count = re.subn(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
            if count:
                logger.warning("tool_output_prompt_injection_redacted pattern=%s count=%s", pattern, count)
        return sanitized
