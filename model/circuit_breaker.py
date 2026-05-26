"""Small shared circuit breaker for structured LLM provider calls."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SimpleCircuitBreaker:
    """Open after a fixed number of consecutive provider failures."""

    def __init__(self, failure_threshold: int = 5) -> None:
        """Create a breaker with a consecutive failure threshold."""
        self._failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._open = False

    def before_call(self) -> None:
        """Raise immediately when the circuit is already open."""
        if self._open:
            raise RuntimeError("circuit open")

    def record_success(self) -> None:
        """Reset the breaker after any successful provider call."""
        was_open = self._open or self._consecutive_failures > 0
        self._consecutive_failures = 0
        self._open = False
        if was_open:
            logger.info("llm_circuit_reset")

    def record_failure(self) -> None:
        """Increment failure count and open the circuit when threshold is reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold and not self._open:
            self._open = True
            logger.warning("llm_circuit_open")
