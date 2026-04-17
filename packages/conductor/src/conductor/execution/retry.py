"""Retry configuration for node execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryConfig:
    """Global retry configuration, applied to all nodes unless the node
    has its own max_retries set.

    Args:
        max_retries: Maximum retry attempts (0 = no retry).
        delay: Initial delay in seconds between retries.
        backoff_factor: Multiplier applied to delay after each retry.
                        E.g., delay=1, backoff_factor=2 → waits 1s, 2s, 4s, ...
    """

    max_retries: int = 0
    delay: float = 1.0
    backoff_factor: float = 2.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt (1-indexed)."""
        return self.delay * (self.backoff_factor ** (attempt - 1))


# Sentinel: no retry
NO_RETRY = RetryConfig(max_retries=0)
