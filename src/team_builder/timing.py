"""Timing utilities.

The prototype evaluation framework includes runtime and scalability as practical
comparison dimensions. This module provides a small timing helper that can be
used by the pipeline without introducing external dependencies.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


@dataclass
class PipelineTimer:
    """Collect elapsed seconds for named pipeline stages."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a named stage and accumulate its elapsed seconds."""

        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + elapsed

    def total_seconds(self) -> float:
        """Return the sum of all recorded stage durations."""

        return sum(self.stages.values())

    def as_ordered_dict(self) -> dict[str, float]:
        """Return stage timings in insertion order plus total runtime."""

        return {
            **self.stages,
            "total_runtime_seconds": self.total_seconds(),
        }
