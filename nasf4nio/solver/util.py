from __future__ import annotations

import math
import time
import sys


class Timer:
    def __init__(self, limit: float | None = None) -> None:
        self.limit = math.inf if limit is None else limit
        self.start = time.perf_counter()

    def budget(self: Timer) -> float:
        return self.limit

    def elapsed(self: Timer) -> float:
        return time.perf_counter() - self.start

    def finished(self) -> bool:
        return self.elapsed() > self.limit


def debug(*args: object) -> None:
    print(*args, file=sys.stderr)
