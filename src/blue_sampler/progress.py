"""
Console output: one status line rewritten in place, plus plain log lines.

    >>> bar = Progress("[nufft]", total=120, verbose=1)
    >>> for _ in range(120):
    ...     bar.tick(f"loss {loss:.1e}")
    >>> bar.done()
    [nufft] done in 3s

Every solver reports through this module, so all outputs share the same look:
``label k/total [▓▓▓▓░░░░] info — ~ETA remaining`` then ``label done in 12s``.
"""
from __future__ import annotations

import sys
import time

BAR_WIDTH = 20
_line = {"len": 0}          # length of what is currently written on the status line


def fmt_duration(seconds: float) -> str:
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}min{rest:02d}"


def _write(text: str, newline: bool) -> None:
    pad = " " * max(0, _line["len"] - len(text))       # erase the previous, longer, status
    sys.stdout.write(f"\r{text}{pad}" + ("\n" if newline else ""))
    sys.stdout.flush()
    _line["len"] = 0 if newline else len(text)


def log(message: str, verbose: int = 1) -> None:
    """A plain, permanent line."""
    if verbose >= 1:
        _write(message, newline=True)


class Progress:
    """Status line for one task. ``tick()`` = one unit of work completed."""

    def __init__(self, label: str, total: int | None = None, verbose: int = 1):
        self.label, self.total, self.verbose = label, total, verbose
        self.count = 0
        self.start()

    def start(self) -> None:
        """(Re)start the clock, e.g. once compilation or child tasks are over."""
        self._t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def note(self, message: str) -> None:
        """Show `label message` on the status line."""
        if self.verbose >= 1:
            _write(f"{self.label} {message}", newline=False)

    def tick(self, info: str = "") -> None:
        self.count += 1
        if self.verbose < 1:
            return
        parts = []
        if self.total:
            filled = min(BAR_WIDTH, BAR_WIDTH * self.count // self.total)
            parts.append(f"{self.count}/{self.total} [{'▓' * filled}{'░' * (BAR_WIDTH - filled)}]")
        if info:
            parts.append(info)
        if self.total:
            remaining = self.elapsed / self.count * (self.total - self.count)
            parts.append(f"— ~{fmt_duration(remaining)} remaining")
        self.note(" ".join(parts))

    def done(self, info: str = "") -> None:
        if self.verbose >= 1:
            _write(f"{self.label} done in {fmt_duration(self.elapsed)}" + (f" {info}" if info else ""),
                   newline=True)
