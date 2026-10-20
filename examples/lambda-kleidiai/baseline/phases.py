"""Low-overhead phase timers for the lambda-kleidiai baseline. Standard library only.

Every duration comes from time.monotonic_ns() (CLOCK_MONOTONIC) and is never compared across a
SnapStart snapshot/restore boundary: a restored environment's monotonic clock is not guaranteed to
continue from the snapshot's. Cross-boundary facts use time.time_ns() and are labelled as wall-clock.

Each record is one stdout line, `LKAI {json}`, so it lands in CloudWatch Logs beside Lambda's own
REPORT line (Init Duration, Restore Duration, Billed Duration, Max Memory Used), which this module
never touches. `parse_report` and `parse_lkai` read both back from a log tail.

LKAI_PHASES=0 in the function's environment turns every record off (lkai_entry.handler then calls
the uninstrumented app.handler directly); that is the REPORT-only arm of the overhead measurement.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

PREFIX = "LKAI "
SCHEMA = "lkai-phases/1"


def enabled(env: dict[str, str] | None = None) -> bool:
    return (os.environ if env is None else env).get("LKAI_PHASES", "1") != "0"


def runtime_to_import_s() -> float | None:
    """Seconds from this process's start to now, from /proc/self/stat and CLOCK_BOOTTIME.

    At module import this is the runtime bootstrap's share of the init (interpreter start, the
    runtime client, our imports up to here). None where /proc or CLOCK_BOOTTIME is missing."""
    try:
        with open("/proc/self/stat", encoding="ascii") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])  # field 22 overall; 20th after the comm field's ')'
        boot = time.clock_gettime(time.CLOCK_BOOTTIME)
        return round(boot - start_ticks / os.sysconf("SC_CLK_TCK"), 4)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class Recorder:
    """Phase durations of one execution environment.

    `init` holds the init phases (seconds), filled by `phase()`; `counters` holds counts such as
    bytes downloaded. Thread-safe enough for the handler's one background load thread: each key
    is written by one thread."""

    def __init__(
        self,
        t0_ns: int | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        wall: Callable[[], int] = time.time_ns,
        write: Callable[[str], Any] | None = None,
    ):
        self.clock, self.wall = clock, wall
        self.t0 = clock() if t0_ns is None else t0_ns
        self.wall0 = wall()
        self.init: dict[str, float] = {}
        self.counters: dict[str, Any] = {}
        self.restored = False
        self.restore_mono: int | None = None
        self.restore_wall: int | None = None
        self.ready_mono: int | None = None
        self._write = write or _stdout
        self._lock = threading.Lock()

    def since_t0(self) -> float:
        return (self.clock() - self.t0) / 1e9

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = self.clock()
        try:
            yield
        finally:
            self.add(name, (self.clock() - start) / 1e9)

    def add(self, name: str, seconds: float) -> None:
        with self._lock:
            self.init[name] = round(self.init.get(name, 0.0) + seconds, 6)

    def count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        rec = {"schema": SCHEMA, "kind": kind, "wall_ns": self.wall(), **fields}
        self._write(PREFIX + json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        return rec


def _stdout(line: str) -> None:
    sys.stdout.write(line)
    sys.stdout.flush()


def timed(rec: Recorder, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """fn wrapped so each call's duration adds to rec.init[name]."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with rec.phase(name):
            return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def first_call_timed(rec: Recorder, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """fn wrapped so its FIRST call's duration is rec.init[name]; later calls run untimed."""
    done = False

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal done
        if done:
            return fn(*args, **kwargs)
        done = True
        with rec.phase(name):
            return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def start_class(n: int, init_type: str | None, restored: bool) -> str:
    """cold, restore or warm for this environment's n-th invocation (1-based)."""
    if n > 1:
        return "warm"
    if restored:
        return "restore"
    return "cold" if init_type == "on-demand" else "init-other"


def observed_class(report: dict[str, float]) -> str:
    """The start class Lambda's REPORT line shows: Restore Duration -> restore, Init Duration -> cold, else warm."""
    if "Restore Duration" in report:
        return "restore"
    if "Init Duration" in report:
        return "cold"
    return "warm" if report else "unknown"


# ---------------------------------------------------------------------------- log parsing

_REPORT_NUM = re.compile(r"\s*([\d.]+)")


def parse_report(log: str) -> dict[str, float]:
    """The last REPORT line's numeric fields, e.g. {"Duration": 812.3, "Restore Duration": 410.2,
    "Billed Restore Duration": 296, "Max Memory Used": 4321}. Units are Lambda's (ms, MB)."""
    out: dict[str, float] = {}
    for line in log.splitlines():
        if not line.startswith("REPORT"):
            continue
        out = {}
        for part in line.split("\t")[1:]:
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            m = _REPORT_NUM.match(value)
            if m:
                out[key.strip()] = float(m.group(1))
    return out


def parse_lkai(log: str) -> list[dict[str, Any]]:
    """Every `LKAI {json}` record in a log text, in order. Lines that do not parse are skipped."""
    recs = []
    for line in log.splitlines():
        i = line.find(PREFIX + "{")
        if i < 0:
            continue
        try:
            recs.append(json.loads(line[i + len(PREFIX) :]))
        except ValueError:
            continue
    return recs
