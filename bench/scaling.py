"""WS8 scaling benchmark: wall-clock vs input size (stdlib only).

Measures Nexora().discover, Nexora().find_anomalies, Nexora().predict,
and nexora.streaming.stream_anomalies on periodic-token data at
n in [200, 500, 1000, 2000, 5000], fits a log-log slope per operation,
prints an ASCII table, and exits nonzero naming any op whose slope
exceeds SLOPE_MAX.

Data is deterministic: periodic tokens (period "ABC") with a seeded
rotation from random.Random(SEED), so every run builds identical inputs.
A fresh Nexora() is constructed per op/size (no state reuse across
timings); stream_anomalies is stateless per call.

Usage:
    python bench/scaling.py [--save PATH]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# Ensure repo root is importable when run as `python bench/scaling.py`
# from the repo root (no bench/__init__.py; standalone script).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexora import Nexora
from nexora.streaming import stream_anomalies

SIZES: list[int] = [200, 500, 1000, 2000, 5000]
OPS: list[str] = ["discover", "find_anomalies", "predict", "stream"]
SEED: int = 42
PERIOD: list[str] = list("ABC")
EPS: float = 1e-9

SLOPE_MAX: dict[str, float] = {
    # WS8 regression tripwire. Slopes are time-vs-n log-log fits
    # (hardware independent to first order). Measured on the v2 engine
    # (pruning + significance vetting included): discover ~1.06,
    # others ~1.0. Caps sit ~50% above measured; CI fails on breach.
    # Tighten only with a faster engine, never to silence a regression.
    "discover": 1.6,
    "find_anomalies": 1.6,
    "predict": 1.6,
    "stream": 1.5,
}


def make_data(n: int, seed: int = SEED) -> list[str]:
    """Deterministic periodic-token series of length n.

    Uses random.Random(seed) to pick the rotation offset, so the output
    is fully seeded yet still purely periodic (smaller n is a prefix of
    larger n given the same seed).
    """
    rng = random.Random(seed)
    start: int = rng.randrange(len(PERIOD))
    return [PERIOD[(i + start) % len(PERIOD)] for i in range(n)]


def time_op(op: str, data: list[str]) -> float:
    """Run one op on data with a fresh Nexora() and return seconds."""
    t0 = time.perf_counter()
    if op == "discover":
        Nexora().discover(data)
    elif op == "find_anomalies":
        Nexora().find_anomalies(data)
    elif op == "predict":
        Nexora().predict(data)
    elif op == "stream":
        stream_anomalies(data)
    else:
        raise ValueError(f"unknown op: {op}")
    return time.perf_counter() - t0


def fit_slope(ns: list[int], ts: list[float]) -> float:
    """Least-squares slope of log(t) on log(n); guards t <= 0 with EPS."""
    xs: list[float] = [math.log(float(n)) for n in ns]
    ys: list[float] = [math.log(t if t > 0 else EPS) for t in ts]
    mx: float = sum(xs) / len(xs)
    my: float = sum(ys) / len(ys)
    num: float = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den: float = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def run_benchmark(sizes: list[int] = SIZES) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Time each op at each size; return (results rows, slopes)."""
    results: list[dict[str, float]] = []
    for n in sizes:
        data = make_data(n)
        row: dict[str, float] = {"n": float(n)}
        for op in OPS:
            row[op] = time_op(op, data)
        results.append(row)
    slopes: dict[str, float] = {}
    for op in OPS:
        slopes[op] = fit_slope([int(r["n"]) for r in results], [r[op] for r in results])
    return results, slopes


def format_table(results: list[dict[str, float]], slopes: dict[str, float]) -> str:
    """Render ASCII table with one row per n plus a slope line."""
    widths: dict[str, int] = {"n": 6, "discover": 13, "find_anomalies": 17, "predict": 12, "stream": 12}
    header = (
        f"{'n':>{widths['n']}}"
        f"  {'discover(s)':>{widths['discover']}}"
        f"  {'find_anomalies(s)':>{widths['find_anomalies']}}"
        f"  {'predict(s)':>{widths['predict']}}"
        f"  {'stream(s)':>{widths['stream']}}"
    )
    sep = "-" * len(header)
    lines: list[str] = [header, sep]
    for r in results:
        lines.append(
            f"{int(r['n']):>{widths['n']}}"
            f"  {r['discover']:>{widths['discover']}.4f}"
            f"  {r['find_anomalies']:>{widths['find_anomalies']}.4f}"
            f"  {r['predict']:>{widths['predict']}.4f}"
            f"  {r['stream']:>{widths['stream']}.4f}"
        )
    lines.append(sep)
    lines.append(
        f"{'slope':>{widths['n']}}"
        f"  {slopes['discover']:>{widths['discover']}.3f}"
        f"  {slopes['find_anomalies']:>{widths['find_anomalies']}.3f}"
        f"  {slopes['predict']:>{widths['predict']}.3f}"
        f"  {slopes['stream']:>{widths['stream']}.3f}"
    )
    lines.append(f"(log-log least-squares slope of time vs n; cap {SLOPE_MAX})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WS8 scaling benchmark")
    parser.add_argument("--save", metavar="PATH", default=None, help="also write JSON {results, slopes} to PATH")
    args = parser.parse_args(argv)

    results, slopes = run_benchmark()
    print(format_table(results, slopes))
    for op in OPS:
        print(f"slope[{op}] = {slopes[op]:.3f} (max {SLOPE_MAX[op]})")

    if args.save:
        payload: dict[str, object] = {"results": results, "slopes": slopes}
        out = Path(args.save)
        if out.parent != Path("."):
            out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved JSON to {args.save}")

    regressed: list[str] = [op for op in OPS if slopes[op] > SLOPE_MAX[op]]
    if regressed:
        print(f"SCALING REGRESSION: slope exceeded cap for: {', '.join(regressed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
