#!/usr/bin/env python3
"""Compare normalized baseline/candidate vLLM GPU performance metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_thresholds(values: list[str], default: float) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid threshold {value!r}; expected metric=percent")
        result[name] = float(raw)
    result["*"] = default
    return result


def metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("metrics", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"metrics must be a JSON object: {path}")
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float))
    }


def higher_is_better(name: str) -> bool:
    lowered = name.lower()
    return (
        "throughput" in lowered
        or "tokens_per_second" in lowered
        or lowered.endswith("_tps")
    )


def compare(
    baseline: dict[str, float], candidate: dict[str, float], limits: dict[str, float]
) -> list[dict[str, object]]:
    results = []
    for name in sorted(set(baseline) & set(candidate)):
        old, new = baseline[name], candidate[name]
        if old == 0:
            delta = 0.0 if new == 0 else float("inf")
            regression = delta
        else:
            delta = (new - old) / abs(old) * 100.0
            regression = -delta if higher_is_better(name) else delta
        limit = limits.get(name, limits["*"])
        results.append(
            {
                "metric": name,
                "baseline": old,
                "candidate": new,
                "delta_percent": round(delta, 4),
                "regression_percent": round(regression, 4),
                "threshold_percent": limit,
                "passed": regression <= limit,
                "direction": "higher_is_better"
                if higher_is_better(name)
                else "lower_is_better",
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--default-threshold-percent", type=float, default=5.0)
    parser.add_argument("--threshold", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        baseline = metrics(args.baseline)
        candidate = metrics(args.candidate)
        results = compare(
            baseline,
            candidate,
            parse_thresholds(args.threshold, args.default_threshold_percent),
        )
        if not results:
            raise ValueError("baseline and candidate have no common numeric metrics")
        passed = all(bool(result["passed"]) for result in results)
        print(
            json.dumps(
                {"status": "passed" if passed else "failed", "metrics": results},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if passed else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
