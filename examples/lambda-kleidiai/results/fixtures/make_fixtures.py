#!/usr/bin/env python3
"""Write the synthetic fixture runs in results/fixtures/. They exist to show and test the format before P1's
baseline is measured: every file has "fixture": true, and the dashboard labels them FIXTURE and hides them once
any real run is published. Deterministic (fixed seed). Standard library only.

    python examples/lambda-kleidiai/results/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lkai_results as lr  # noqa: E402

BASE_SHA = "7147a308205e1e51f8403b321d5d57b367c158d7"


def sample(rng: random.Random, mu: float, sd: float, n: int) -> list[float]:
    return [max(0.001, rng.gauss(mu, sd)) for _ in range(n)]


def classes(rng: random.Random, dl: float, warm: float) -> dict:
    cold_n, rest_n, warm_n = 50, 50, 100
    cold = {
        "n": cold_n,
        "weight_download": lr.dist(sample(rng, dl, dl * 0.08, cold_n)),
        "model_load_mmap": lr.dist(sample(rng, 0.9, 0.05, cold_n)),
        "context_init": lr.dist(sample(rng, 0.12, 0.01, cold_n)),
        "warmup_inference": lr.dist(sample(rng, 0.6, 0.04, cold_n)),
        "first_request_inference": lr.dist(sample(rng, 0.62, 0.05, cold_n)),
        "report_init_duration": lr.dist(sample(rng, dl + 2.2, 0.3, cold_n)),
        "first_request_total": lr.dist(sample(rng, dl + 2.9, 0.35, cold_n)),
        "report_max_memory_mb": lr.dist(sample(rng, 4300, 40, cold_n), unit="MB"),
    }
    cold["model_ready"] = cold["report_init_duration"]
    restore = {
        "n": rest_n,
        "report_restore_duration": lr.dist(sample(rng, 3.9, 0.4, rest_n)),
        "restore_to_invoke": lr.dist(sample(rng, 0.004, 0.001, rest_n)),
        "first_request_inference": lr.dist(sample(rng, 0.85, 0.08, rest_n)),
        "first_request_total": lr.dist(sample(rng, 4.85, 0.45, rest_n)),
        "snapshot_init": {
            "weight_download": lr.dist(sample(rng, dl, dl * 0.08, 10)),
            "model_load_mmap": lr.dist(sample(rng, 0.9, 0.05, 10)),
        },
    }
    warm_d = {
        "n": warm_n,
        "steady_state_inference": lr.dist(sample(rng, warm, warm * 0.05, warm_n)),
        "client_round_trip": lr.dist(sample(rng, warm + 0.06, warm * 0.06, warm_n)),
        "report_billed_duration": lr.dist(sample(rng, warm + 0.01, warm * 0.05, warm_n)),
    }
    return {"cold": cold, "restore": restore, "warm": warm_d}


def common(experiment_id: str, run_id: str, status: str) -> dict:
    return {
        "schema": lr.SCHEMA_ID,
        "fixture": True,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "status": status,
        "commits": {"experiment": BASE_SHA, "baseline": BASE_SHA, "repo_dirty_at_build": False},
        "deployment": {
            "region": "us-east-2",
            "functions": {"inst": "lkai-p1-inst", "rep": "lkai-p1-rep"},
            "runtime": "python3.12",
            "runtime_version_arn": None,
            "architecture": "arm64",
            "memory_mb": 10240,
            "snapstart": {"ApplyOn": "PublishedVersions"},
            "az": "not exposed to the function; Lambda chooses per environment",
        },
        "model": {
            "id": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
            "revision": "216d1ccb",
            "format": "GGUF Q8_0",
            "gguf_bytes": 2_147_483_648,
        },
        "conditions": {
            "rounds": 5,
            "concurrency": 10,
            "warm_per_arm_per_round": 20,
            "arms": ["inst", "rep"],
            "request_ids": ["probe", "r1", "r2", "r3", "r4"],
            "started": "2026-10-20T00:00:00Z",
            "finished": "2026-10-20T00:30:00Z",
            "weight_fetch": "HEAD + parallel 64 MiB ranged GETs (32 threads) into a memfd",
        },
        "classification": {
            "method": "FIXTURE: synthetic",
            "expected_vs_observed": {"cold->cold": 50, "restore->restore": 50, "warm->warm": 100},
            "mismatches": 0,
        },
        "errors": {"count": 0, "rows": []},
        "cost": {
            "billed_duration_s": 512.4,
            "total_usd": 6.12,
            "s3_request_usd": 0.0012,
            "s3_gb_transferred": 214.7,
        },
        "correctness": {
            "compared": 100,
            "bit_identical": 100,
            "choice_flips": 0,
            "max_abs_delta": 0.0,
        },
        "profiling_artifacts": [],
        "artifacts": {"invocations": "invocations.jsonl"},
        "reproduce": ["FIXTURE: no command; synthetic data from results/fixtures/make_fixtures.py"],
    }


def main() -> None:
    rng = random.Random(20261020)
    runs = []

    r = common("P1-baseline", "fixture-baseline", "baseline")
    r |= {
        "hypothesis": "FIXTURE. Baseline: the pinned reference.",
        "changed_variable": None,
        "conclusion": None,
    }
    r["phases"] = {"inst": classes(rng, 25.0, 0.80)}
    runs.append(r)

    r = common("E3-parallel-range-get", "fixture-improved", "improved")
    r |= {
        "hypothesis": "FIXTURE. More parallel ranged GETs cut weight download time.",
        "changed_variable": {"name": "range_get_threads", "baseline": 32, "experiment": 64},
        "conclusion": "FIXTURE. Cold first request p50 falls; warm unchanged; answers bit-identical.",
    }
    r["commits"]["experiment"] = "0123456789abcdef0123456789abcdef01234567"
    r["phases"] = {"inst": classes(rng, 18.0, 0.80)}
    runs.append(r)

    r = common("E2-onezone-ia", "fixture-null", "no-change")
    r |= {
        "hypothesis": "FIXTURE. One Zone-IA storage class changes download time.",
        "changed_variable": {
            "name": "s3_storage_class",
            "baseline": "STANDARD",
            "experiment": "ONEZONE_IA",
        },
        "conclusion": "FIXTURE. Null result: within run-to-run variance; download is bound by the per-environment network quota.",
    }
    r["phases"] = {"inst": classes(rng, 25.1, 0.80)}
    r["phases"]["inst"]["restore"]["restore_to_invoke"] = None
    runs.append(r)

    r = common("E1-s3-express", "fixture-failed", "failed")
    r |= {
        "hypothesis": "FIXTURE. An S3 Express One Zone directory bucket lowers first-byte latency.",
        "changed_variable": {
            "name": "weights_bucket_type",
            "baseline": "general-purpose",
            "experiment": "directory",
        },
        "conclusion": "FIXTURE. Failed: every cold start timed out reading weights (AccessDenied on CreateSession).",
        "errors": {
            "count": 20,
            "rows": [
                {
                    "arm": "inst",
                    "round": 1,
                    "wave": "cold",
                    "error": "AccessDenied",
                    "detail": "s3express:CreateSession",
                }
            ],
        },
        "correctness": None,
        "phases": {},
        "cost": {"billed_duration_s": 18000.0, "total_usd": 3.01},
    }
    r.pop("classification")
    runs.append(r)

    for r in runs:
        lr.check(r)
        d = os.path.join(HERE, r["experiment_id"], r["run_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "results.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, sort_keys=True)
            f.write("\n")
        with open(os.path.join(d, "invocations.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"fixture": True, "note": "synthetic placeholder"}) + "\n")
        print("wrote", os.path.relpath(d, HERE))


if __name__ == "__main__":
    main()
