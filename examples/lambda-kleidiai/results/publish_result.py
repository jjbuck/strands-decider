#!/usr/bin/env python3
"""Publish one lambda-kleidiai experiment run to the results store and rebuild the dashboard.

    python examples/lambda-kleidiai/results/publish_result.py <run>/results.json [extra artifacts...]

Steps, all idempotent:
  1. validate the run file against results/schema.json (and the semantic checks in lkai_results.validate);
  2. upload the artifacts it lists (artifacts{}, profiling_artifacts[].path) plus any extra paths, and then its
     result.json, under runs/<experiment_id>/<run_id>/ in the store. Nothing outside that prefix is written;
  3. regenerate site/ in the store from every result.json there, and write the single-file dashboard to --out;
  4. print the Artifactory step: re-upload --out/dashboard.html as a new version of the dashboard artifact
     (get_artifact for its SHA-256, then upload_artifact_version with expectedCurrentSha256). An agent does this
     through the Artifactory MCP; a stale-SHA refusal means another run published first: rerun with --regenerate-only
     and upload again.

The store is --store or $LKAI_RESULTS_STORE: s3://bucket/prefix, or a local directory for a dry run.
Exit codes: 0 published, 2 invalid run file, 3 run already published with different content.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lkai_results as lr

# The durable dashboard: Harmony Artifactory artifact (stable /view/ URL across versions), in cluster
# https://artifactory.amazon.dev/clusters/907e0057-b2bb-46c9-abd4-6e934dbf63d4 (lambda-kleidiai experiments).
DASHBOARD_ARTIFACT_ID = os.environ.get(
    "LKAI_DASHBOARD_ARTIFACT", "aece9ccf-c1d5-4217-b31e-e5b3c78ee759"
)
DASHBOARD_URL = os.environ.get(
    "LKAI_DASHBOARD_URL", "https://artifactory.amazon.dev/view/lambda-kleidiai-experiments-results"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "result", nargs="?", help="the run's results.json (its directory is the run directory)"
    )
    ap.add_argument(
        "extra", nargs="*", help="more files or directories inside the run directory to upload"
    )
    ap.add_argument(
        "--store",
        default=os.environ.get("LKAI_RESULTS_STORE"),
        help="s3://bucket/prefix or a directory",
    )
    ap.add_argument("--region", default=os.environ.get("LKAI_RESULTS_REGION"))
    ap.add_argument("--profile", default=os.environ.get("LKAI_RESULTS_PROFILE"))
    ap.add_argument(
        "--out", default="build/lkai-dashboard", help="where the generated pages are written"
    )
    ap.add_argument(
        "--replace", action="store_true", help="overwrite this run's existing result.json"
    )
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument(
        "--regenerate-only", action="store_true", help="skip upload; rebuild the dashboard"
    )
    ap.add_argument(
        "--include-fixtures", action="store_true", help="show fixture runs beside real ones"
    )
    a = ap.parse_args(argv)

    if not a.regenerate_only:
        if not a.result:
            ap.error("result is required unless --regenerate-only")
        with open(a.result, encoding="utf-8") as f:
            res = json.load(f)
        problems = lr.validate(res)
        if problems:
            print(f"INVALID {a.result}", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            return 2
        print(
            f"valid {lr.SCHEMA_ID}: {res['experiment_id']}/{res['run_id']} status={res['status']}"
        )
        if a.validate_only:
            return 0
    if not a.store:
        ap.error("--store or LKAI_RESULTS_STORE is required")
    store = lr.open_store(a.store, a.region, a.profile)
    if not a.regenerate_only:
        try:
            rep = lr.publish_run(
                res, os.path.dirname(os.path.abspath(a.result)), store, a.extra, a.replace
            )
        except lr.PublishConflict as e:
            print(f"CONFLICT {e}", file=sys.stderr)
            return 3
        print(f"result  {rep.result_uri}")
        print(f"uploaded {len(rep.uploaded)}, unchanged {len(rep.unchanged)}")
        for m in rep.missing:
            print(f"  missing artifact (listed, not found): {m}", file=sys.stderr)
    site = lr.regenerate(store, a.out, include_fixtures=True if a.include_fixtures else None)
    dash = os.path.abspath(os.path.join(a.out, "dashboard.html"))
    print(f"dashboard rebuilt from {len(site['runs'])} run(s): {store.uri('site/index.html')}")
    print(
        "ARTIFACTORY "
        + json.dumps(
            {
                "artifactId": DASHBOARD_ARTIFACT_ID,
                "filePath": dash,
                "url": DASHBOARD_URL,
                "steps": [
                    "artifactory-mcp:get_artifact {artifactId} -> sha256",
                    "artifactory-mcp:upload_artifact_version {artifactId, filePath, expectedCurrentSha256: sha256}",
                    "stale SHA: rerun with --regenerate-only, then repeat both steps",
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
