"""A small synchronous client for a running decision-model server.

Deliberately `urllib` rather than a client library: these examples should need nothing
beyond `strands-agents`, so that a reader can drop one into their own project without
first agreeing to a dependency.

**AWS Lambda.** A URL that is a Lambda Function URL (`https://<id>.lambda-url.<region>.on.aws`,
as examples/lambda deploys) takes only SigV4-signed requests, so requests to one are signed
with botocore and the usual AWS credentials. botocore comes with `strands-agents`, through
boto3; nothing else is signed.

Synchronous on purpose too. Strands calls a handler method that is a plain `def`
directly, so a blocking request is the simplest thing that works -- and it keeps requests
to the server strictly sequential, which is how it is best driven: one request at a time.

**Renaming.** The product name is spelled out in exactly one place, `NAME` below, and the
things derived from it sit directly underneath. Everything else -- this module's filename,
the `Decider` class, the demos -- is named for what it does rather than what it is called,
so a future rename is an edit to this block and nothing else.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

# --------------------------------------------------------------------------- the name

NAME = "strands-decider"
"""What this thing is called today. The only place the name is written."""

ENV_URL = "STRANDS_DECIDER_URL"
"""Environment variable that overrides the endpoint. Derived from NAME, so rename together."""

SERVE_HINT = "strands-decider serve <checkpoint> --port 8099"
"""How to start a server, shown when nothing is listening.

This is the upstream CLI entry point (`[project.scripts]` in `pyproject.toml`), which is a
separate thing from NAME: it only changes if the Python package itself is renamed.
"""

MODEL_ID = "hobson-latest"
"""Sent as `model` on every request. The server ignores it and answers as whatever
checkpoint it loaded, so this is only here to show the full wire shape."""

DEFAULT_PORT = 8099
DEFAULT_URL = os.environ.get(ENV_URL, f"http://127.0.0.1:{DEFAULT_PORT}")


class ServerUnavailable(RuntimeError):
    """The server is not answering. Raised with the command that starts one."""


class Decider:
    """One state, many typed questions, one round trip.

    The questions cannot see each other, so their answers stay independent -- but the
    state is encoded once and shared across them, which is why asking eight things costs
    far less than eight requests. Ask everything you need at once.
    """

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 120.0) -> None:
        self.url = url.rstrip("/")
        # Generous: on Apple silicon the *first* request at a given input length pays a
        # one-off MPS shape compile, which is seconds on a long document.
        self.timeout = timeout
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None

    # ---- questions -------------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true), and with two outcomes that *is* the confidence.

        `criteria` optionally describes what `"true"` and `"false"` mean. It is worth
        writing: a noul renders as a two-option list, so those descriptions are read the
        same way a choice's options are, and they sharpen the boundary considerably.
        """
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = dict(criteria)
        return question

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means.

        The option names are read from this request, not baked into the model, so you can
        change them without retraining anything.
        """
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking ----------------------------------------------------------

    def _headers(self, method: str, url: str, body: bytes | None) -> dict[str, str]:
        """JSON headers, SigV4-signed when `url` is a Lambda Function URL (IAM auth)."""
        headers = {"content-type": "application/json"}
        host = urllib.parse.urlparse(url).hostname or ""
        if not (host.endswith(".on.aws") and ".lambda-url." in host):
            return headers
        import botocore.session
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials = botocore.session.get_session().get_credentials()
        if credentials is None:
            raise ServerUnavailable(
                f"{self.url} is a Lambda Function URL; signing needs AWS credentials"
            )
        region = host.split(".lambda-url.", 1)[1].split(".", 1)[0]
        request = AWSRequest(method=method, url=url, data=body, headers=headers)
        SigV4Auth(credentials.get_frozen_credentials(), "lambda", region).add_auth(request)
        return dict(request.headers)

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        return ServerUnavailable(
            f"no {NAME} at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers come back under the same keys."""
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        url = f"{self.url}/v1/systemone"
        request = urllib.request.Request(url, data=body, headers=self._headers("POST", url, body))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving."""
        url = f"{self.url}/health"
        request = urllib.request.Request(url, headers=self._headers("GET", url, None))
        try:
            # A Lambda environment starting up answers once its model has loaded.
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        h = self.health()
        # `serve` names its checkpoint, base model and device; examples/lambda its model and backend.
        on = f" on {h['base_model']}" if "base_model" in h else ""
        return (
            f"{NAME}: {h.get('checkpoint', h.get('model'))}{on} "
            f"[{h.get('device', h.get('backend'))}, window {h['max_length']}] at {self.url}"
        )

    def __str__(self) -> str:
        return f"{self.calls} decisions, {self.input_tokens} input tokens"
