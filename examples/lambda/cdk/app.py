"""CDK app for the strands decider Lambda recipe.

    cd examples/lambda/cdk
    npx cdk deploy                                 # SnapStart on, build/model, stack strands-decider-lambda
    npx cdk deploy -c snapstart=off -c stack=my-decider

Deploys what examples/lambda/build.sh writes: build/function.zip as the function's code and
the Q8_0 GGUF beside it as an S3 asset. CDK keys both by a hash of their contents, so a new
model or a new zip always publishes a new version, and with it a new SnapStart snapshot.
The function is arm64 with 10 GB, the most Lambda gives and the only size with 6 vCPUs, and
it is reached through an IAM-authenticated Function URL on its `live` alias.

Context (`-c key=value`):
  stack               stack name (default strands-decider-lambda)
  model_dir           directory holding decider-q8_0.gguf (default ../build/model)
  zip                 the function zip (default ../build/function.zip)
  snapstart           on or off (default on). On: a new environment restores in seconds, and
                      the snapshot cache is billed for as long as the version exists, about
                      $39.50 a month at 10 GB. Off: nothing is billed while idle, and a new
                      environment loads the model before its first answer.
  strict_window       1 refuses a prompt over the window with a 422 (default 0: trim the state)
  max_request_tokens  most tokens one request may forward (default 24576; 0: no limit)
"""

from __future__ import annotations

import os

import aws_cdk as cdk
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct

RECIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DeciderStack(cdk.Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        model_dir: str,
        zip_path: str,
        snapstart: bool,
        strict_window: bool,
        max_request_tokens: int,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        model = s3_assets.Asset(self, "Model", path=os.path.join(model_dir, "decider-q8_0.gguf"))
        log_group = logs.LogGroup(
            self,
            "Logs",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        fn = lambda_.Function(
            self,
            "Decider",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.handler",
            code=lambda_.Code.from_asset(zip_path),
            memory_size=10240,
            timeout=cdk.Duration.minutes(15),
            snap_start=lambda_.SnapStartConf.ON_PUBLISHED_VERSIONS if snapstart else None,
            log_group=log_group,
            environment={
                "MODEL_BUCKET": model.s3_bucket_name,
                "MODEL_KEY": model.s3_object_key,
                "STRICT_WINDOW": "1" if strict_window else "0",
                "MAX_REQUEST_TOKENS": str(max_request_tokens),
            },
            # A new version replaces the last one, which CloudFormation then deletes: Lambda
            # bills each SnapStart version's snapshot cache for as long as it exists.
            current_version_options=lambda_.VersionOptions(
                removal_policy=cdk.RemovalPolicy.DESTROY
            ),
        )
        # current_version publishes the version SnapStart snapshots.
        cdk.Annotations.of(fn).acknowledge_warning(
            "@aws-cdk/aws-lambda:snapStartRequirePublish", "published through current_version"
        )
        # The model object only: Asset.grant_read would cover every object in the bucket.
        fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[model.bucket.arn_for_objects(model.s3_object_key)],
            )
        )
        alias = lambda_.Alias(self, "Live", alias_name="live", version=fn.current_version)
        url = alias.add_function_url(auth_type=lambda_.FunctionUrlAuthType.AWS_IAM)
        cdk.CfnOutput(self, "FunctionUrl", value=url.url)
        cdk.CfnOutput(self, "FunctionArn", value=alias.function_arn)


# Upload the GGUF from where build.sh wrote it, rather than copy gigabytes into cdk.out.
app = cdk.App(context={"aws:cdk:disable-asset-staging": True})


def _context(key: str, default: str) -> str:
    value = app.node.try_get_context(key)
    return default if value is None else str(value)


snapstart = _context("snapstart", "on")
if snapstart not in ("on", "off"):
    raise SystemExit(f"snapstart must be on or off, not {snapstart!r}")
DeciderStack(
    app,
    _context("stack", "strands-decider-lambda"),
    model_dir=os.path.abspath(_context("model_dir", os.path.join(RECIPE, "build", "model"))),
    zip_path=os.path.abspath(_context("zip", os.path.join(RECIPE, "build", "function.zip"))),
    snapstart=snapstart == "on",
    strict_window=_context("strict_window", "0") == "1",
    max_request_tokens=int(_context("max_request_tokens", "24576")),
)
app.synth()
