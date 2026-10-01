# Running the recipe on AWS

Scripts that bring up one multi-GPU EC2 host, set it up like the Linux environment in
[training/README.md, Setup](../README.md#setup), run `recipe.sh` on it stage by stage, and score the result
on JevBench. There is no SSH: every command goes to the host through SSM `send-command`,
and output comes back through S3. Measured stage timings are in
[Measured stage timings](#measured-stage-timings).

## What you need

- An AWS account, and credentials the AWS CLI v2 can use (set `AWS_PROFILE` if not the
  default profile). They must be able to create an S3 bucket, an IAM role and instance
  profile, a security group, EC2 instances, and send SSM commands.
- `bash`, `git` and `python3` on your machine (macOS or Linux). No Session Manager plugin.
- A default VPC in the region you launch in. The host gets a public IP for outbound
  traffic (Hugging Face, PyPI, GitHub); its security group has no inbound rules.
- EC2 quota for the shape. `p5.48xlarge` and `g6e.48xlarge` have 192 vCPUs, so check
  *Running On-Demand P instances* (or *G and VT*) in Service Quotas is at least 192.

Settings live in `training/aws/scripts/common.sh`; override them with environment variables or
put them in `training/aws/scripts/local.env` (git-ignored). The scripts that run on the host read
their settings from the command's environment. All settings:

| variable | read by | default |
| --- | --- | --- |
| `AWS_PROFILE` | the AWS CLI | the default profile |
| `HOBSON_HOME_REGION` | the local scripts | `us-west-2` (the bucket's region) |
| `HOBSON_BUCKET` | the local scripts | `hobson-v17-<account id>-<HOBSON_HOME_REGION>` |
| `HOBSON_ACCOUNT` | `common.sh` | the account of the credentials (used for the bucket name only) |
| `HOBSON_ROLE` | `ensure-infra.sh`, the launchers | `hobson-v17-host` (IAM role and instance profile) |
| `HOBSON_NAME_TAG`, `HOBSON_JOB_TAG` | the local scripts | `hobson-v17`, `v17-replication` (tags, and the scripts find hosts by `Name`) |
| `HOBSON_REGIONS` | the local scripts | `us-west-2 us-east-1 us-east-2` (where the scripts look for hosts) |
| `HOBSON_STATE_DIR` | the local scripts | `training/aws/scripts/.state` (the host file and the launch log) |
| `HOBSON_INSTANCE`, `HOBSON_REGION` | `ssm-run.sh`, `sync-code.sh` | the host in the state file, else the host with the `Name` tag |
| `AMI_PARAM` | the launchers | SSM parameter of the latest Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) |
| `ROOT_GB` | the launchers | `1024` (GiB of gp3 root volume) |
| `SHAPES` | `launch-host.sh` | `p5.48xlarge p5e.48xlarge p5en.48xlarge g6e.48xlarge` |
| `REGIONS` | `launch-host.sh` | `us-west-2 us-east-1`. Put each region in `HOBSON_REGIONS` too, or `teardown.sh` does not find the host |
| `SHUTDOWN_MIN` | `launch-host.sh` | `720` (minutes to the safety poweroff) |
| `CR_ID`, `REGION` | `launch-capacity-block.sh` | none: the reservation and its region are required |
| `SHUTDOWN_AT`, `POLL_S` | `launch-capacity-block.sh` | 30 min before the block ends, and `120` (seconds between state checks) |
| `PY`, `S3_PREFIX`, `RUN_DIR`, `NGPU`, `GIT_REV`, `TRAIN_CONFIG`, `PARENT_CONFIG`, `CKPT`, `SEED`, `FAST`, `ALLOW_COUNT_MISMATCH` | `run_recipe.sh`, on the host | see the header of `training/run_recipe.sh` and section 4 |
| `PY`, `GPU`, `PORT`, `JEVBENCH_DIR`, `JEVBENCH_COMMIT`, `EXPECT_MAX_LENGTH`, `MODEL_LABEL`, `HEALTH_TIMEOUT_S`, `HOBSON` | `jevbench.sh`, on the host | see the header of `evaluation/jevbench/jevbench.sh` |

## 1. Bucket and IAM role

```bash
training/aws/scripts/ensure-infra.sh
```

Idempotent. A private SSE-S3 bucket, and a role `hobson-v17-host` with SSM core and
read/write on that bucket only. `launch-host.sh` runs it too.

## 2. A host

**On-Demand**, if there is capacity:

```bash
training/aws/scripts/launch-host.sh
SHAPES="g6e.48xlarge" REGIONS="us-east-1" training/aws/scripts/launch-host.sh   # narrower search
```

It tries `p5.48xlarge`, `p5e.48xlarge`, `p5en.48xlarge`, `g6e.48xlarge` in every AZ of
us-west-2 then us-east-1, moves on at each capacity error, and prints the instance once
SSM is online. The host is tagged `Name=hobson-v17` and the other scripts find it.

**Capacity Block**, when On-Demand has no capacity. On-Demand can return
`InsufficientInstanceCapacity` for every 8-GPU shape in every AZ, while a block is on
offer in the same hour:

```bash
aws ec2 describe-capacity-block-offerings --region us-east-2 \
  --instance-type p5.48xlarge --instance-count 1 --capacity-duration-hours 24
aws ec2 purchase-capacity-block --region us-east-2 \
  --capacity-block-offering-id cbo-... --instance-platform Linux/UNIX    # prepaid, no refund
CR_ID=cr-... REGION=us-east-2 training/aws/scripts/launch-capacity-block.sh
```

`launch-capacity-block.sh` waits for the block to go `active`, then launches into it
(same tag `Name=hobson-v17` and state file as `launch-host.sh`).

Both launchers set a safety timer: the host powers off, which terminates it, after 12 h
(`SHUTDOWN_MIN=720`) or 30 min before the block ends (`SHUTDOWN_AT`). To move it:
`training/aws/scripts/ssm-run.sh 'hobson-extend 600'` (600 min from now). The root volume and
the NVMe scratch go with the host, so keep anything you need in S3.

## 3. Code and environment

```bash
training/aws/scripts/sync-code.sh . hobson --upload-only
training/aws/scripts/ssm-run.sh -t 1800 -f training/aws/image/setup-host.sh hobson
training/aws/scripts/ssm-run.sh -t 1800 -f training/aws/image/verify-host.sh hobson
```

`sync-code.sh` uploads one commit to the bucket with `git archive`: exactly the files that
the commit tracks, `data/synthetic/` included. Untracked files, ignored files and
uncommitted changes are not shipped, and the script warns when the worktree has
uncommitted changes. The default commit is `HEAD`. Use `--rev <commit>` to ship another
commit. The archive also holds `.hobson-rev`, the full SHA of the commit, which
`run_recipe.sh` and `jevbench.sh` record. To update `/opt/hobson/code/hobson` on the host,
commit your change and run the script again without `--upload-only`. On the host, `data/`
and `checkpoints/` are links to the NVMe scratch, and `data/synthetic/` is extracted through
the link.

`setup-host.sh` builds the reference environment of [training/README.md, Setup](../README.md#setup) in
`/opt/hobson/venv`: Python 3.12, torch 2.7.1 (cu126), transformers 5.17.0, peft 0.21.0,
flash-linear-attention, and `pip install -e ".[dev,train]"` (the `train` extra installs
`datasets`, which the corpus build needs). `causal_conv1d` is not installed, as in that
environment. It puts the NVMe instance store at `/opt/hobson/scratch` (Hugging Face cache,
`data/`, `checkpoints/`) and downloads both Qwen3.5 models. About a minute on a p5.

`verify-host.sh` prints the GPUs and versions, runs a forward and backward pass of
Qwen3.5-2B-Base on the fla kernels, fails if the Gated DeltaNet layers fell back to
PyTorch, and runs the CPU test suite. It ends with `VERIFY RESULT: PASS`.

## 4. The recipe

```bash
training/aws/scripts/ssm-run.sh 'hobson-bg v19 "cd /opt/hobson/code/hobson && PY=/opt/hobson/venv/bin/python FAST=1 S3_PREFIX=s3://$HOBSON_BUCKET/results/v19 training/run_recipe.sh all"'
training/aws/scripts/ssm-run.sh 'tail -n 5 /opt/hobson/logs/v19.log; cat /opt/hobson/code/hobson/runs/recipe/stages.jsonl'
```

`hobson-bg` runs the command as a systemd unit (`hobson-v19`), so it survives the SSM
call returning; the log is `/opt/hobson/logs/v19.log` and the exit code lands in
`v19.rc`. Do not hold an SSM command open for a long job.

Each stage is `bash training/recipe.sh STAGE` with `NGPU` set (default: every GPU), so the runner
runs the same commands as the recipe: `teacher` and `replay` one shard per GPU and then a
merge, `parent` and `train` under `torchrun`. The runner adds:

- the CPU stages side by side: `build`; `fetch`, then `multistep` and `adequacy`; `generated`;
- a row-count check on every data file (the counts in `recipe.sh` and `training/steps.md`), and a
  `sha256` of each in `runs/recipe/sha256.txt`; a stage fails on a mismatch;
- one line per stage in `runs/recipe/stages.jsonl` (`{stage, wall_s, gpus_used,
  host_shape, exit_code, git_rev, parent_config, train_config, ckpt, ...}`), its log in
  `runs/recipe/logs/`, a copy of the two training configs as run in `runs/recipe/configs/`,
  and a copy of its outputs to `S3_PREFIX`.

Run one stage by name (`training/run_recipe.sh teacher`) or a group (`cpu`, `gpu`, `all`).
The stage names are those of `recipe.sh`: `build`, `fetch`, `multistep`, `generated`,
`adequacy`, `catchall`, `teacher`, `distill`, `parent`, `replay`, `train`, `calibrate` and
`eval`. No group includes `catchall` or `distill`, so they run only when you name them.
The runner checks every name before it runs a stage. To resume on a new host, copy
`S3_PREFIX/data` and `S3_PREFIX/checkpoints` back and run the stages that are left. The
host's code has no `.git`: the runner records the commit in `.hobson-rev` as `git_rev`.
Set `GIT_REV` to record another value.

The runner reads `TRAIN_CONFIG`, `PARENT_CONFIG` and `CKPT` as `recipe.sh` does. It makes
sure that each config file exists and that `CKPT` is a path under `checkpoints/`. With none
of them set, it runs v19: `configs/train.yaml` into `checkpoints/hobson-2b-recipe`. A run
of another recipe names its config, its checkpoint and every stage. For example, this is
the v20 retrain from the committed labels (diagnostic evidence, not the default), as the
command inside `hobson-bg`:

```bash
cd /opt/hobson/code/hobson && PY=/opt/hobson/venv/bin/python FAST=1 \
  S3_PREFIX=s3://$HOBSON_BUCKET/results/v20 TRAIN_CONFIG=configs/experiments/v20.yaml \
  CKPT=checkpoints/hobson-2b-v20-retrain training/run_recipe.sh cpu catchall distill train calibrate eval
```

`eval` also runs the v20 catch-all and paraphrase-pair evaluations when their input files
exist (`recipe.sh`, function `evaluate`).

`SEED=k` and `FAST=1` train from copies of the two configs in `runs/recipe/configs/`,
passed to `recipe.sh` as `PARENT_CONFIG` and `TRAIN_CONFIG`, so the two config files need
different names. `SEED=k` sets `seed: k`. `FAST=1` sets the speed settings from
[training/README.md, Training on several GPUs](../README.md#training-on-several-gpus); it is measured for 8x 80 GB only, and the
runner refuses any other `NGPU`. A `RUN_DIR` holds one run: the copies are never
overwritten, so a stage whose config differs from an earlier stage's copy (a run without
`FAST` after one with it, for example) stops and asks for another `RUN_DIR`. The
`stages.jsonl` fields `parent_config` and `train_config` hold the names as given.

## 5. JevBench

```bash
training/aws/scripts/ssm-run.sh 'hobson-bg jev "cd /opt/hobson/code/hobson && PY=/opt/hobson/venv/bin/python GPU=0 evaluation/jevbench/jevbench.sh checkpoints/hobson-2b-recipe /opt/hobson/scratch/jev/v19 v19"'
training/aws/scripts/ssm-run.sh 'cat /opt/hobson/scratch/jev/v19/paired.txt; aws s3 sync /opt/hobson/scratch/jev/v19 s3://$HOBSON_BUCKET/results/jev/v19'
```

The setup of [evaluation/jevbench.md, Reproducing, and two caveats](../../evaluation/jevbench.md#reproducing-and-two-caveats), with JevBench at a
pinned commit: unmodified, the 231 public tasks, the `typesafe` adapter against
`strands-decider serve` on one GPU. The script refuses a port that already answers and a `/health`
that names another checkpoint or window (the stale server that `evaluation/jevbench.md` warns
about). With a third argument it compares per task against that run in
`research/data/jevbench_results.csv` (for example `v19`) with an exact McNemar test, using
`evaluation/jevbench/paired.py`.

## 6. Teardown

```bash
training/aws/scripts/teardown.sh                                   # lists what it would terminate
training/aws/scripts/teardown.sh --now                             # Name=hobson-v17 hosts
```

It terminates and waits until every host is terminated. The bucket, role and security groups
stay; they cost nothing idle except the bucket's storage. See
[Costs and cleanup](#costs-and-cleanup).

## Measured stage timings

Each stage is `recipe.sh STAGE` with `NGPU=8`; the runner times it and checks its row counts.

**v17**, on a `p5.48xlarge` (8x H100 80GB), seed 0, effective batch 32, against the RTX
3090 column below. These runs used the v17 recipe, when v17 was the default, and an
earlier runner that ran the same stages:

| stage | RTX 3090 | 8x H100 | 8x H100, `FAST=1` |
| --- | --- | --- | --- |
| build, fetch, multistep, generated | ~2 min for build | 1.6 min, all four | 1.4 min |
| teacher | 55 min | 179 s | 154 s |
| parent | ~5 h | 46 min | 25 min |
| replay | ~40 min | 131 s | 136 s |
| train | ~5 h | 47 min | 26 min |
| calibrate, then the three evals side by side | | 4.4 min | 4.4 min |
| **all** | **about 10 h** | **1 h 45 min** | **1 h 01 min** |
| GPU-hours | ~11.7 (3090) | 14.0 (H100) | 8.2 (H100) |
| JevBench public | 164/231 | 162/231 | 165/231 |

Against v17 per task, the 162 run differs on 10 tasks (McNemar p = 0.75) and the 165 run
on 15 (p = 1.0). The labelling stages scale with the GPUs: eight independent shards.
Training scales less, because a step is only 32 rows cut into forwards of at most 8 and
every rank waits for the slowest. On a `p4d.24xlarge` (8x A100 40GB) the recipe took
2 h 33 min without `FAST`.

**v19**, the default, on the same shape with `NGPU=8 FAST=1` (an earlier version of the
v19 recipe and runner that ran the same stages), seed 0, against the figures in [Usage](../README.md#usage) and [v19: answer adequacy](../../research/history.md#v19-answer-adequacy):

| stage | RTX 3090 | 8x H100, `FAST=1` |
| --- | --- | --- |
| build, fetch, multistep, generated, adequacy | | 2.5 min, side by side |
| teacher | | 179 s |
| parent | ~5 h | 26 min |
| replay | | 133 s |
| train | ~6 h | 28 min |
| calibrate, then the six evals one after another | | 7.7 min |
| **all** | **about 11 h** | **1 h 10 min** |
| GPU-hours | ~11 (3090) | 9.3 (H100) |
| JevBench public, at 3072 / 4096 | 167 / 168 | 167 / 167 |

## Costs and cleanup

AWS bills the resources that these scripts make. Check the prices for your regions on the
AWS pricing pages before you launch a host.

- **The instance**, from launch to termination. This includes setup, verification,
  JevBench and idle time, not only the recipe. The safety timer terminates the host after
  12 h by default. See [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/).
- **A Capacity Block**. You pay for the whole block when you buy it, and you get no refund
  for time that you do not use. See
  [Capacity Blocks pricing](https://aws.amazon.com/ec2/capacityblocks/pricing/).
- **The root volume**, while the host exists: 1024 GiB of gp3 at 16,000 IOPS and
  1,000 MiB/s. These are above the gp3 baseline, so AWS also bills the extra IOPS and
  throughput. See [EBS pricing](https://aws.amazon.com/ebs/pricing/).
- **The public IPv4 address** of the host, while the host exists. See
  [VPC pricing](https://aws.amazon.com/vpc/pricing/).
- **S3 storage** for code archives, data, checkpoints, run records and SSM output. It
  stays after teardown until you delete it. See [S3 pricing](https://aws.amazon.com/s3/pricing/).
- **Data transfer between regions**, when the host is not in `HOBSON_HOME_REGION`. An
  example is a Capacity Block in us-east-2 with the bucket in us-west-2. See the data
  transfer part of [EC2 On-Demand pricing](https://aws.amazon.com/ec2/pricing/on-demand/).

`teardown.sh` deletes only the hosts. To delete the other resources, first copy what you
need out of the bucket. Then, after teardown is complete, run these commands. Replace
`ACCOUNT_ID`. If you changed `HOBSON_BUCKET`, `HOBSON_HOME_REGION` or `HOBSON_ROLE`, use
your values.

```bash
B=hobson-v17-ACCOUNT_ID-us-west-2   # HOBSON_BUCKET
H=us-west-2                         # HOBSON_HOME_REGION
R=hobson-v17-host                   # HOBSON_ROLE
aws s3 rb "s3://$B" --force --region "$H"     # deletes every object, then the bucket
aws iam remove-role-from-instance-profile --instance-profile-name "$R" --role-name "$R"
aws iam delete-instance-profile --instance-profile-name "$R"
aws iam delete-role-policy --role-name "$R" --policy-name hobson-bucket-rw
aws iam detach-role-policy --role-name "$R" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam delete-role --role-name "$R"
for r in us-west-2 us-east-1 us-east-2; do    # every region that you launched a host in
  sg=$(aws ec2 describe-security-groups --region "$r" --filters Name=group-name,Values=hobson-v17-host \
    --query 'SecurityGroups[0].GroupId' --output text)   # this name is fixed in common.sh
  [ "$sg" = None ] || aws ec2 delete-security-group --region "$r" --group-id "$sg"
done
```

The local directory `training/aws/scripts/.state/` keeps the host file and the launch log. You can
delete it too.

## What this does not do

- Held-out isolation is by convention: data, checkpoints and eval sets share one host,
  one role and one bucket. Nothing in IAM stops a training job from reading eval files.
- No CDK or CloudFormation. Plain AWS CLI, one host at a time.
- The Hugging Face datasets the build reads are not pinned (as in `recipe.sh`); the
  runner checks row counts and records `sha256` of every data file instead.
