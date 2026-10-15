# strands decider on AWS Lambda

**Serverless decider inference: no GPU, no instance to reserve, no server to keep running.** This
recipe serves the 2B Qwen3.5 decider from an AWS Lambda function on plain arm64 CPUs. Lambda
starts execution environments when requests arrive, runs as many side by side as the traffic
needs, and drops them when it stops. You pay per request, and the only things that persist are
the function and a 2 GB model file in S3.

Under the hood the torso runs through [llama.cpp](https://github.com/ggml-org/llama.cpp) on a
Q8_0 GGUF, and everything else is the package's own torch-free code: the prompts and readouts in
`prompting.py`, and the window fitting and the pointer head in `hidden_engine.py`, which
tests/test_hidden_engine.py holds to the same answers as the torch engine. Neither torch nor a
GPU is involved. The API is `strands-decider serve`'s `/v1/systemone` and `/health`, behind an
IAM-authenticated Function URL, and the function can be invoked directly as well.

A typical request (a state of about 500 tokens and three questions) takes 4.2 s on a Graviton3
host and 6.7 s on a Graviton2 host, about $0.0006 to $0.0009 at Lambda's arm64 price. Lambda
picks the CPU generation, not you, so expect a mix.

## When it fits

Seconds per request and nothing standing make Lambda a fit for decisions that can wait a few
seconds, at volumes where a GPU would sit idle:

- Asynchronous work: triaging tickets, emails or documents as they arrive, moderation and
  routing queues, batch scoring. A request asks every question about one state at once, and
  extra questions cost little, because the state is decoded once.
- Low or bursty traffic. A GPU instance costs the same at one request a day as at a million;
  this costs per request, and nothing while nobody calls (with SnapStart off).
- Checkpoints before a consequential agent action (sending, paying, deleting), where a few
  seconds are acceptable.

It is too slow for a guard on every tool call of an agent loop, as in
`examples/strands/tool_call_intervention.py`: serve the decider on a GPU for that
(`strands-decider serve`, about 100 ms a question).

## What it costs

At us-east-1 prices for a 10 GB arm64 function ($0.0000133334 per GB-second):

| | SnapStart on (default) | SnapStart off |
| --- | --- | --- |
| Each request | about $0.00013 per second of run time (table below) | the same |
| Each new execution environment | $0.0014 to restore the snapshot | about $0.004, the 30 s model load |
| Standing cost, nobody calling | **about $39.50 a month** for the snapshot cache, plus S3 | **S3 only, about $0.05 a month** |
| First request on a new environment | **about 7 s** | **about 36 s** |

SnapStart is what makes a new environment answer in seconds: Lambda snapshots an environment
with the model already loaded and restores new ones from it. For Python functions Lambda bills
that snapshot's cache, $0.0000015046 per GB-second of configured memory, for as long as the
version exists. That is about $1.30 a day at 10 GB, whether or not anyone calls the function.
The snapshots of Python functions do not expire. Every deploy that changes the function
publishes a new version with its own snapshot, so the stack deletes the version it replaces;
the `live` alias is all the Function URL serves. Destroy the stack when you are done with it,
and the cache charges stop.

With SnapStart off (`-c snapstart=off`) there is no standing cost beyond the model in S3, and
each new environment downloads and loads the model before its first answer, about 30 s.
Environments stay warm for a while between requests, so steady traffic pays that only when
Lambda scales out.

Requests are billed at Lambda's duration price for the whole 10 GB, whatever the request uses.
CloudWatch Logs adds $0.50 per GB ingested, and a request writes a few lines; the stack keeps
two weeks of them.

## Requirements

- The package installed with its torch dependencies (`pip install -e .`), for `prepare_model.py`
  and llama.cpp's GGUF converter. A CPU is enough.
- [finch](https://github.com/runfinch/finch) or Docker able to run `linux/arm64` containers
  (`CONTAINER=docker` for Docker). An Apple-silicon Mac runs them natively.
- The [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/home.html): Node.js for its CLI
  (`npx cdk`), and its Python library (`pip install -r examples/lambda/cdk/requirements.txt`).
  The account and region must be bootstrapped for CDK once (`npx cdk bootstrap`).
- AWS credentials that can deploy the stack, and boto3 for `client.py` (`pip install boto3`).

## Usage

```bash
examples/lambda/convert_model.sh         # checkpoint -> merged fp16 torso, head, f16 GGUF
examples/lambda/build.sh                 # f16 GGUF -> Q8_0, and build/function.zip
cd examples/lambda/cdk
npx cdk deploy                           # the model and the zip as assets, the function, its URL
cd -
python examples/lambda/client.py health
python examples/lambda/client.py ask
```

No weights live in the repo: `convert_model.sh` downloads the checkpoint and writes every
derived file under the git-ignored `build/`, about 10 GB on disk. Merging the checkpoint peaks at
about 10-12 GB of RAM, so stop other heavy processes, such as a container VM, while it runs.
`convert_model.sh` passes its arguments to `prepare_model.py`, which takes a checkpoint path or
Hub id; the default is `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`. Any Qwen3.5 torso
with a pointer head works. `local_test.sh` runs the built zip through its handler in an arm64
container, with the GGUF from `build/` instead of S3.

`cdk/app.py` is the stack. `npx cdk deploy` takes these as `-c key=value`:

| Context | Default | |
| --- | --- | --- |
| `stack` | `strands-decider-lambda` | the stack name, which `client.py --stack` reads |
| `snapstart` | `on` | `off` for no standing cost (see [What it costs](#what-it-costs)) |
| `strict_window` | `0` | `1` refuses a prompt over the window with a 422, as `serve --strict-window` does |
| `max_request_tokens` | `24576` | the most tokens one request may forward; `0` for no limit |
| `model_dir`, `zip` | `../build/model`, `../build/function.zip` | where `build.sh` wrote them |

CDK uploads the GGUF and the zip to its bootstrap bucket, named by a hash of their contents: a
new model or a new zip always deploys a new version, and an unchanged one is not uploaded
again. `npx cdk destroy` removes the stack; the uploaded files stay in the bootstrap bucket
until `npx cdk gc` (or you) delete them.

### Calling it

The Function URL takes `serve`'s requests and answers with its bodies and status codes, and
the response's `x-decider-cpu-part` header names the host's CPU (0xd0c Graviton2, 0xd40
Graviton3, 0xd4f Graviton4). It accepts only SigV4-signed requests from principals with
`lambda:InvokeFunctionUrl`. `client.py` signs with boto3's credentials, and so does the Strands
example's client (`examples/strands/_client.py`) when `STRANDS_DECIDER_URL` is a Function URL.

A direct invocation passes the request as the event and returns the response, which suits Step
Functions, EventBridge targets and other AWS callers; an invalid request fails the invocation:

```bash
aws lambda invoke --function-name <FunctionArn from the stack outputs> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"state": "Help! My payouts have failed for 3 days!",
              "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}' \
  answer.json
```

## How it works

| File | Role |
| --- | --- |
| `prepare_model.py` | Folds the LoRA into the base weights, saves the torso as a plain fp16 `Qwen3_5ForCausalLM`, and writes the head weights, temperatures and tokenizer to `build/model/assets/` |
| `convert_model.sh` | Fetches llama.cpp at a pinned commit, runs `prepare_model.py` and converts the merged torso to an f16 GGUF |
| `build.sh`, `container_build.sh` | Quantize the f16 GGUF to Q8_0, then build the libraries, the shim and the Python dependencies in `amazonlinux:2023` (arm64) and zip them |
| `shim/decider_hidden.c` | A small C API over libllama that returns the final hidden states for exact token ids. `dh_forward_shared` decodes the state once and copies its memory to each question |
| `src/strands_decider/hidden_engine.py` | The engine, in the package: window fitting as `serve` does it, the pointer head in numpy and the readouts, over any backend that returns hidden states |
| `app/engine.py` | Binds that engine's torso forward to the shim |
| `app/loader.py` | Streams the GGUF from S3 into a memfd, retrying each range from where it broke off. SnapStart limits `/tmp` to 512 MB |
| `app/handler.py` | The Lambda handler. With SnapStart the model load and a warm-up run at import, so the snapshot holds them; without it they run in a background thread, because Lambda gives an on-demand init only 10 s and re-runs an init that overruns inside the first invocation. Every invocation waits for the load |
| `cdk/app.py`, `client.py` | The CDK stack (arm64, 10 GB, SnapStart, IAM Function URL) and a SigV4 client |

The build enforces four things, each found by a failed deploy or a failed test:

1. No i8mm. Lambda runs arm64 functions on Graviton2, Graviton3 and Graviton4, and a SnapStart
   snapshot taken on one can restore on another. An i8mm (`smmla`) instruction kills the process
   on Graviton2, so llama.cpp is built for `armv8.2-a+fp16+dotprod`, and the build fails if
   `libggml-cpu.so` contains `smmla`.
2. `libgomp`, `libstdc++` and `libgcc_s` ship in the zip. The python3.12 runtime has no libgomp, and
   ggml's CPU backend uses OpenMP. The build fails if `ldd` cannot resolve a library from the bundle.
3. The micro-batch is capped at 512 tokens (`DH_UBATCH`). An uncapped micro-batch grows the compute
   buffers with the prompt, and a 4,096-token state ran out of 10 GB.
4. No token is an output unless its hidden state is read. llama.cpp keeps a row of vocabulary
   logits, about 1 MB here, for every output token, and asking for embeddings makes every token
   an output. So the shim decodes the state with embeddings off, and the questions at most 512
   tokens per decode. Before this, a 3,900-token state with 64 questions ran out of 10 GB, and
   the longest states took a third longer.

The state is decoded once per request, so a request costs about the state's tokens plus 65 to 90
per question. Asking several questions in one request is much cheaper than separate requests.
Questions go through in groups of 32 (`DH_GROUP`) against the one decoded state.

## Measurements

Measured on `strands-decider-2B-qwen3.5-v1-2610`: a 10 GB function, three questions per
request, the median of 16 warm requests per row.

| Tokens per request | Graviton2 | Graviton3 | Cost (Graviton2) |
| --- | --- | --- | --- |
| 196 | 2.3 s | 1.5 s | $0.0003 |
| 656 | 6.7 s | 4.2 s | $0.0009 |
| 2,134 | 22.4 s | 13.8 s | $0.0030 |
| 3,964 | 44.3 s | 27.3 s | $0.0059 |

Peak memory stayed at or below 6.1 GB, including a 3,900-token state with 64 questions and a
request with 300 questions. A new environment answers its first request in 7.1 s with SnapStart
(5.4 to 7.4 s) and 36 s without (32 to 37 s), each the median of 10, mostly on Graviton2 hosts.

Compared with the torch engine in fp32 on CPU, over 17 requests that cover every question type,
structured states, 255 options and a 3,800-token state, Q8_0 moved option probabilities by at
most 0.017 and the derived confidences by at most 0.026, and changed no choice. The f16 GGUF,
on six of those requests, stayed within 0.0095, so most of that is the quantization. Q4_0 moved
noul and choice scores by up to 0.05, so this recipe does not use it.

## Limits

- The state and the longest question must fit the checkpoint's window (`max_length`, 4,096 tokens
  for the v1 release). As with `serve`, a question may take at most three quarters of it, and the
  state is trimmed from the end to fit the rest; a question trimmed into its options is refused
  with a 422, before any work is done. With `strict_window=1` a prompt over the window is
  refused instead, with a 422 that says how many tokens to cut.
- A request may forward at most `max_request_tokens` tokens, the state once plus every
  question (default 24,576). That is about 5 minutes on a Graviton3 host and 8 on a Graviton2
  host, inside the function's 15-minute timeout. A larger request gets a 422 with its token
  count, before any work is done.
- Text only. Requests with `images` get a 422.
- One request at a time per execution environment. Lambda scales by adding environments.
