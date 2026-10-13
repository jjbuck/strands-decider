# strands decider on AWS Lambda

A recipe for serving the 2B Qwen3.5 decider on AWS Lambda without a GPU and without torch. The
torso runs through [llama.cpp](https://github.com/ggml-org/llama.cpp) on a Q8_0 GGUF, the pointer
head runs in numpy, and the prompts and readouts come from the package's own `prompting.py` and
`schema.py`. The API is `strands-decider serve`'s `/v1/systemone` and `/health`, behind an
IAM-authenticated Function URL.

On a 10 GB arm64 function with three questions per request, a 514-token state takes 5.6 s on a
Graviton3 host and 9.0 s on a Graviton2 host, about $0.0008 to $0.0012 per request at Lambda's
arm64 price. Lambda picks the CPU generation, not you, so expect a mix. Scores stay within 0.004
of the torch engine in fp32.

## Requirements

- The package installed with its torch dependencies (`pip install -e .`), for `prepare_model.py`
  and llama.cpp's GGUF converter. A CPU is enough.
- [finch](https://github.com/runfinch/finch) or Docker able to run `linux/arm64` containers
  (`CONTAINER=docker` for Docker). An Apple-silicon Mac runs them natively.
- The AWS CLI, credentials that can deploy a CloudFormation stack with an IAM role, and an S3
  bucket in the deploy region.

## Usage

```bash
examples/lambda/convert_model.sh                 # checkpoint -> merged fp16 torso, head, f16 GGUF
examples/lambda/build.sh                         # f16 GGUF -> Q8_0, and build/function.zip
BUCKET=my-bucket examples/lambda/deploy.sh       # upload both, deploy the stack
python examples/lambda/client.py health
python examples/lambda/client.py ask
```

No weights live in the repo: `convert_model.sh` downloads the checkpoint and writes every
derived file under the git-ignored `build/`, about 10 GB on disk. Merging the checkpoint peaks at
about 10-12 GB of RAM, so stop other heavy processes, such as a container VM, while it runs.
`convert_model.sh` passes its arguments to `prepare_model.py`, which takes a checkpoint path or Hub id; the default is
`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`. Any Qwen3.5 torso with a pointer head works.
`deploy.sh` reads `STACK` (default `strands-decider-lambda`) and `PREFIX` for the S3 keys, and the
AWS CLI's usual `AWS_PROFILE` and `AWS_REGION`. It skips the 2 GB model upload when the object is
already in S3. To remove everything, delete the stack and the `PREFIX/` keys in the bucket.

The Function URL accepts only SigV4-signed requests from principals with
`lambda:InvokeFunctionUrl`, and `client.py` signs with boto3's credentials. Changing `AuthType` to
`NONE` in `template.yaml` makes the function public.

## How it works

| File | Role |
| --- | --- |
| `prepare_model.py` | Folds the LoRA into the base weights, saves the torso as a plain fp16 `Qwen3_5ForCausalLM`, and writes the head weights, temperatures and tokenizer to `build/model/assets/` |
| `convert_model.sh` | Fetches llama.cpp at a pinned commit, runs `prepare_model.py` and converts the merged torso to an f16 GGUF |
| `build.sh`, `container_build.sh` | Quantize the f16 GGUF to Q8_0, then build the libraries, the shim and the Python dependencies in `amazonlinux:2023` (arm64) and zip them |
| `shim/decider_hidden.c` | A small C API over libllama that returns the final hidden states for exact token ids. `dh_forward_shared` decodes the state once and copies its memory to each question |
| `app/engine.py` | Renders the prompts with `strands_decider.prompting`, tokenizes, calls the shim and runs the pointer head |
| `app/loader.py` | Streams the GGUF from S3 into a memfd. SnapStart limits `/tmp` to 512 MB |
| `app/main.py`, `app/run.sh` | The FastAPI app, started by the Lambda Web Adapter. The model load and a warm-up run at import, so SnapStart snapshots them |
| `template.yaml`, `deploy.sh`, `client.py` | The SAM template (arm64, 10 GB, SnapStart, IAM Function URL), the deploy and a SigV4 client |

The build enforces three things, each found by a failed deploy:

1. No i8mm. Lambda runs arm64 functions on Graviton2 and Graviton3, and a SnapStart snapshot taken
   on one can restore on the other. An i8mm (`smmla`) instruction kills the process on Graviton2, so
   llama.cpp is built for `armv8.2-a+fp16+dotprod`, and the build fails if `libggml-cpu.so`
   contains `smmla`.
2. `libgomp`, `libstdc++` and `libgcc_s` ship in the zip. The python3.12 runtime has no libgomp, and
   ggml's CPU backend uses OpenMP. The build fails if `ldd` cannot resolve a library from the bundle.
3. The micro-batch is capped at 512 tokens (`DH_UBATCH`). An uncapped micro-batch grows the compute
   buffers with the prompt, and a 4,096-token state ran out of 10 GB.

The state is decoded once per request, so a request costs about the state's tokens plus 65 to 90
per question. Asking several questions in one request is much cheaper than separate requests. If
tokenizing the state and a question separately gives different ids from tokenizing them together,
the engine falls back to one full forward per question and reports `"state_shared": false`.

## Measurements

Measured on `strands-decider-2B-hobson-v21`, which has the same architecture as the v1 release:
a 10 GB function, three questions per request, the median of two warm requests.

| State tokens | Graviton2 | Graviton3 | Peak memory |
| --- | --- | --- | --- |
| 24 | 2.7 s | 1.8 s | 4.7 GB |
| 514 | 9.0 s | 5.6 s | 5.5 GB |
| 2,059 | 30.5 s | 20.2 s | 7.0 GB |
| 3,901 | 57.5 s | | 9.0 GB |

The same model through torch on CPU in the same function took 17.8 s at 514 tokens on Graviton2.
Q8_0 moves the scores by at most 0.006 from the torch engine; Q4_0 moved noul and choice scores by
up to 0.05, so this recipe does not use it. A cold start without a snapshot takes about 28 s,
mostly the model download.

## Limits

- The state and the longest question must fit the checkpoint's window (`max_length`, 4,096 tokens
  for the v1 release), about 3,900 tokens of state with short questions. A longer prompt gets a 422
  that says how many tokens to cut, rather than being truncated.
- Text only. Requests with `images` get a 422.
- One request at a time per execution environment. Lambda scales by adding environments.
