# Generators and their exports

Paths are relative to the repository root unless they are links. A module path such as
`data/recipes.py` is relative to `src/strands_decider/`, but a data file such as `data/train_v5.jsonl`,
and the scripts under `data/generators/` and `data/checks/`, are relative to the repository
root.

The generators in `data/generators/` call models through OpenRouter, or through Amazon Bedrock
(see [Backends](#backends)); one early pilot used Bedrock's native API. They need an API key
and cost money, so the recipes never run them. Each export of
documents, adequacy items or flips holds the documents or batches, every verifier answer
(`verify.jsonl`), the kept rows (`gen_train.jsonl`, `gen_eval.jsonl`) and a `stats.json`.
The paraphrase exports hold the candidates and the checked paraphrases. A run also writes
its API errors to `failures.jsonl`; those logs are not committed.
`python -m strands_decider.data.generated` turns an export into the training and evaluation files
in `data/synthetic/`. The commands are in the comments of `recipe.sh`.

| Export directory | Generator | Writer and verifier | Recipe input |
| --- | --- | --- | --- |
| `data/generators/gen_v16/` | `gen_documents_openrouter.py` (the default skill set) | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: `generated_v16*.jsonl` |
| `data/generators/gen_pilot_qwen/` | `gen_documents_openrouter.py` (the OpenRouter pilot) | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: part of `generated_v18*.jsonl` |
| `data/generators/gen_mixed_pilot/` | `gen_documents_openrouter.py --skill-set mixed` | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: part of `generated_v18*.jsonl` |
| `data/generators/gen_weak/` | `gen_documents_openrouter.py --skill-set weak` | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: part of `generated_v18*.jsonl` |
| `data/generators/gen_adequacy/` | `gen_adequacy_openrouter.py` | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: `adequacy_gen*.jsonl` |
| `data/generators/gen_flips/` | `gen_flips_openrouter.py` | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: `flips_v20*.jsonl` |
| `data/generators/gen_paraphrases/` | `gen_paraphrases_openrouter.py` | Qwen3.6-27B and Qwen3.5-397B-A17B | Yes: `paraphrases.jsonl`, which `recipe.sh generated` reads |

The exports of the pilots that no recipe reads were removed before publication: the Nova
Premier pilot of `gen_documents.py` (Amazon Bedrock), whose rows did not record the model,
and three pilots of `gen_paraphrases_openrouter.py`, whose writer and checker were not
recorded. `gen_documents.py` stays as the record of the Nova pilot's method.

## Backends

The four generators share one client, `data/generators/llm_client.py`. It sends the same OpenAI-style
chat completion to one of three backends, chosen with `--backend` or the environment
variable `HOBSON_LLM_BACKEND`. The default is `openrouter`, and a run with no new flag makes
the same requests as before.

| Backend | For | Endpoint | Auth | Cost in the run summary |
| --- | --- | --- | --- | --- |
| `openrouter` (default) | a solo developer: one key, many models | `openrouter.ai/api/v1/chat/completions` | `OPENROUTER_API_KEY` | reported by OpenRouter per call; `--max-cost` works as is |
| `bedrock` | production and enterprise use | the OpenAI-compatible Chat Completions API on `bedrock-runtime.{region}.amazonaws.com`, the endpoint AWS recommends | a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK`, or the default AWS credential chain (IAM roles, profiles) when `botocore` is installed | tokens times `--price-in` and `--price-out` (USD per million; default 0, so `--max-cost` needs them) |
| `bedrock-mantle` | the same, on Bedrock's `bedrock-mantle.{region}.api.aws` endpoint | as above, model ids without the `-v1:0` suffix | a Bedrock API key in `AWS_BEARER_TOKEN_BEDROCK` | as `bedrock` |

The region comes from `--region`, else `AWS_REGION`, else `AWS_DEFAULT_REGION`, else
`us-west-2`. No package is added for any backend: the client uses the standard library, and
`botocore` only if it is already installed.

**Models.** Bedrock does not offer the two models that produced the committed exports,
Qwen3.6-27B and Qwen3.5-397B-A17B. When `--writer`, `--verify-models` or `--checker` is not
given, the client uses the backend's defaults from the table `DEFAULT_MODELS` at the top of
`data/generators/llm_client.py`:

| Role | `openrouter` | `bedrock` | `bedrock-mantle` |
| --- | --- | --- | --- |
| writer | `qwen/qwen3.6-27b` | `qwen.qwen3-235b-a22b-2507-v1:0` | `qwen.qwen3-235b-a22b-2507` |
| verifier, checker | `qwen/qwen3.5-397b-a17b` | `qwen.qwen3-235b-a22b-2507-v1:0` | `qwen.qwen3-235b-a22b-2507` |

The Bedrock default is the nearest open-weight Qwen3 model it offers, in both roles (the
verifier judges in a fresh context, without the writer's answer). The defaults need a
region that offers it. As of 2026-09 (`aws bedrock list-foundation-models --by-provider
qwen`) these are us-west-2, us-east-2, eu-central-1, eu-north-1, ap-northeast-1, ap-south-1
and ap-southeast-2; us-east-1 offers Qwen3 32B only, although the 235B model card lists it.
They are different models. **A run on another backend or with another model gives new data with its own
labels, not the committed data**: `data/synthetic/`, `data/SHA256SUMS` and the exports in
`data/generators/gen_*/` come from the OpenRouter runs recorded in their rows. Each generator prints
its backend and models at startup, with a note when they are not the recorded ones.

**Max output tokens and reasoning.** The Qwen3 models on Bedrock cap output at 8K tokens.
The table `MODELS` in `data/generators/llm_client.py`, keyed by model id, gives each listed
model's default (8000 for the Qwen3 models on Bedrock and Mantle) and whether it takes
`reasoning_effort`. The explicit flags (`--writer-max-tokens`, `--verify-max-tokens`,
`--check-max-tokens`) always override the cap, and a model not in the table keeps the
generator's own default. The `reasoning` column marks the models that think: Qwen3 235B A22B
2507 and Qwen3 Next 80B A3B are Instruct releases that never answer a request carrying
`reasoning_effort`, so the client drops the field for them whatever `--reasoning-effort` says
(a model not in the table gets the flag as given). On bedrock-runtime Qwen3 32B returns its
thinking inline, in a `<reasoning>` block ahead of the answer, which the client strips from the
start of the content. Edit that table when a model's cap or its handling of `reasoning_effort`
changes.

**Provenance.** Every writer row records `backend`, `writer` and `provider` (the host
OpenRouter routed to, or `bedrock:{region}` / `bedrock-mantle:{region}`), and every verifier
row records `backend`, `model` and `provider`. The paraphrase rows record `backend`,
`writer`, `checker` and `provider`. Rows written before this change carry `writer`, `model`
and `provider` only; they are all OpenRouter rows.

Examples, each a small pilot:

```bash
# OpenRouter (today's path)
export OPENROUTER_API_KEY=...
python data/generators/gen_flips_openrouter.py --out gen_flips_pilot --batches 5 --max-cost 1

# Amazon Bedrock with an API key
export AWS_BEARER_TOKEN_BEDROCK=...
python data/generators/gen_flips_openrouter.py --backend bedrock --region us-west-2 \
    --out gen_flips_bedrock --batches 5 --price-in <USD per M input> --price-out <USD per M output>

# Amazon Bedrock with the AWS credential chain (botocore installed, a profile or a role)
python data/generators/gen_documents_openrouter.py --backend bedrock --region us-west-2 \
    --out gen_docs_bedrock --docs 10

# Bedrock Mantle
export AWS_BEARER_TOKEN_BEDROCK=...
python data/generators/gen_adequacy_openrouter.py --backend bedrock-mantle --region us-west-2 \
    --out gen_adequacy_mantle --batches 3
```

`tests/test_llm_client.py` checks the three request shapes offline, on a fake transport. The
workflow `.github/workflows/generators-live-test.yml` runs one batch per backend for real
on pull requests that touch the client or a generator; it needs an AWS role and a secrets
entry configured in the repository, see the workflow's header.
