# GLM-5.3-Flash stage 0 CPU receipt

Last updated: 2026-10-11

Status: **local CPU preflight passed; stage 0 is not fully accepted**. Linux
locked-environment full CPU regression and an immutable built image remain open.
No 320B weights were downloaded; no GPU, Vast, push or PR operation was performed.
AI assistance: implementation and local validation used OpenAI Codex.

## Frozen inputs and dependency changes

- Base verl: `1876b06d0a3e4e71e06230be10af14492ca8a75b` (v0.9.1).
- Branch: `feat/glm53-sft`, worktree: `../.worktree/verl-glm53-sft` from EliAgent.
- Transformers: `536ecc007387a50e77603bb5d92100e9b07514cc`, package version
  `5.19.0.dev0`. The installed distribution's direct_url.json confirms this SHA.
- Model metadata only: `zai-org/GLM-5.3-Flash-BF16` revision
  `a5b45eb41df6402735dedc900be14a42e8d5e538`.
- Target config.json SHA256:
  `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`.
- Every explicit Transformers extra and the unconditional uv override now uses
  the same full git pin. `requirements.txt` and `setup.py` use that pin too;
  retaining setup.py's old `<5.11` would break non-uv installations.
- kernels override is now `>=0.17.0,<0.18`, resolved to `0.17.2`, matching the
  fixed Transformers integration API. This is a necessary companion dependency
  change, not evidence that accelerated kernels work.
- `uv lock --default-index https://pypi.org/simple`: success, 350 packages.
  `uv lock --check --default-index https://pypi.org/simple`: success.
  The explicit index avoids the machine's Aliyun mirror and preserves existing
  PyPI source preferences. Offline validation initially lacked cached build
  metadata; online validation succeeded.
- Comparing package versions and sources against the old lock confirms torch,
  Megatron Core, vLLM, sglang, PEFT, TransformerEngine and TransferQueue unchanged.
  Megatron Core remains `0.18.0` at
  `ba7b5ebce12af60627a80985792a1449ce45f46c`; vLLM `0.24.0`, sglang `0.5.12`,
  PEFT `0.19.1`; torch `2.11.0+cpu` / `2.11.0+cu130`.
- Necessary lock updates include tokenizers `0.23.3`, huggingface-hub `1.33.0`,
  hf-xet and kernels' verification dependencies. No Qwen model code was changed.

## CPU environment and reproducible commands

Host: macOS arm64; uv `0.12.22`; CPython `3.12.15`; native torch `2.11.0`.
The project universal lock explicitly supports Linux x86_64/aarch64 only.
Therefore local execution used a separate uv-created environment, **not** a
successful `uv sync --extra cpu` of that lock. Its exact exploratory package
freeze is [glm53-stage0-macos-requirements.txt](glm53-stage0-macos-requirements.txt).
In particular local PEFT is `0.21.2`, whereas the unchanged Linux lock has
`0.19.1`; local results do not validate the locked PEFT training path.

```bash
# Run in the independent worktree. Keep install invocation outside a different
# checkout: uv pip can otherwise inherit that checkout's old override.
uv venv --python 3.12 /tmp/glm53-stage0/venv
(cd /tmp && uv pip install --python /tmp/glm53-stage0/venv/bin/python \
  -r /absolute/worktree/docs/receipts/glm53-stage0-macos-requirements.txt)
uv pip install --python /tmp/glm53-stage0/venv/bin/python -e . --no-deps
export PATH=/tmp/glm53-stage0/venv/bin:$PATH
USE_HUB_KERNELS=NO python -m pytest tests/utils/test_glm5_next_loading_on_cpu.py -q
python -m pytest tests/utils/test_model_on_cpu.py tests/utils/test_padding_on_cpu.py \
  tests/utils/test_import_utils_on_cpu.py tests/workers/config/ -q
python -m pytest tests/trainer/ppo/test_core_algos_on_cpu.py \
  tests/trainer/ppo/test_loss_aggregation_on_cpu.py \
  tests/trainer/ppo/test_value_loss_normalization_on_cpu.py \
  tests/utils/test_config_on_cpu.py tests/utils/test_temp_env_on_cpu.py \
  tests/utils/test_normalize_peft_param_name_on_cpu.py \
  tests/checkpoint_engine/test_fsdp_lora_only_checkpoint_on_cpu.py -q
python -m pytest tests/ --override-ini='python_files=*_on_cpu.py' --collect-only -q
```

The new tests use public auto factories only, no private auto mapping assertions.
AutoConfig reads only snapshot config metadata. The target config class is reused
with tiny dimensions before any model construction. Tiny has one KDA and one
full DSA layer, sparse routed/shared MoE in both, four mHC residual streams and a
small vision shell. Pure text uses input_ids/attention_mask, eager attention,
use_cache=False, random CPU weights; outputs have shape `(1, 4, 64)` and finite
values. The returned class is `Glm5NextForConditionalGeneration`, has trainable
parameters and generation support. Tiny save/reload reproduces logits with
`rtol=1e-5, atol=1e-6`. Qwen2 and Qwen3 random tiny models also save/reload logits.

## Results and baseline comparison

| Command group | Baseline 5.9.0 | Candidate 536ecc0 |
| --- | --- | --- |
| GLM metadata / tiny + Qwen2/Qwen3 | GLM unsupported, expected pre-upgrade failure | 4 passed |
| Model helpers, padding, imports, worker configs | 57 passed, 10 skipped, 1 failed | 57 passed, 10 skipped, same failure |
| PPO losses, config CLI, env utils, PEFT names, LoRA-only checkpoint | 92 passed | 92 passed |
| Full CPU collection | 1030 collected, 29 errors | 1034 collected, same 29 errors |

The unchanged worker-config failure references missing
`~/models/Qwen/Qwen2.5-0.5B` and raises HFValidationError before its intended
assertion. It was not patched or marked xfail. Full collection error paths match
exactly; dependencies missing include cachetools, uvicorn, Megatron and
TransferQueue. Full suite **execution** has not passed and is not replaced by
collection comparison. Optional engines are unavailable on this host.

Baseline comparisons ran the original checkout with Transformers `5.9.0`
(and kernels `0.12.3` for expanded tests/collection), then restored candidate
Transformers/kernels; other local packages and host were held constant. These
are controlled local comparisons, not full old/new locked Linux images.
The initial config CLI failure was due to subprocesses finding system Python;
setting PATH to the uv environment made both baseline and candidate pass.
An initial install accidentally inherited the main checkout's 5.9.0 override;
that expected GLM failure was corrected, not counted as candidate acceptance.

Changed-file `pre-commit run --files ...` passed all hooks (ruff, mypy,
generated trainer config verification, documentation/license/structure checks).
`git diff --check` also passed.

## Image candidate and remaining capabilities

`docker/Dockerfile.uv.cu130` already copies pyproject.toml, setup.py and uv.lock,
then runs `manage_envs.py prefetch cu130 dev -- --frozen`. No Dockerfile dependency
rewrite is needed: rebuilding consumes the updated universal lock. Existing
`verl:uv.cu130` is a mutable old tag, not an accepted GLM image. This machine has
no Docker runtime; **no OCI digest exists for this candidate**. The recipe SHA256
and lock SHA256 below identify build inputs only. Build on an authorized Linux
runner, record the resulting immutable OCI digest, then execute the frozen CPU
workflow before acceptance. No GPU launch is necessary for that CPU gate.

| Capability | Evidence / gap |
| --- | --- |
| KDA/DSA/mHC/MoE | Native fixed-source torch reference forward works on tiny CPU. Final explicit USE_HUB_KERNELS=NO run avoids accelerated hub dispatch. CUDA FLA chunk/recurrent KDA and convolution kernels, dtype behavior, backward and performance untested; CPU forward does not establish training support. |
| PEFT | No GLM adapter or backward test in stage 0. The engine's CAUSAL_LM wrapper, conditional-generation kwargs, exact attention targets, gradients and adapter restore remain stage 1/2 gates. Local PEFT differs from Linux lock. |
| CUDA / torch ABI | Linux lock retains torch 2.11/cu130 and native backend versions. macOS CPU torch cannot validate CUDA 13, TE, flash-attn, FLA, sglang-kernel, NVLink or FSDP2 ABI. Neither GPU backend was executed. |
| FP8 | Actual fixed BF16 snapshot config has no quantization_config. No FP8 files loaded, and no BF16 target weights loaded. FP8 reads, controlled BF16 dequantization, memory expansion and differentiability remain separate experiments; never delete quantization metadata to bypass validation. |
| Target 320B | Metadata class selection only. No full model initialization, weight loading, tokenizer download, pretrained-quality assertion or training acceptance. |

Local raw logs and diagnostic environments are in `/tmp/glm53-stage0/`;
verbose resolver logs may contain transient signed asset URLs and are deliberately
excluded from Git. No credentials or raw signed URLs are copied into this receipt.

## Content hashes

- `uv.lock` SHA256: `1e7576183f796ee8a3dba3b56d7f2986aa85e1bcf2cae6a8cee97595060b07f0`.
- `docker/Dockerfile.uv.cu130` SHA256: `1e0b55f748132d7631efc5c7517d61d036c735294edab3e180735cc407bd1d64`.
- `docs/receipts/glm53-stage0-macos-requirements.txt` SHA256: `23c1ee82eab6165eb4589f69bc3bac3ee4dfc38f109d191eefe26180ebba6766`.
