# ServeTune

[![CI](https://github.com/nac7/servetune/actions/workflows/ci.yml/badge.svg)](https://github.com/nac7/servetune/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Auto-tuning + fidelity-gated local LLM serving.**

You point ServeTune at a model and your GPU; it searches serving configurations
(quantization, speculative decoding, KV-cache dtype, batching), measures each on your
hardware, and returns the **fastest configuration that provably does not degrade output
quality** — gated by a statistical fidelity check against the full-precision reference.

> Most local-serving tools make models *easy* to run (Ollama) or give you a *library* of
> optimization techniques to apply by hand (NVIDIA Model-Optimizer). Neither tells you
> **which** combination is fastest on your box, nor whether a faster config silently
> changed the model's answers. ServeTune does both — and it orchestrates existing engines
> (vLLM) rather than reimplementing them.

## Why a fidelity gate?

A config that's 2× faster is worthless if it quietly produces different output. ServeTune
measures greedy-token agreement against the reference and requires the **lower bound of a
Wilson 95% confidence interval** to clear your threshold — so a config is accepted only when
we're statistically confident it's faithful, not merely when it looks faithful.

## Install

```bash
pip install -e .            # core (no GPU needed; mock backend)
pip install -e ".[vllm]"    # real serving on a CUDA GPU
```

## Quickstart (no GPU required)

The bundled mock backend runs the full pipeline so you can try it anywhere:

```bash
servetune optimize meta-llama/Llama-3.1-8B --params 8 --backend mock --report report.html
```

Example output:

```
config                        tok/s  speedup   agree  ci_low   gate
------------------------------------------------------------------------
int4-awq + spec + kv:fp8      134.4    3.36x    93.8%   91.9%   FAIL
int4-awq + spec + kv:fp16     134.4    3.36x    94.0%   92.1%   FAIL
fp8 + spec + kv:fp8            89.6    2.24x    98.6%   97.6%   PASS *
...
RECOMMENDED: fp8 + spec + kv:fp8  (2.24x faster, agreement 98.6%, gate >= 97%)
```

The headline: the most aggressive int4 config *looks* fastest but **fails the fidelity
gate** (silent quality loss); ServeTune instead recommends the fastest config that's
verifiably faithful.

## Run on a real GPU (vLLM)

On a CUDA machine (Linux; ~24 GB VRAM for a 7–8B fp16 reference):

```bash
pip install -e ".[vllm]"

# Cheap smoke run first (fp16 reference + a few fp8 configs):
servetune optimize Qwen/Qwen2.5-7B-Instruct --params 7 --backend vllm \
  --quants fp16,fp8 --max-configs 4 --report report.html

# Full sweep:
servetune optimize Qwen/Qwen2.5-7B-Instruct --params 7 --backend vllm --report report.html
```

Each config runs in its own subprocess so the GPU is fully reclaimed between candidates
(and a single OOM/bad config can't abort the sweep). Decoding is greedy (`temperature=0`)
so token agreement against the reference is meaningful. `int4-awq` configs require a
pre-quantized AWQ checkpoint as the model id. Speculative-decoding configs take a draft model
via `--draft-model`.

## How it works

```
optimize(model)
  ├─ profiler   → detect GPU / VRAM
  ├─ search     → enumerate configs, prune by VRAM
  ├─ backend    → run reference + each candidate (vLLM, or mock)
  ├─ fidelity   → greedy-agreement vs reference, Wilson-CI gate
  └─ pareto     → pick fastest fidelity-passing config + serve command
```

## Status

v0.1 — core pipeline + mock backend + **vLLM backend** + CLI + tests (22 passing).
The vLLM path runs each config in an isolated subprocess and is validated on a real GPU via
`SERVETUNE_GPU_TEST=1`. Next: real 7–8B benchmark numbers, PyPI release, then a llama.cpp
backend (v0.2).

## License

Apache-2.0.
