# ServeTune Roadmap

ServeTune's job is to find the **fastest serving config that provably doesn't degrade
output quality**. The roadmap below is ordered by impact on that promise. Contributions
welcome — see the issues.

## Where v0.1 is today
- Search over quantization × speculative decoding × KV-cache dtype × batching
- vLLM backend with per-config subprocess isolation; mock backend for GPU-free runs
- Fidelity gate: **greedy-token agreement** vs. the fp16 reference, gated on the
  **Wilson-score 95% CI lower bound**
- CLI, HTML/console reports, CI on Python 3.9/3.11/3.13

## The most important next step: richer fidelity metrics

Greedy-token agreement is a **conservative proxy** for quality — two different token
sequences can both be correct (paraphrases), so a config the gate flags *might* still be
fine. v0.1 is therefore a fast, conservative regression detector. The plan is to make the
gate pluggable so users can choose the right notion of "faithful" for their workload.

**Design — a `Metric` interface** (`score(candidate_outputs, reference_outputs) -> FidelityReport`),
selected via `--metric`:

| `--metric` | What it measures | Needs | Gate |
|---|---|---|---|
| `agreement` (default, today) | exact greedy-token match rate | nothing (zero-dep) | Wilson-CI lower bound ≥ threshold |
| `kl` | avg KL divergence between candidate & reference **top-k next-token logprobs** | vLLM `logprobs=k` | mean KL ≤ threshold (label-free, more principled) |
| `semantic` | embedding cosine similarity of decoded outputs | optional `sentence-transformers` | mean similarity ≥ threshold (captures paraphrase equivalence) |
| `task-accuracy` | accuracy delta on a labeled eval set | `--eval-set` (prompt, gold) pairs | accuracy drop within threshold, with CI |

Priority order: **`kl`** (label-free + principled, the best default upgrade) → **`task-accuracy`**
(ground-truth, for teams with an eval set) → **`semantic`** (optional dependency). `agreement`
stays the zero-dependency default.

Implementation notes:
- `kl`: request top-k logprobs per token from vLLM, align by position, average KL. Already
  supported by the backend's greedy decode — just needs logprobs plumbed through `_vllm_worker`.
- `task-accuracy`: deterministic scoring (exact-match / multiple-choice), no LLM-as-judge.
- `semantic`: compare full decoded texts; guarded behind an optional extra.

## Other planned work
- **v0.2 — llama.cpp backend** (Mac / CPU / GGUF), opening the largest local-LLM audience.
- **Smarter search** — replace brute force with a Bayesian/bandit search over configs.
- **AWQ/GPTQ auto-handling** — resolve or build int4 checkpoints automatically.
- **Auto draft-model selection** for speculative decoding (trained selector).
- **Multi-GPU** — tensor-parallel config search.
- **CI mode** — `servetune gate` to fail a build when a serving-config change regresses outputs.
- **Community results** — a shared, reproducible leaderboard of `(model, GPU, best config)`.

## Non-goals
- Reimplementing an inference engine. ServeTune orchestrates vLLM / llama.cpp; it does not
  replace them.
- LLM-as-judge quality scoring (non-reproducible); fidelity metrics stay deterministic.
