# Accuracy dataset templates

The machine-readable catalog is `accuracy-datasets.json`. Select a template
with `aisbench_run.py --template <name>`. Accuracy templates default to the
`reasoning` output-length profile. Select `standard` or `long-reasoning` with
`--generation-profile`, or set an exact cap with `--max-out-len`.

| Template | Hugging Face source | standard | reasoning (default) | long-reasoning | concurrency |
|---|---|---:|---:|---:|---:|
| `gsm8k-cot` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main/test` | 4096 | 16384 | 32768 | 64 |
| `gpqa-diamond-cot` | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa), Diamond (gated) | 8192 | 65536 | 131072 | 64 |
| `math500-cot` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), `test` (500 rows) | 8192 | 32768 | 65536 | 64 |
| `mmlu-pro-cot` | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), `test` | 4096 | 32768 | 65536 | 128 |

All lengths are token caps, not expected output lengths. Generation still stops
at EOS. The service context must be large enough for prompt tokens plus the
selected output cap; use `standard` when the deployed model or service has a
shorter context window.

## Why these caps

- Upstream vLLM's `tests/evals/gsm8k/gsm8k_eval.py` deliberately keeps a
  concise-model default of 256 tokens, while its model configs use 1024 for
  Qwen3-30B Thinking and 12000 for Qwen3.5-397B. This is why GSM8K has separate
  profiles instead of one universal cap; 16384 is rounded headroom over the
  current long-thinking test case.
- The official [DeepSeek-R1 evaluation](https://github.com/deepseek-ai/DeepSeek-R1)
  uses a 32768-token maximum generation length for its reported benchmarks,
  including MMLU-Pro, GPQA-Diamond, and MATH-500.
- The official [Qwen3 model guidance](https://huggingface.co/Qwen/Qwen3-4B)
  recommends 32768 output tokens for most queries and 38912 for complex math or
  programming tasks.
- Current vllm-ascend accuracy configurations exercise GSM8K at 10240 or 32768
  tokens and GPQA at 32768, 65536, or 131072 tokens depending on model family.
  The three profiles preserve a concise baseline while covering those deployed
  reasoning-model regimes.

The scoring target does not by itself imply a long answer: GSM8K and MATH-500
extract a final numeric or symbolic answer, while GPQA-Diamond and MMLU-Pro end
in a multiple-choice selection. The extra budget is for visible reasoning
tokens produced before that short scored answer. Choose the profile from the
model's generation behavior and published reproduction protocol, not only from
the metric name.

Hugging Face supplies dataset identity, task structure, fields, and splits. It
does not generally prescribe vLLM/AISBench `max_out_len`, temperature, top-p,
or request concurrency. Those values are repository operational defaults:

- temperature 0 and top-p 1 make pass@1 accuracy runs deterministic;
- output profiles distinguish concise instruct models from reasoning models;
- concurrency is deliberately non-trivial so service scheduling or DP defects
  are not hidden by serial evaluation;
- published model reproductions may require different sampling. In that case,
  capture those parameters as a separate template instead of silently changing
  a shared baseline.

The bundled AISBench dataset config remains the execution contract. The Hugging
Face source is provenance and a download reference; GPQA additionally requires
accepting its access conditions.

Examples:

```bash
# Default reasoning cap for GSM8K: 16384
aisbench_run.py ... --template gsm8k-cot

# Concise instruct model on GPQA: 8192
aisbench_run.py ... --template gpqa-diamond-cot --generation-profile standard

# Exact published/model-specific reproduction: explicit value wins
aisbench_run.py ... --template math500-cot \
  --generation-profile reasoning --max-out-len 38912
```
