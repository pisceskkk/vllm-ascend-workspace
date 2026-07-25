# Benchmark Command Recipes

## Single-run: minimal

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --machine 173.131.1.2 \
  --model /home/weights/Qwen3.5-0.8B \
  --tp 1
```

Session-scoped equivalent:

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --session-id pr123 \
  --model /home/weights/Qwen3.5-0.8B \
  --tp 1
```

## Single-run: full-featured (MTP + graph mode)

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --machine 173.131.1.2 \
  --model /home/weights/Qwen3-Next-80B-A3B-Instruct \
  --health-timeout 1800 \
  --tp 4 \
  --extra-env OMP_NUM_THREADS=10 \
  --extra-env HCCL_BUFFSIZE=1024 \
  --extra-env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --serve-args \
    --max-model-len 40960 \
    --trust-remote-code \
    --async-scheduling \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.8 \
    --max-num-seqs 64 \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
  --bench-args \
    --num-prompts 256 \
    --max-concurrency 64 \
    --output-len 1500
```

## Multi-run with warmup: statistical benchmarking

Start the service once, run 5 iterations, discard the first as warmup, aggregate the remaining 4:

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --machine 173.131.1.2 \
  --model /home/weights/Qwen3.5-35B-A3B \
  --tp 4 \
  --runs 5 --warmup-runs 1 \
  --serve-args \
    --max-model-len 4096 \
    --trust-remote-code \
    --async-scheduling \
    --compilation_config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4,8,12,16]}' \
    --speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --bench-args \
    --num-prompts 64 \
    --max-concurrency 16 \
    --output-len 1500
```

## Single-run: with nightly reference as fallback

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --machine 173.131.1.2 \
  --model /home/weights/Qwen3-Next-80B-A3B-Instruct \
  --refer-nightly Qwen3-Next-80B-A3B-Instruct-A2
```

For baseline-versus-candidate scheduling and decisions, use
`vllm-ascend-performance-regression`; do not reproduce that orchestration here.
