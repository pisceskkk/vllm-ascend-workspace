---
name: jiguang-performance-benchmark
description: Submit, monitor, archive, and link a standardized Jiguang performance benchmark for a clean committed and pushed vLLM Ascend code state, including warmup and repeated measurements. Use only when the user explicitly asks to use Jiguang or 极光 for benchmark or performance validation. Do not trigger for generic local benchmarks, performance debugging, profiling, dirty branches, or administrator operations.
---

# Jiguang Performance Benchmark

Require explicit Jiguang opt-in for this task. Keep `vllm-ascend-benchmark` and `vllm-ascend-performance-regression` as the normal local workflows.

1. Use `jiguang-runtime-management` to pass the Git gate, inspect real NPU occupancy, acquire a lease, and prepare or reuse `vaws-jiguang`.
2. Query `jiguang.evaluation_catalog_list`; select only a supported performance app and current-account container connection.
3. Fix model, hardware, topology, input/output distribution, concurrency, request count or duration, warmup count, repetition count, and metric definitions.
4. Build a canonical payload with `python3 scripts/performance_request.py ...`, including every selected `--device-id` and a non-empty workload configuration.
5. Call `jiguang.evaluation_plan`, inspect the normalized request, and submit with `confirm=true` only after it matches the intended case.
6. Poll the task and archive every measured repetition, not only aggregates. Keep warmup separate.
7. Link the archive and canonical summary digest into a local performance Run Manifest.
8. Stop the service and release the host lease on every terminal path.

Do not profile or root-cause performance inside Jiguang. Reproduce a suspected regression with the local benchmark, regression, and profiling Skills using the archived configuration.

Read [references/behavior.md](references/behavior.md) for evidence requirements and [references/acceptance.md](references/acceptance.md) before changing submission behavior.
