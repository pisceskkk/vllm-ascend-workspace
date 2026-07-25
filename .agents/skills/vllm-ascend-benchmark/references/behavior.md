# Benchmark Skill Behavior

## Relationship to remote-dev

Use `.remote-dev` tools for ad hoc remote read/edit/bash/search/patch around
benchmark setup and result inspection. This skill owns benchmark lifecycle and
keeps the existing scripts as the managed VAWS compatibility backend.

## Lifecycle

1. **Resolve target** from local inventory or a VAWS session spec.
2. **Assemble config** from user args + optional nightly reference.
3. **Stop existing service** on the target target if any. In session mode this means only the session service.
4. **Start service** via `serve_start.py` (which handles parity sync internally). If startup returns non-ready, call `serve_stop.py --force` for the same target before failing.
5. **Run benchmark iterations** via SSH on the remote container — all against the same warm service.
6. **Stop service** via `serve_stop.py`, passing through `--session-id` when used.
7. **Output structured JSON** on stdout.

## Configuration Priority

User-provided arguments always take priority:

```
user CLI args  >  agent-assembled context  >  nightly YAML fallback
```

Nightly YAML is a reference source for discovering how to configure a model/feature, not an execution template. When `--refer-nightly` is given:

- `server_cmd` and `server_cmd_extra` are merged (minus `--tensor-parallel-size` and `--port`, which are handled separately).
- `envs` are used as a base, with user `--extra-env` overriding.
- `benchmarks.perf` fields (`num_prompts`, `max_out_len`, `batch_size`) are mapped to bench CLI args.
- User-provided `--serve-args` / `--bench-args` completely override the nightly values.

## Multi-Run (Warm-Service) Mode

When `--runs N` is given with N > 1, the service starts once and all N iterations run sequentially against the same warm service instance. The service is never restarted between runs.

`--warmup-runs M` excludes the first M runs from the aggregated statistics. This accounts for JIT compilation, graph capture, and other one-time costs that skew initial measurements.

### Why warm-service matters

Restarting the service between runs means every run pays the full startup cost (model loading, graph capture, JIT). The "discard first run" strategy only works when subsequent runs hit the already-warm service. If the service restarts each time, there are no warm runs to keep.

### Aggregation

The output JSON includes:

- `per_run`: every run's metrics, tagged with `warmup: true/false`
- `aggregated`: mean + sample stddev over the non-warmup runs, for each metric key
- `aggregated.count`: number of runs included in the statistics

## Boundary with A/B experiments

This package measures exactly one code state. It does not switch worktrees,
alternate baseline and candidate order, enforce cross-state configuration
parity, choose regression thresholds, or produce a regression verdict.

Use the separate performance-regression workflow when the question requires a
baseline-versus-candidate conclusion. That workflow consumes this package's raw
single-run or aggregated JSON without changing Benchmark's measurement contract.

## Service readiness timeout

`--health-timeout` is forwarded unchanged to `serve_start.py`. Use it for large
models whose loading time exceeds the Serving default. It changes only how long
Benchmark waits for readiness; it does not change measured request metrics.

## Remote Execution

`vllm bench serve` runs inside the container via SSH. The result JSON file is written to `/tmp/` with a target token, local process id, and random suffix in the file name, then `cat`-ed back through the SSH session. The script parses the last JSON object from stdout. The unique file name matters because session containers can share the host `/tmp` mount on the same machine.

## Defaults

When the user provides no bench args and no nightly reference:
- `--num-prompts 64`
- `--max-concurrency 16`

These are conservative defaults suitable for a quick smoke test. For production benchmarking, users should specify explicit parameters.

## State Management

Benchmark results are not persisted locally by default. The structured JSON is returned on stdout for the agent or user to consume. The serving skill handles its own state under `.vaws-local/serving/` in legacy mode and `.vaws-local/sessions/<session-id>/serving.json` in session mode.
