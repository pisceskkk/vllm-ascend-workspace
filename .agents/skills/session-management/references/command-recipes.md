# Command Recipes

## Multi-session group

After creating each isolated member session:

```bash
python3 .agents/skills/session-management/scripts/session_group.py create \
  --group-id pd-a \
  --member ray-head=head-session \
  --member ray-worker=worker-session \
  --startup-order ray-head,ray-worker
```

Use the parameterized Ray helpers inside the corresponding member containers:

```bash
scripts/ray_head.sh --node-ip 10.0.0.1 --port 6379 --interface eth0
scripts/ray_worker.sh --head-address 10.0.0.1:6379 --node-ip 10.0.0.2 --interface eth0
```

Neither helper stops an existing cluster unless `--stop-existing` is explicit.

Create a session on a ready base machine:

```bash
python3 .agents/skills/session-management/scripts/session_create.py \
  --machine a2-01 \
  --session-id pr123 \
  --devices 0,1
```

Measure the raw base-image bootstrap path without the prepared image cache:

```bash
python3 .agents/skills/session-management/scripts/session_create.py \
  --machine a2-01 \
  --session-id timing-raw \
  --no-worktree \
  --devices 0 \
  --disable-prepared-image-cache
```

Normal session creation keeps the prepared image cache enabled. Set `VAWS_DOCKER_PULL_POLICY=always` only when the image tag must be refreshed before session creation.

Default creation uses SSH-only readiness verification so parallel agents do not all run NPU smoke during setup:

```bash
python3 .agents/skills/session-management/scripts/session_create.py \
  --machine a2-01 \
  --session-id pr123 \
  --devices 0 \
  --verification-mode ssh
```

Use full verification only when the create step itself must validate `torch` / `torch_npu`:

```bash
python3 .agents/skills/session-management/scripts/session_create.py \
  --machine a2-01 \
  --session-id timing-full \
  --no-worktree \
  --devices 0 \
  --verification-mode full
```

Use the session for parity and serving:

```bash
python3 .agents/skills/remote-code-parity/scripts/parity_sync.py --session-id pr123
python3 .agents/skills/vllm-ascend-serving/scripts/serve_start.py \
  --session-id pr123 \
  --model /data/models/Qwen \
  --tp 2
```

Run a session-scoped benchmark:

```bash
python3 .agents/skills/vllm-ascend-benchmark/scripts/bench_run.py \
  --session-id pr123 \
  --model /data/models/Qwen \
  --tp 2
```

Collect and analyze profiling data in the same session container:

```bash
python3 .agents/skills/ascend-profiling-collection/scripts/collect_torch_profile_case.py \
  --session-id pr123 \
  --model /data/models/Qwen \
  --served-model-name Qwen \
  --tp 2 \
  --tag pr123_smoke \
  --mode enforce_eager \
  --request-kind text \
  --benchmark-output-tokens 32

python3 .agents/skills/ascend-profiling-analysis/scripts/profile_analyze.py \
  --manifest .vaws-local/ascend-profiling-collection/runs/<timestamp>_pr123_smoke/manifest.json \
  --tag pr123_smoke
```

Collect memory data from a session-scoped service:

```bash
python3 .agents/skills/ascend-memory-profiling/scripts/mem_collect.py \
  --session-id pr123 \
  --attach \
  --tag pr123_memory
```

Remove a session and its resources:

```bash
python3 .agents/skills/session-management/scripts/session_remove.py \
  --session-id pr123 \
  --remove-container \
  --remove-worktree \
  --release-leases
```
