# vLLM upstream sync command recipes

## Ensure refs exist

Fetch the intended refs explicitly if needed. This Skill never fetches on its
own.

## Plan

```bash
python -B .agents/skills/vllm-ascend-upstream-sync/scripts/upstream_sync.py plan \
  --output-dir .vaws-local/upstream-sync/vllm-upgrade-001 \
  --old-ref <current-ref> \
  --new-ref <target-ref> \
  --run-id upstream-sync-001
```

Review `report.md` and `sync-plan.json`.

## Apply exact target

```bash
python -B .agents/skills/vllm-ascend-upstream-sync/scripts/upstream_sync.py apply \
  --output-dir .vaws-local/upstream-sync/vllm-upgrade-001
```

Then inspect the parent gitlink diff and run Change Validation.
