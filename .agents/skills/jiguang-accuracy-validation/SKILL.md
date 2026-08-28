---
name: jiguang-accuracy-validation
description: Submit, monitor, archive, and link a standardized dataset accuracy evaluation on Jiguang for a clean committed and pushed vLLM Ascend code state. Use only when the user explicitly asks to use Jiguang or 极光 for accuracy, correctness, or dataset evaluation. Do not trigger for generic correctness validation, local AISBench, debugging, dirty branches, or administrator operations.
---

# Jiguang Accuracy Validation

Require explicit Jiguang opt-in for this task. Keep the normal local correctness toolchain available and unchanged.

1. Use `jiguang-runtime-management` to pass the Git gate, inspect real NPU occupancy, acquire a lease, and prepare or reuse `vaws-jiguang`.
2. Query `jiguang.evaluation_catalog_list`; select only a supported app, dataset, version, split, and current-account container connection.
3. Build a canonical payload with `python3 scripts/accuracy_request.py ...`, including every selected `--device-id` and a non-empty configuration.
4. Call `jiguang.evaluation_plan` and verify commit, submodule SHAs, model, dataset, configuration, deployment, and devices.
5. Call `jiguang.evaluation_submit` with `confirm=true` only after the plan is complete.
6. Poll `jiguang.evaluation_get`; use `jiguang.evaluation_artifacts` after terminal status. Archive passed, failed, inconclusive, and cancelled Runs.
7. Add the archive URL and canonical summary digest to the local correctness Run Manifest using the runtime Skill's `jiguang_manifest_link.py`.
8. Stop the service and release the host lease on every terminal path.

Do not diagnose inside the Jiguang container. Return a failed Run to the local correctness, graph-debug, distributed-debug, or operator-debug workflow with its archive link and exact reproduction configuration.

Read [references/behavior.md](references/behavior.md) for evidence requirements and [references/acceptance.md](references/acceptance.md) before changing submission behavior.
