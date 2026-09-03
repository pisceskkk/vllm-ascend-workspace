# Runtime behavior

The workspace owns physical-machine probing, NPU coordination, Docker creation, container replacement, rollback, and garbage collection. Jiguang owns only account-scoped inventory records, an attachment to an existing container, evaluations, and archives.

Use these runtime decisions:

| Condition | Decision |
|---|---|
| No recorded runtime | Create `vaws-jiguang-next` |
| Runtime hash changed | Replace |
| Native code hash changed | Replace |
| Runtime is unhealthy | Replace |
| User explicitly requests clean environment | Replace |
| Only Python, dataset, model, or case configuration changed | Reuse |

Compute `runtime_hash` from the immutable image digest plus Python, PyTorch, torch_npu, CANN, and toolchain facts. Compute `native_code_hash` from tracked native/build entries in both submodules. A branch name is metadata; exact commit SHAs are authoritative.

Keep local untracked state in `.vaws-local/jiguang/`. Keep host-shared advisory coordination in `/tmp/vaws-npu-coordinator/v1`. Missing coordination state begins a new epoch and never proves an NPU is free.

The workspace gate also proves source pairing. Unless the user explicitly
passes `--vllm-commit`, vLLM HEAD must equal the full commit in
`.github/vllm-main-verified.commit` at vllm-ascend HEAD. A missing/invalid pin
or mismatch blocks before any Jiguang runtime mutation.
