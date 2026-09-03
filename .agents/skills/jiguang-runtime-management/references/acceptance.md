# Acceptance

- Reject dirty, untracked, unpushed, or detached-from-upstream workspace states before runtime, deployment, or evaluation mutation.
- Reject an unproven or mismatched vLLM/vllm-ascend commit pair; accept a
  verified-pin override only through an explicit user-supplied vLLM commit.
- Allow exact, confirmed current-account device-record maintenance without treating Git state as platform inventory state.
- Prevent two cooperating tasks from acquiring the same physical NPU.
- Reject an NPU occupied by an external process even when no lease exists.
- Reuse the runtime for Python-only changes.
- Replace the runtime for image/runtime/native hash changes.
- Preserve the working generation when validation of `vaws-jiguang-next` fails.
- Never ask Jiguang to create, reserve, release, or delete a container or NPU.
- Never return a token, cookie, password, private key, or plaintext SSH credential.
- Release a lease on every terminal path, or mark it `orphaned_busy` when occupancy remains.
