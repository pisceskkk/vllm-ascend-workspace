---
name: jiguang-runtime-management
description: Prepare, inspect, replace, connect, and clean the dedicated vaws-jiguang runtime on physical machines owned by the current Jiguang account, including real NPU occupancy checks and cooperative host leases. Use only when the user explicitly asks to use Jiguang or 极光 for an evaluation. Do not use for ordinary local development, dirty-worktree execution, administrator operations, interactive debugging, or generic serving and benchmark requests.
---

# Jiguang Runtime Management

Treat Jiguang as an account-owned resource directory, existing-container connection layer, evaluation runner, and archive. Never treat it as an NPU scheduler or container creator.

Before first use, store a fresh platform token in Windows Credential Manager with `.jiguang/host/set_jiguang_credential.ps1`. Store each device password or private key under a separate target and pass only that target name to `jiguang.device_registration_plan`. Never reuse credentials pasted into chat.

When Jiguang requires root password authentication for an existing workspace-managed container, use `.jiguang/host/prepare_jiguang_device_password.ps1`. It generates a fresh per-device password on Windows, passes it to `scripts/jiguang_device_access.py` only through standard input, and stores it in Windows Credential Manager without returning it. Preserve the inventory-recorded high SSH port; workspace containers use host networking, so replacing it with a documentation example such as port 2222 would break the managed endpoint. Keep public-key authentication enabled for workspace recovery and verification.

## Enforce the boundary

- Require explicit Jiguang opt-in for the current task; do not carry opt-in to later tasks.
- Operate only on resources returned for the authenticated account.
- Reject administrator, cross-user, permission-assignment, plaintext-password, terminal, and arbitrary-script operations.
- Allow current-account inventory reads, plans, device registration, and device-record removal independently of Git state.
- Require a clean, committed, pushed workspace and clean `vllm` and `vllm-ascend` submodules before runtime replacement, container connection, deployment, or evaluation submission.
- Never sync uncommitted code to a Jiguang runtime and never debug inside it.

## Maintain account inventory

Resolve exact current-account resource IDs before a mutation. Device deletion removes only the Jiguang registration; it does not delete a physical server, Docker container, or NPU. Register an existing physical machine or container through a `device_registration_plan` that references a Windows Credential Manager target, then apply only after the endpoint, device type, and owner are verified. Inventory maintenance does not carry or deploy workspace code and therefore does not use the Git gate.

## Prepare a runtime

1. Run `python3 scripts/jiguang_runtime.py gate`. Stop before runtime, container-connection, deployment, or evaluation mutation when it reports `blocked`.
2. Read account-owned devices with `jiguang.own_devices_list` and map the chosen physical machine to the workspace inventory.
3. If password authentication is required, run `scripts/jiguang_device_access.py --machine <alias>` first to inspect the plan, then invoke `.jiguang/host/prepare_jiguang_device_password.ps1` from Windows for the confirmed mutation and credential storage.
4. Run the machine-management `npu_occupancy.py` read-only probe. Trust observed hardware and process occupancy over platform availability.
5. Acquire a host-local cooperative lease through `scripts/jiguang_lease.py`. Run the SSH-backed wrapper outside the filesystem sandbox from its first call.
6. Run `python3 scripts/jiguang_runtime.py plan` with an immutable image reference and recorded runtime components.
7. Run `python3 scripts/jiguang_container.py ensure ...` outside the filesystem sandbox. Omit `--confirm` to inspect the plan; add it only after the exact machine, image, devices, and decision are accepted.
8. Reuse `vaws-jiguang` for matching runtime/native hashes. Create `vaws-jiguang-next` through the existing machine-management bootstrap helpers for first use, runtime changes, native changes, unhealthy drift, or an explicit clean-environment request.
9. Validate `vaws-jiguang-next`, atomically promote it, and retain the stopped old generation as `vaws-jiguang-prev-<generation>`. Never replace the working generation after a failed validation. Use `jiguang_container.py rollback` for a confirmed rollback.
10. Register the already-created container as an account-owned device when needed, connect it with `jiguang.container_connection_plan`, then call `jiguang.container_connection_apply` only after the plan matches the intended resource.
11. Record the ready generation with `scripts/jiguang_runtime.py record`; `jiguang_container.py ensure` records successful promotions automatically.

## Isolate each Run

Create `/vllm-workspace/.jiguang-runtime/runs/<run-id>/` inside the persistent container. Stop the previous service, check out the exact pushed commit, start a new service, warm it up, and keep logs/config/results under that directory. Allow only one Jiguang Run per machine unless disjoint NPU leases are proven.

Heartbeat the lease while any service or evaluation uses the devices. On success, failure, cancellation, or timeout, stop the service and release only after repeated free occupancy probes. Quarantine `orphaned_busy`; never force-release observed work.

Read [references/platform-api.md](references/platform-api.md) before changing platform calls, [references/behavior.md](references/behavior.md) for the state model, and [references/acceptance.md](references/acceptance.md) before changing lifecycle code.
