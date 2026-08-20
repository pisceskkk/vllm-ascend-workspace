---
name: vllm-gpu-image-deploy
description: Deploy a local Docker-save vLLM CUDA image tar and a commit-aligned Python-source overlay to one or more already reachable NVIDIA GPU hosts. Preserve image-built runtime artifacts, keep local source authoritative, validate configurable GPU count and model expectations, replace only the named managed container, and retain a stopped rollback container. Use for requests such as "把本地 vLLM tar 部署到 GPU 服务器", "把镜像编译产物嵌入源码目录", "重建 H20/H100/H200 开发容器", or "同步同一 vLLM 镜像到多台 NVIDIA GPU 服务器". Do not use for Ascend NPU hosts, service benchmarking, model download, registry builds, or arbitrary Docker cleanup.
---

# vLLM GPU image deployment

Deploy immutable local image content to a small NVIDIA GPU fleet without pulling from a registry or rebuilding vLLM operators.

## Critical rules

- Require key-based host SSH before transfer. Never use `sshpass`, `scp`, `sftp`, `rsync`, or saved passwords.
- Run `ssh -G` for every host before the first network SSH attempt.
- Inspect the Docker-save tar and pin the config-derived `sha256:<image-id>`; never deploy a moving `latest` tag by itself.
- Verify free space on both the image staging filesystem and Docker root before transfer.
- Hash the local tar once and verify the streamed bytes on every host before renaming the remote `.part` file.
- Start from the exact image build commit and use the local `vllm/vllm` package as the authoritative managed source layer. Do not run `pip install`, build extensions, or copy local `.so` files.
- Seed every image-built runtime file missing from the local package into the source tree without overwriting local files. This includes `.so`, generated `_version.py`, vendored/JIT packages such as DeepGEMM, generated FlashMLA interfaces, headers, and other wheel-only assets.
- Apply the source archive's deletion manifest after seeding so a file intentionally deleted relative to the image commit is not resurrected.
- Rename the original site-package directory to `vllm.image`, then symlink site-packages `vllm` to `/workspace/vllm/vllm`.
- Create and validate a candidate container before stopping the stable container.
- Replace only the explicitly named managed container. Preserve the previous one as a stopped timestamped rollback container.
- Do not stop, remove, or restart unrelated containers or GPU workloads.
- Use host networking, host IPC, all GPUs, a read-write `/workspace` bind, and a read-only model-root bind.
- Keep transfer and deployment metadata free of credentials.

## Workflow

1. Inspect the image tar:

   ```bash
   python3 scripts/image_tar_inspect.py --image-tar /path/image.tar
   ```

   Confirm the expected architecture, creation time, source image ID, rootfs SHA256, `VLLM_BUILD_COMMIT`, CUDA version, and tag metadata. Docker 29 may normalize legacy empty Config fields on import and produce a different runtime image ID; accept that only when the rootfs layer digest and build commit still match.

2. Run read-only preflight on each host:

   - `ssh -G`
   - key-auth probe
   - `nvidia-smi`
   - Docker root and filesystem free space
   - named container and workspace state

3. Stream the image tar once to all hosts:

   ```bash
   python3 scripts/fleet_stream.py \
     --file /path/image.tar \
     --host 10.0.0.1 --host 10.0.0.2 \
     --remote-path /home/user/.vaws-images/image.tar
   ```

   The script emits progress on stderr and one JSON result on stdout.
   When the system OpenSSH config is intentionally unusable, pass an explicit trusted config such as `--ssh-config /dev/null` to both transfer and deployment commands; the required `ssh -G` check uses the same config.

4. Prepare a source tar containing the local `vllm/vllm` tree and no local shared libraries:

   ```bash
   python3 scripts/prepare_source.py \
     --vllm-repo /path/to/vllm \
     --image-build-commit <exact-VLLM_BUILD_COMMIT> \
     --output .vaws-local/vllm-gpu-image-deploy/source/vllm-python.tar.gz
   ```

   Stream it with the same helper, or reuse a previously hash-verified source tar.

5. Deploy and atomically replace the named container:

   ```bash
   python3 scripts/deploy_fleet.py \
     --host 10.0.0.1 --host 10.0.0.2 \
     --remote-image-tar /home/user/.vaws-images/image.tar \
     --image-tar-sha256 ... \
     --image-id sha256:... \
     --image-ref vllm/vllm-openai:latest \
     --rootfs-sha256 ... \
     --image-build-commit ... \
     --remote-source-tar /home/user/.vaws-transfer/vllm-python.tar.gz \
     --source-sha256 ... \
     --python-sha256 ... \
     --source-head ... \
     --container vaws-user-vllm-gpu \
     --workspace-root /home/user/vllm-gpu-workspaces \
     --model-root /home/weight \
     --expected-gpus 8 \
     --expected-gpu-model "NVIDIA H20" \
     --approved-replace
   ```

6. Accept only when every host reports:

   - stable container running from the imported tar, validated by source image ID plus rootfs SHA256 and build commit
   - source manifest and every local source-file checksum pass after image-runtime seeding
   - at least one image-built `.so` present and the count of seeded runtime files recorded
   - installed vLLM package resolves to `/workspace/vllm/vllm`
   - `import vllm` and `import vllm._C_stable_libtorch` pass
   - when the image contains vendored DeepGEMM, it resolves through a real source loader and its scale-transform plus MQA/paged-MQA APIs are callable
   - at least one NVIDIA GPU visible, with expected count and exact model enforced when configured
   - model root visible and mounted read-only

7. Initialize the host as a GPU-only test workspace with the exact
   `gpu_workspace_setup_command` returned for that host. The command points to
   the repository-level `.agents/scripts/gpu_workspace.py`; all later GPU
   operations use the sibling `gpu_*.py` tools instead of the NPU/dual-repo
   Skill scripts.

Read [references/acceptance.md](references/acceptance.md) when diagnosing a failed candidate or reviewing rollback state.

## Idempotency and recovery

- A rerun with the same source-manifest identity and validated stable container returns `already_ready`; image-only generated files do not perturb idempotency.
- A failed candidate never replaces the stable container.
- A successful swap retains the prior container as `<name>-rollback-<UTC timestamp>` in stopped state.
- Never delete rollback containers automatically. Remove them only after explicit user approval.
- A partial `.part` transfer is removed by the remote transfer trap; a verified final tar is reused.

## Outputs

Keep local run records under `.vaws-local/vllm-gpu-image-deploy/`. Report the Docker-save source image ID, per-host runtime image ID, rootfs/build identity, source head/hash, container ID, observed GPU count/models, runtime-seed file count, shared-library count, rollback container name, and validation status.

Also report the generated `gpu_workspace_setup_command`. Running it creates an
untracked `.vaws-local/gpu-workspaces/*.json` target file whose `tool_root` is
this repository's `.agents/scripts/` and whose entries all use the `gpu_`
prefix.
