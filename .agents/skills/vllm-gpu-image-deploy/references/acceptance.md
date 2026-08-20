# Acceptance and failure handling

## Acceptance evidence

Record for each host:

- host and alias
- Docker-save source image ID, runtime image ID, rootfs SHA256, and image-build commit
- local and remote image-tar SHA256
- source HEAD, image-base commit, source-manifest SHA256, and deletion-manifest count
- stable and rollback container names/IDs
- shared-library count
- image-runtime files seeded because they were absent from local source
- visible GPU count and unique model names
- configured expected GPU count/model and whether each matched
- installed-package resolved path
- import-smoke result
- conditional vendored DeepGEMM loader and required-API result
- model-root mount source, destination, and read-only state
- repository-level `gpu_workspace_setup_command` and resulting untracked GPU workspace config

## Candidate failure

If load, source materialization, runtime seeding, deletion replay, checksum validation, GPU visibility/expectation, or import smoke fails:

1. Leave the stable container unchanged.
2. Query `.agents/knowledge/` with the concrete failure signature before retrying.
3. Inspect the candidate logs and image/source commit relationship.
4. Do not rebuild extensions or install alternate dependencies automatically. First confirm whether the missing item already exists in the exact image package and should have been seeded.
5. Keep or remove only the Skill-labelled candidate after the cause is known.

## Swap failure

If stable-to-rollback rename succeeds but candidate rename fails, rename the rollback container back to the stable name and start it. Report both the original failure and rollback result.

## Cleanup

Never prune Docker globally. Delete only explicitly approved Skill-labelled candidate, rollback, transfer, or versioned-workspace artifacts.

## Runtime-overlay invariant

The accepted filesystem layout is:

```text
/workspace/vllm/vllm/                         local source + missing image runtime files
/usr/local/lib/python*/dist-packages/vllm     symlink -> /workspace/vllm/vllm
/usr/local/lib/python*/dist-packages/vllm.image/ original immutable image package
```

Local source wins on path collisions. Image files fill only absent paths. Explicit source deletions win after both layers are composed.

Source extraction is staged and checksum-validated before it becomes the versioned workspace. A pre-existing incomplete workspace is timestamp-renamed rather than overwritten or deleted.
