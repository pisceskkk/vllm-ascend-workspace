---
name: vllm-ascend-upstream-sync
description: Compare old and new vLLM refs, detect changed files and Python API signatures, locate affected vllm-ascend imports or symbol consumers, recommend validation evidence, and apply a guarded clean submodule checkout. Use only for intentional vLLM submodule upgrades or compatibility assessment. Do not use for initial repository setup, arbitrary workspace diffs, merge conflict resolution, or automatic upstream merges.
---

# vLLM Ascend Upstream Sync

Turn a requested vLLM submodule upgrade into a reviewable compatibility report
before moving the gitlink. Reuse Change Validation after the pointer changes;
this Skill owns only upstream-ref comparison and guarded checkout.

## Workflow

1. Ensure the old and new vLLM refs exist locally. Fetching remains an explicit
   repository operation; the script does not access the network.
2. Run `scripts/upstream_sync.py plan`.
3. Review changed paths, Python signature changes, vllm-ascend consumers, risk,
   and recommended validation.
4. Resolve high-risk compatibility work before or together with the upgrade.
5. Run `apply` only when the vLLM submodule worktree is clean and still at the
   planned old SHA.
6. Review the parent workspace gitlink diff.
7. Run `vllm-ascend-change-validation` on the resulting workspace diff.

## Entry point

`scripts/upstream_sync.py` provides:

- `plan`: resolve refs and create compatibility evidence without mutation;
- `apply`: verify clean/still-current preconditions and checkout the exact
  planned new SHA in detached mode.

Read only the reference needed for the active phase:

- [Behavior contract](references/behavior.md)
- [Command recipes](references/command-recipes.md)
- [Acceptance](references/acceptance.md)

## Boundaries

- Remotes, credentials, first clone, and submodule initialization belong to
  `repo-init`.
- Arbitrary PR/workspace impact and evidence aggregation belong to
  `vllm-ascend-change-validation`.
- This Skill never resolves merge conflicts, commits, pushes, or opens a PR.
- Model onboarding methodology is not part of upstream sync.

## Rules

- Never fetch implicitly.
- Never apply from a dirty vLLM worktree.
- Never apply when current HEAD differs from the planned old SHA.
- Checkout only the fully resolved planned new SHA.
- Treat static consumer detection as a risk signal, not proof of compatibility.
- Suppress only `SyntaxWarning` emitted while parsing historical source text;
  real syntax errors remain non-fatal unknowns for static analysis.
- Keep reports under `.vaws-local/upstream-sync/`.
