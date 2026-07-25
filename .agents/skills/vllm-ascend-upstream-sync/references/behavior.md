# vLLM upstream sync behavior contract

## Planning

Both refs must already resolve locally to commits. Planning is read-only and
records:

- exact old and new SHAs;
- renamed/added/modified/deleted upstream paths;
- top-level function, class, and method signature changes in changed Python files;
- vllm-ascend files importing changed modules or mentioning removed/changed symbols;
- path-based validation recommendations.

Risk is:

- `high`: changed/removed API and at least one static consumer;
- `medium`: API risk or consumers, but not both;
- `low`: neither signal.

Static analysis can miss dynamic imports, monkey patches, generated code, and
semantic behavior changes. It is a review aid, never proof of compatibility.

## Applying

Apply is intentionally narrow:

1. vLLM worktree has no tracked or untracked changes;
2. current HEAD equals the planned old SHA;
3. checkout uses the planned full new SHA in detached mode;
4. observed HEAD is verified;
5. no commit, push, fetch, merge, or PR operation occurs.

The parent repository will show the submodule gitlink change. General Change
Validation consumes that final diff.
