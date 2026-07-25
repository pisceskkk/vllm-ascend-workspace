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

Planning may analyze any locally resolvable ref pair, but records the actual
worktree HEAD and cleanliness rather than assuming HEAD equals `old-ref`. A plan
is apply-ready only when the observed clean HEAD is the exact old SHA.

## Applying

Apply is intentionally narrow:

1. vLLM worktree has no tracked or untracked changes;
2. current HEAD has not changed since planning;
3. the plan-time HEAD equals the planned old SHA;
4. checkout uses the planned full new SHA in detached mode;
5. observed HEAD is verified;
6. no commit, push, fetch, merge, or PR operation occurs.

The parent repository will show the submodule gitlink change. General Change
Validation consumes that final diff.
