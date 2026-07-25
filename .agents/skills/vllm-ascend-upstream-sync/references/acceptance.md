# vLLM upstream sync acceptance

## Plan

- [ ] Old and new refs resolve to immutable full SHAs.
- [ ] Changed and renamed upstream paths are listed.
- [ ] Python signature additions, removals, and changes are reported.
- [ ] Historical invalid-escape `SyntaxWarning` messages do not pollute structured planning output.
- [ ] vllm-ascend static consumers are linked to upstream modules or symbols.
- [ ] Risk and recommended validation are explicit.
- [ ] The actual HEAD and cleanliness at plan time are recorded; apply readiness is true only when that clean HEAD is the requested old ref.
- [ ] Planning performs no fetch, checkout, merge, or write to either source repo.

## Apply

- [ ] Dirty vLLM worktrees are rejected.
- [ ] HEAD drift from the observed plan-time SHA is rejected.
- [ ] Analysis-only plans created while HEAD differs from the old SHA are rejected.
- [ ] Only the exact planned new SHA is checked out.
- [ ] Detached HEAD is verified after checkout.
- [ ] The script does not commit, push, or alter remotes.
- [ ] Parent gitlink diff is handed to Change Validation.
