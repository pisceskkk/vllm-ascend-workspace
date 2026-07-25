# vLLM upstream sync acceptance

## Plan

- [ ] Old and new refs resolve to immutable full SHAs.
- [ ] Changed and renamed upstream paths are listed.
- [ ] Python signature additions, removals, and changes are reported.
- [ ] vllm-ascend static consumers are linked to upstream modules or symbols.
- [ ] Risk and recommended validation are explicit.
- [ ] Planning performs no fetch, checkout, merge, or write to either source repo.

## Apply

- [ ] Dirty vLLM worktrees are rejected.
- [ ] HEAD drift from the planned old SHA is rejected.
- [ ] Only the exact planned new SHA is checked out.
- [ ] Detached HEAD is verified after checkout.
- [ ] The script does not commit, push, or alter remotes.
- [ ] Parent gitlink diff is handed to Change Validation.
