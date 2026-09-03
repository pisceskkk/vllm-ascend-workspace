# Repo-init command recipes

Prefer the helper scripts in `scripts/` and `.agents/scripts/` when possible.

## Probe

macOS / Linux / WSL:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_probe.py \
  --compact --network-context unknown
```

Windows:

```powershell
py -3 .agents/skills/repo-init/scripts/repo_init_probe.py `
  --compact --network-context unknown
```

If the probe reports `auth_state: unverified`, rerun only the read-only auth
check from a network-enabled execution context:

```bash
python3 .agents/scripts/github_auth_probe.py --network-context enabled
```

Do not ask the user to log in again unless that second result is
`auth_state: auth_failed`.

## Broad-init machine profile

Get the exact three-option machine-username question:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_profile.py plan
```

Apply the Git-username option:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply --choice git-username
```

Apply the random `agent#####` option:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply --choice random
```

Apply the custom option after the user gave the literal username:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply --choice custom --custom-username alice123
```

Apply the unified alias choice after the machine profile exists:

```bash
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply-alias --choice machine-username
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply-alias --choice custom --custom-alias team42
python3 .agents/skills/repo-init/scripts/repo_init_profile.py apply-alias --choice none
```

Inspect or maintain only the local identity:

```bash
python3 .agents/scripts/workspace_identity.py summary
python3 .agents/scripts/workspace_identity.py ensure
python3 .agents/scripts/workspace_identity.py set-alias team42
python3 .agents/scripts/workspace_identity.py decline-alias
```

## Low-level profile helper

Validate one user-provided name:

```bash
python3 .agents/scripts/workspace_profile.py validate alice123
```

Read the current profile summary:

```bash
python3 .agents/scripts/workspace_profile.py summary
```

## Submodules

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Resolve CI-pinned vLLM ref

Use this after `vllm-ascend/` is populated. Without an explicit user override,
the checked-out vllm-ascend HEAD's verified pin is mandatory:

```bash
python3 .agents/skills/repo-init/scripts/resolve_vllm_ci_pin.py --vllm-ascend-dir vllm-ascend
```

When the user explicitly supplied a vLLM commit, add
`--vllm-commit <sha>`. Then check out `vllm/` at the returned `vllm_ref` and
verify the result:

```bash
python3 .agents/scripts/vllm_version_pairing.py check
```

Add the same `--vllm-commit <sha>` to the check for an explicit override.
There is no automatic fallback to workflow, docs, release-tag, `main`, or the
current vLLM checkout.

## Quiet main comparison

```bash
python3 .agents/skills/repo-init/scripts/repo_topology.py compare-main --repo .
python3 .agents/skills/repo-init/scripts/repo_topology.py compare-main --repo vllm
python3 .agents/skills/repo-init/scripts/repo_topology.py compare-main --repo vllm-ascend
```

## Remote configuration

Workspace example:

```bash
python3 .agents/skills/repo-init/scripts/repo_topology.py configure   --repo .   --origin-url git@github.com:USER/vllm-ascend-workspace.git   --upstream-url git@github.com:maoxx241/vllm-ascend-workspace.git
```

`vllm-ascend` example:

```bash
python3 .agents/skills/repo-init/scripts/repo_topology.py configure   --repo vllm-ascend   --origin-url git@github.com:USER/vllm-ascend.git   --upstream-url git@github.com:vllm-project/vllm-ascend.git
```

Optionally set `gh repo set-default` during configure:

```bash
python3 .agents/skills/repo-init/scripts/repo_topology.py configure   --repo vllm-ascend   --origin-url git@github.com:USER/vllm-ascend.git   --upstream-url git@github.com:vllm-project/vllm-ascend.git   --gh-default upstream
```

## Branch tracking

```bash
python3 .agents/skills/repo-init/scripts/repo_topology.py ensure-main   --repo vllm-ascend   --remote origin
```
