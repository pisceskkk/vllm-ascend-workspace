# vllm-ascend-workspace

**[中文](README.md)** | **English**

A composable local development scaffold for working on [vLLM](https://github.com/vllm-project/vllm) and [vLLM Ascend Plugin](https://github.com/vllm-project/vllm-ascend) in a single workspace, with built-in AI Agent skills for automated environment setup, remote NPU machine management, and code synchronization.

## What problem does this solve

Developing vLLM Ascend typically involves editing code locally, running tests on remote Ascend NPU servers, and tracking upstream vLLM changes — all of which require repetitive Git, SSH, and environment configuration.

`vllm-ascend-workspace` wraps these operations into a set of AI Agent skills. You can ask an Agent to handle them in natural language, or ignore the skills entirely and use it as a plain multi-repo workspace.

## Quick start

```bash
# Clone the repository
git clone https://github.com/pisceskkk/vllm-ascend-workspace.git
cd vllm-ascend-workspace

# Initialize submodules
git submodule update --init --recursive
```

If you use an Agent-capable IDE (Cursor, Windsurf, etc.) or terminal tool (Claude Code, Codex CLI, etc.), you can complete the rest of the setup in natural language:

> "Initialize this workspace and set me up for vLLM Ascend development."

The Agent will detect your environment, install required tools, and configure Git remotes and forks.

## Built-in skills


| Skill                  | Purpose                                                                                      | When to use                                                |
| ---------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **repo-init**          | Install GitHub CLI, authenticate, initialize submodules, configure forks and remote topology | After first clone                                          |
| **machine-management** | Add, verify, repair, or remove a remote Ascend NPU server and its managed container          | When setting up a remote NPU dev machine                   |
| **session-management** | Create, inspect, and clean isolated sessions: local worktree, remote container, state namespace, and resource leases | For parallel remote work or multiple agents |
| **remote-toolbox**    | Structured target/probe/exec/job/sync/service/artifact/cleanup tools for remote containers | When agents need local-tool-like control of a remote session container |
| **remote-code-parity** | Sync the full local workspace state (including uncommitted changes) to a remote container    | Triggered automatically before remote test or service runs |
| **modelscope**       | Download, resume, status-check, and SHA256-verify ModelScope model weights                  | When model weights need to be downloaded into an explicit local directory |
| **vllm-ascend-serving** | Launch a vLLM Ascend inference service on a remote container, with NPU probing, auto card selection, and incremental restart | When you need an inference service on a remote machine |
| **vllm-ascend-benchmark** | Run `vllm bench serve` for one code state, with multi-run warmup and statistical aggregation | When measuring throughput or latency for the current state |
| **ascend-memory-profiling** | Profile and attribute HBM memory usage on Ascend NPU, with per-component breakdown and evidence chains | When you need to analyze memory consumption of a vLLM serving workload |
| **ascend-profiling-collection** | Collect Ascend torch-profiler data: start service, bracket profile window, run workload, remote analyse, and write a manifest | When you need kernel_details/trace_view captures |
| **ascend-profiling-analysis** | Analyze collected profiler roots/manifests and generate step/layer/operator/cross-rank reports | When you need to analyze profiling output |
| **vllm-ascend-graph-debug** | Diagnose graph-mode failures through stage classification, controlled experiments, and graph/eager intermediate-state comparison | When debugging graph compile, capture, replay, or correctness divergence |
| **vllm-ascend-correctness-validation** | Plan and compare baseline/candidate, eager/graph, offline/online, and task-metric correctness evidence | When validating accuracy, comparing outputs, or checking correctness regressions |
| **vllm-ascend-change-validation** | Analyze code impact, plan minimum sufficient validation, and aggregate downstream evidence into a PR report | When validating a workspace diff, commit range, or PR |
| **vllm-ascend-performance-regression** | Control alternating baseline/candidate A/B experiments, enforce config parity, and assess variance and regression thresholds | When checking throughput, latency, HBM, or other performance regressions |
| **vllm-ascend-distributed-debug** | Diagnose multi-node, multi-rank, and collective failures from topology, process groups, endpoints, and per-rank events | When a failure depends on ranks, parallel topology, nodes, or distributed communication |
| **ascend-operator-debug** | Reduce a model symptom to one Ascend operator call and run an explicit dtype/shape/layout/mode reference matrix | When an operator is already suspected or a minimal operator reproducer is needed |
| **vllm-ascend-pd-serving** | Orchestrate prefill/decode services, connector config, proxy health, rollback, and KV request smoke on a Session Group | When deploying PD disaggregation with NIXL, Mooncake, or another connector |
| **vllm-ascend-upstream-sync** | Compare vLLM refs, locate API and vllm-ascend consumer impact, and guard the submodule pointer update | When intentionally upgrading the vLLM submodule and assessing compatibility |


All skills are **optional**. Use any subset, or none at all.

## Usage examples

When talking to an Agent:

```
# Initialization
"Help me initialize this repo"
"Help me set up this repo"

# Machine management
"Add these two servers, IPs are x.x.x.1 and x.x.x.2, password is xxxx"
"Set up this server for me, IP is x.x.x.x, password is xxxx"
"Remove the x.x.x.x server"

# Code sync
"Sync my code to the server and rebuild"

# Model weight download
"Download Qwen/Qwen3-32B from ModelScope to /root/Qwen/Qwen3-32B and show progress"
"Verify the ModelScope weights under /root/Qwen/Qwen3-32B"

# Serving (--model must point to weights already present on the remote container)
"Launch a 4-card inference service on x.x.x.x using /home/weights/Qwen3-32B-W8A8 with W8A8 quantization"
"Check the service status on x.x.x.x"
"Stop the service on x.x.x.x"

# Benchmarking
"Run a benchmark on x.x.x.x with Qwen3.5-35B, 4 cards, 5 runs keep last 4"
"Compare throughput between main and this PR"
```

## Repository layout

```
.
├── vllm/                  # Upstream vLLM (Git submodule)
├── vllm-ascend/           # vLLM Ascend Plugin (Git submodule)
├── .agents/
│   ├── skills/
│   │   ├── repo-init/         # Workspace initialization skill
│   │   ├── machine-management/    # Remote machine management skill
│   │   ├── session-management/    # Parallel session isolation skill
│   │   ├── remote-toolbox/        # Structured remote toolbox
│   │   ├── remote-code-parity/    # Code synchronization skill
│   │   ├── modelscope/            # ModelScope weight download and verification skill
│   │   ├── vllm-ascend-serving/   # Inference serving skill
│   │   ├── vllm-ascend-benchmark/ # Performance benchmarking skill
│   │   ├── ascend-memory-profiling/ # Memory profiling skill
│   │   ├── ascend-profiling-collection/ # Torch profiler collection skill
│   │   ├── ascend-profiling-analysis/ # Profiling analysis/report skill
│   │   ├── vllm-ascend-graph-debug/ # Graph-mode debugging skill
│   │   ├── vllm-ascend-correctness-validation/ # Correctness validation skill
│   │   ├── vllm-ascend-change-validation/ # Change impact and PR validation skill
│   │   ├── vllm-ascend-performance-regression/ # A/B performance regression skill
│   │   ├── vllm-ascend-distributed-debug/ # Distributed failure diagnosis skill
│   │   ├── ascend-operator-debug/ # Isolated Ascend operator diagnosis skill
│   │   ├── vllm-ascend-pd-serving/ # PD disaggregated serving skill
│   │   └── vllm-ascend-upstream-sync/ # vLLM upstream upgrade skill
│   ├── lib/               # Shared local-state library
│   └── scripts/           # Shared helper scripts
├── .cursor/rules/         # Cursor IDE specific rules
├── .trae/                 # TRAE IDE specific rules and skills
├── AGENTS.md              # Cross-tool Agent instructions (Agents read this)
├── CLAUDE.md              # Claude Code instruction entry point
└── README.md              # Chinese README (default)
```

## Design principles

- **Nothing is mandatory** — All skills are optional. Developers choose what to use.
- **Local state stays untracked** — User-specific remotes, auth, and machine config live only in the untracked `.vaws-local/` directory.
- **Parallel tasks stay isolated** — Remote parallel work should use sessions: each task gets its own local worktree, remote container, state namespace, and resource leases.
- **Remote operations are structured** — Agents should prefer the remote toolbox for JSON results, observable logs, resumable artifact manifests, and cleanup-capable state.
- **Submodules point to community** — `.gitmodules` always targets `vllm-project` official repos. Personal forks are a local runtime concern.
- **Agent-driven, not Agent-dependent** — Everything can be done manually. Agent skills just make it more convenient.

## Recommended remote topology

Skills recommend the following topology, but never enforce it:


| Repository    | `origin`             | `upstream`                       |
| ------------- | -------------------- | -------------------------------- |
| workspace     | `pisceskkk/vllm-ascend-workspace` or your fork | `maoxx241/vllm-ascend-workspace` |
| `vllm`        | Your fork (optional) | `vllm-project/vllm`              |
| `vllm-ascend` | Your fork            | `vllm-project/vllm-ascend`       |


## Multi-tool support

This repository supports mainstream AI coding tools:


| File             | Tools covered                                     |
| ---------------- | ------------------------------------------------- |
| `AGENTS.md`      | Codex CLI, GitHub Copilot, Cursor, TRAE, OpenCode |
| `CLAUDE.md`      | Claude Code                                       |
| `.cursor/rules/` | Cursor                                            |
| `.trae/`         | TRAE                                              |


## Roadmap

### Done

- **repo-init** — Workspace initialization: GitHub CLI install, auth, submodules, fork & remote topology
- **machine-management** — Remote machine management: add, verify, repair, remove Ascend NPU servers and managed containers
- **remote-code-parity** — Code sync: push full local workspace state (including uncommitted changes) to remote containers
- **vllm-ascend-serving** — Service launch: idle NPU detection, idle port detection, one-click vLLM Ascend inference serving
- **vllm-ascend-benchmark** — Single-code-state online performance measurement: single-run / multi-run (warm-service) mode, warmup exclusion, and statistical aggregation
- **ascend-memory-profiling** — Memory profiling: collect and analyze HBM usage, per-component breakdown (fixed overhead, weights, KV cache, HCCL, activations, runtime), with msprof component-level attribution

### Planned

- **Accuracy testing & aisbench integration** — Automated evaluation based on aisbench, with HTML report analysis, system scheduling assessment, and DP balance analysis
- **Performance profiling** — Automatic operator latency breakdown, hot operator AIC/AIV/MTE2 ratio analysis, AICPU operator identification, host bound detection and diagnosis
- **Sync-break optimization** — Provide async copy overlap strategies for specific cases to reduce synchronization overhead
- **Compute graph analysis** — Build model compute graphs, generate theoretical performance evaluation reports and optimization recommendations
- **External knowledge base** — Integrate external knowledge sources to extend Agent capabilities

## License

This scaffold repository is licensed independently from its submodules. `vllm/` and `vllm-ascend/` each follow their respective upstream licenses.
