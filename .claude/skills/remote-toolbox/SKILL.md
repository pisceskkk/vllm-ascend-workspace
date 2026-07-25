<!-- Generated Claude Code shim from .agents/skills/remote-toolbox/SKILL.md. Do not edit. -->
---
name: remote-toolbox
description: Execute ad hoc structured operations against an already-managed VAWS remote container, including target resolution, remote facts, one-off commands or jobs, named artifact transfer, and job cleanup. Use only when no domain workflow owns the requested remote operation. Do not use for machine setup, session lifecycle, code parity, serving, benchmarks, correctness, profiling, or domain debugging with a dedicated Skill.
---

# VAWS Remote Toolbox

Canonical skill source:

`.agents/skills/remote-toolbox/SKILL.md`

Before using this skill:

1. Read the canonical skill file above.
2. Follow its routing rules, entrypoints, guardrails, and acceptance criteria.
3. Use `.remote-dev` companion tools for ordinary remote endpoint read/edit/bash/search/patch work.
4. Use this Claude project skill only for the domain workflow described by the canonical source.
