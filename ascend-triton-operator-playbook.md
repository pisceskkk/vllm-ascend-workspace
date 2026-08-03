# Ascend Triton 算子工程 Playbook

> 这是 Ascend Triton Skill 组的入口文档，不再承载全部实现细节。
> 详细行为、命令、验收规则和硬件技巧以 `.agents/skills/ascend-triton-*`
> 中的 `SKILL.md`、`references/` 和 `scripts/` 为准。

## 1. 目标与边界

这组 Skill 将 Ascend Triton 算子的开发过程拆成四个可独立调用、可组合、可恢复的阶段：

| Skill | 适用场景 | 核心输出 |
|---|---|---|
| `ascend-triton-workflow` | 从需求到可发布 kernel 的端到端任务 | 父 Run Manifest、阶段链接、最终结论 |
| `ascend-triton-operator-development` | 从 PyTorch 或 GPU Triton 语义得到第一版 Ascend Triton kernel | 任务契约、语义审计、设计草图、候选 kernel |
| `ascend-triton-kernel-validation` | 判断候选是否真的由 Triton 执行并在显式矩阵上正确 | AST gate、逐 case 结果、correctness manifest |
| `ascend-triton-kernel-optimization` | 在正确性锚点上做 profiler 驱动的性能迭代 | 基线、优化轮次、keep/discard 记录、最佳候选 |

本流程以 Vector 类 kernel 为主要适用对象。涉及 `tl.dot`、Cube/Vector 混合、
跨核通信、原子规约、数据依赖的离散访问或复杂控制流时，仍可使用编排与证据框架，
但必须扩展硬件模型和验证矩阵，不能直接套用 Vector 优化结论。

## 2. 路由规则

- 用户要求“迁移、实现、生成第一版”时，使用 `ascend-triton-operator-development`。
- 用户要求“验证、对齐精度、排除 PyTorch fallback”时，使用 `ascend-triton-kernel-validation`。
- 用户要求“profile、提速、优化到目标”时，使用 `ascend-triton-kernel-optimization`。
- 请求横跨两个以上阶段，或要求完整闭环时，先使用 `ascend-triton-workflow`，再调用阶段 Skill。
- 已经缩小为单个 `torch_npu`、ACLNN 或自定义算子故障，而非 Triton kernel 工程问题时，
  转交 `ascend-operator-debug`。
- 模型级 graph/eager 分歧尚未归因到单 kernel 时，先使用 `vllm-ascend-graph-debug`。

四个 Skill 都只管理控制面证据；依赖 `torch`、`torch_npu`、Triton 编译、设备计时或
profiler 的执行必须在目标 Ascend 容器完成。远端执行前遵循仓库的 session、代码同步和
知识查询规则。

## 3. 统一状态机

```text
DISCOVER
   |
   v
DEVELOP  -- task contract + semantic audit + sketch + candidate
   |
   v
VALIDATE -- AST gate + compile/run + explicit case matrix
   |                         |
   | correctness passed      | mismatch / compile / runtime failure
   v                         v
OPTIMIZE <--------------- DEVELOP or VALIDATE diagnosis
   |
   | plan -> edit -> validate -> profile -> KEEP/DISCARD/FAIL
   v
FINALIZE -- publish best correctness-passing candidate and evidence links
```

状态转换遵循以下硬约束：

1. 没有成功的正确性 manifest，不得开始性能比较。
2. 一轮优化只验证一个主要假设，只保留一个可归因的主要改动。
3. 候选必须得到 `KEEP`、`DISCARD` 或 `FAIL`，不得覆盖唯一的 best kernel。
4. 环境、参考实现、编译、运行、数值和性能失败分别分类。
5. 同类故障重复前先查询 `.agents/knowledge/`；连续失败则进入诊断，不原样重试。
6. 最终产物只能指向全量正确性通过且性能证据有效的候选。

## 4. 阶段契约

### 4.1 开发：冻结语义，再写 kernel

开发阶段首先固定：

- 输入输出的 shape、dtype、layout、stride 和动态范围；
- in-place、alias、atomic、随机数、NaN/Inf 等副作用或边界语义；
- 参考实现与逐 dtype 容差；
- 功能用例和性能用例；
- 目标 SoC、CANN、`torch_npu`、`triton-ascend` 版本与能力快照。

然后审计每个 load/store mask、索引表达式、grid 映射、归约 identity 和累加 dtype。
第一版 kernel 以最小正确为目标：host wrapper 只负责参数整理、分配和 launch，核心计算
必须位于 `@triton.jit` 内。

详细规则见：

- `.agents/skills/ascend-triton-operator-development/references/semantic-review.md`
- `.agents/skills/ascend-triton-operator-development/references/architecture-and-codegen.md`

### 4.2 验证：先排除退化，再证明正确

验证阶段的顺序固定为：

1. AST gate：确认存在 `@triton.jit` kernel、目标 `forward` 可达路径真的 launch kernel，
   且核心计算未回退到 PyTorch。
2. 编译：覆盖会触发不同 specialization 的 case。
3. 运行：逐 case 独立记录 compilation/runtime/unsupported 状态。
4. 数值：检查 shape、dtype、stride、NaN/Inf，并使用任务声明的容差。
5. 守卫：在可行时检查越界写和非预期 in-place 修改。

只有 `passed_cases == total_cases > 0` 才可生成成功的 correctness manifest。
`unsupported` 或缺失 case 是 `inconclusive`，不能伪装成通过。

详细规则见：

- `.agents/skills/ascend-triton-kernel-validation/references/case-design.md`
- `.agents/skills/ascend-triton-kernel-validation/references/acceptance.md`

### 4.3 优化：基于证据验证单一假设

优化阶段先记录可复现基线，再按以下循环工作：

```text
profile -> classify bottleneck -> one hypothesis -> one main change
        -> full validation -> repeated benchmark -> KEEP/DISCARD/FAIL
```

性能证据必须保证：

- baseline 和 candidate 使用相同环境、case、计时口径和同步方式；
- 排除首次编译和 warmup；
- 分 case 保留重复样本和稳健统计量，而非只给一个平均值；
- 同时观察 wrapper latency 与 device kernel time；
- 正确性失败的候选不进入性能排名；
- 提升小于噪声阈值时标记 `NOISE`，不强行 `KEEP`。

详细的 Ascend 优化知识位于：

- `.agents/skills/ascend-triton-kernel-optimization/references/profiling-decision-tree.md`
- `.agents/skills/ascend-triton-kernel-optimization/references/ascend-techniques.md`

其中包括 UB live-set 预算、grid 与物理核映射、MTE/Vector/Scalar 流水、dtype、mask、
地址计算、multibuffer、尾块和小 shape 策略。所有核数、存储容量、编译选项和 API 支持
都视为环境事实：先探测并记录，缺失时标记 `unknown`，不得把单一 910B 环境的经验写成
跨芯片常量。

## 5. 证据与可恢复性

四个 Skill 使用仓库的 Run Manifest v1：

| 阶段 | Run Manifest 类型 | 关键链接 |
|---|---|---|
| workflow | `change-validation` | development、correctness、performance 子 manifest |
| development | `debug` | task contract、语义报告、草图、候选 kernel、correctness manifest |
| validation | `correctness` | kernel hash、静态检查、case 矩阵、结果汇总 |
| optimization | `performance` | 起始 correctness、baseline、每轮 candidate correctness 和测量 |

运行状态只写入未跟踪的 `.vaws-local/`。脚本进度写 `stderr`，最终机器可读 JSON 写
`stdout`。manifest 和 artifact 记录绝对路径、hash、环境与父子关系，使流程可以在中断后
恢复，也能区分“未运行”“证据缺失”“已失败”和“已通过”。

## 6. 使用顺序示例

```bash
# 1. 生成开发任务目录和 debug manifest
python3 .agents/skills/ascend-triton-operator-development/scripts/triton_development.py \
  plan --config /abs/path/development.json

# 2. 候选完成后，为验证阶段生成显式 case 计划
python3 .agents/skills/ascend-triton-kernel-validation/scripts/triton_validation.py \
  plan --config /abs/path/validation.json

# 3. 在目标 Ascend 容器执行 case，并逐条 record；最后 analyze
# 4. correctness passed 后，创建优化实验并逐轮 record
python3 .agents/skills/ascend-triton-kernel-optimization/scripts/triton_optimization.py \
  plan --config /abs/path/optimization.json

# 5. 将终态子 manifest 链接到 workflow，并 finalize
python3 .agents/skills/ascend-triton-workflow/scripts/triton_workflow.py \
  finalize --manifest /abs/path/workflow-manifest.json
```

具体参数以各 Skill 的 `references/command-recipes.md` 为准。

## 7. 来源与维护原则

本 Skill 组综合了以下材料的互补部分：

- [`simple-vector-triton-gpu-to-npu`](https://gitcode.com/Ascend/agent-skills/blob/master/community/Op/simple-vector-triton-gpu-to-npu/SKILL.md)：语义审计、最小迁移、mask 职责与错误分流。
- [`vector-triton-ascend-ops-optimizer`](https://gitcode.com/Ascend/agent-skills/blob/master/community/Op/vector-triton-ascend-ops-optimizer/SKILL.md)：UB、分核、流水、访存与 dtype 优化假设。
- [`AscendOpGenAgent`](https://github.com/Just-it/AscendOpGenAgent)：阶段编排、AST fallback gate、全 shape 验证和 keep/discard 状态机。

材料快照日期为 2026-08-03。外部项目中的固定阈值、设备常量和命令没有直接提升为本仓库
的永久事实；快变能力应进入 `.agents/knowledge/`，经目标环境验证后再使用。修改任一 Skill
时，应同步检查其 `SKILL.md`、`scripts/`、`references/`、`tests/` 和 `agents/openai.yaml`。
