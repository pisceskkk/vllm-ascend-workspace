# Ascend Triton 算子编排与优化 Playbook

> 基于 `simple-vector-triton-gpu-to-npu`、`vector-triton-ascend-ops-optimizer`
> 和 `AscendOpGenAgent` 的综合整理。快照日期：2026-08-03。
>
> 本文以 Vector 类算子为主；涉及 `tl.dot`、Cube/Vector 混合、跨核通信、原子规约或复杂控制流时，
> 必须扩展硬件模型与验证矩阵，不能机械套用 Vector 流程。

## 1. 结论先行

三份材料适合组合使用，而不适合任选其一：

| 材料 | 最有价值的部分 | 主要缺口 |
|---|---|---|
| `simple-vector-triton-gpu-to-npu` | 迁移前语义审计、最小修改、按错误类型分流、load/store mask 职责分离 | 规则较单体化；若把核数、UB、API 替换和 `torch.compile` 等表述当永久事实，容易随芯片和软件版本失效 |
| `vector-triton-ascend-ops-optimizer` | UB live-set 预算、批量处理 Token/Block、MTE/Vector/Scalar 流水、访存与 dtype 意识 | 固定 192 KB/85 KB、固定测试命令和“未达 x 倍不停止”过于绝对；缺少噪声控制、失败分类和可恢复状态 |
| `AscendOpGenAgent` | Phase 化编排、AST 防 PyTorch 退化、全 shape 精度闸门、结构化 JSON、专用 kernel 路由、AutoResearch keep/discard 状态机 | 工程复杂且与 Claude hooks 耦合；部分阈值和路径是框架策略，不是硬件定律 |

推荐的统一方法是：

1. 用迁移 Skill 的“语义先行 + 最小可运行版本”建立正确性锚点。
2. 用 AscendOpGenAgent 的“阶段机 + 强制产物 + L1 闸门”控制流程。
3. 用优化 Skill 的“UB、流水、分核、dtype、访存”作为优化假设库。
4. 用 AutoResearch 的 `plan → edit → verify → profile → keep/discard` 做可恢复迭代。
5. 所有芯片参数和编译能力均从目标环境探测、记录版本并实测，不写成跨版本常量。

## 2. 统一编排架构

### 2.1 角色划分

建议拆成七个逻辑角色。它们可以由一个 Agent 顺序执行，也可以实现为多个 Skill；关键是职责和产物不能混淆。

| 角色 | 只负责什么 | 关键产物 |
|---|---|---|
| Orchestrator | 状态迁移、预算、失败分类、选择下一步 | `state.json`、`history.jsonl` |
| Task Extractor | 冻结功能契约、shape/dtype/layout/stride/动态范围 | `task.yaml`、参考实现、输入生成器 |
| Semantic Reviewer | 解释原 GPU/Torch 代码的数据流、mask、索引与副作用 | `semantic_report.md` |
| Kernel Designer | 设计 grid、tiling、UB 预算、算法和候选特化 | `sketch.md` |
| Kernel Generator | 只实现 Triton kernel 和最薄 host wrapper | `candidate.py` |
| Verifier | 编译、全用例精度、越界/NaN/Inf、退化检查 | `verify.json` |
| Profiler/Optimizer | 测基线、归因瓶颈、一次只验证一个优化假设 | `profile.json`、`round.json` |

“设计”和“生成”应分离。草图必须先回答分核、数据切片、UB 峰值和尾块语义，生成器才落代码；否则模型很容易直接复制 GPU kernel，只把 `cuda` 改成 `npu`。

### 2.2 状态机

```text
DISCOVER
  ↓
CONTRACT ──任务/参考非法──> BLOCKED
  ↓
BASELINE ──环境失败──────> BLOCKED
  ↓
DESIGN → IMPLEMENT → VERIFY
             ↑          │
             └─精度失败─┘
                         ↓ 全量通过
                       PROFILE
                         ↓
                 PLAN → EDIT → VERIFY → PROFILE
                   ↑                         │
                   ├──── DISCARD（无提升）──┤
                   └──── KEEP（有提升）─────┘
                         │
               连续失败达到阈值
                         ↓
                     DIAGNOSE → REPLAN
                         │
               预算耗尽/目标达成
                         ↓
                       FINISH
```

必须固化的状态不变量：

- 精度未全量通过，不得进入性能比较。
- 一轮只改一个主要假设；否则无法归因提升或退化。
- 每轮候选必须 `KEEP`、`DISCARD` 或 `FAIL`，不得覆盖唯一的 best kernel。
- 环境/参考实现失败与 kernel 失败分开记录，不能让 Agent 用改 kernel 的方式“修”环境。
- 连续多轮同类失败后进入 `DIAGNOSE`，禁止原样重试。
- 结束时只发布“全量精度通过且性能证据有效”的最佳版本。

## 3. 端到端流程

### Phase 0：环境与能力快照

在任何代码转换前记录：

- SoC 精确型号、可见设备、Vector/Cube 核数量。
- CANN、驱动、`torch`、`torch_npu`、`triton-ascend` 版本和 commit。
- 可用编译选项，例如 `multibuffer`、`unit_flag`、`auto_blockify_size`。
- UB/L1/L0/L2 信息若无法可靠查询，标为 `unknown`，不要套用 910B 常量。
- profiler 可用性、时钟/功耗模式、是否有其他负载。

输出 `environment.json`。同一轮 baseline/candidate 必须使用同一环境快照。

### Phase 1：冻结算子契约

建立可执行的 `task.yaml`，至少包含：

```yaml
op_name: example
semantics: "..."
inputs:
  - {name: x, dtype: [fp16, bf16], shape: ["M", "N"], layout: contiguous}
dynamic_ranges: {M: [1, 4096], N: [1, 8192]}
outputs: []
side_effects: []
target_arch: ascend910b2
correctness:
  fp16: {rtol: 1e-3, atol: 1e-3}
  fp32: {rtol: 1e-4, atol: 1e-4}
performance_cases: []
```

用例矩阵至少覆盖：

- 最小值、典型值、最大值。
- 2 的幂、非 2 的幂、对齐和非对齐尺寸。
- 空/单元素维度（语义允许时）。
- 连续和任务声明支持的非连续 layout。
- 可能引发溢出、下溢、NaN/Inf、重复索引的数值分布。
- 每一种 dtype 和影响控制流的标量参数。

多 shape 任务必须逐 case 保留，禁止为了快速通过裁成单 case。

### Phase 2：迁移前语义审计

对每个 `tl.load`、`tl.store`、mask 和索引表达式做表格化审计：

| 项目 | 必答问题 |
|---|---|
| `tl.load` | 地址来自何处？哪些 lane 有效？无效 lane 的值是否参与 reduce/div/exp/index？正确 identity 是什么？ |
| `tl.store` | 哪些输出 lane 合法？是否错误复用了输入有效性 mask，造成合法输出未写？ |
| 索引 | 连续、规则跨步还是数据依赖离散访问？是否可能越界、重复或为负？ |
| Grid | 每个 program 代表哪个逻辑任务？扁平化后如何还原？会引入多少 `//`、`%` 和分支？ |
| 数值 | 累加 dtype、数学近似、归约顺序变化是否改变容差？ |
| 副作用 | atomic、in-place、随机数、别名和输出初始化是否属于语义？ |

特别注意：load mask 表示“哪些输入地址可读”，store mask 表示“哪些输出地址应写”。二者常常不同。

### Phase 3：建立双基线

需要两个不同目的的基线：

1. 正确性基线：可读、可信、尽量直接的 PyTorch/torch_npu 参考实现。
2. 性能基线：当前 NPU 实现或可比的 NPU 原生实现。

GPU latency 可作为跨平台背景数据，但不能单独作为 NPU 优化验收线。硬件、功耗、软件栈不同，GPU/NPU 比值只说明现状，不解释瓶颈。

性能基线应：

- 排除首次编译和 warmup。
- 显式同步或使用 profiler 的设备时间。
- 同一 shape 重复多次，至少保存 median/p50，并建议保存 p10/p90 或 MAD。
- 分 shape 记录，不能只保存一个均值。
- 同时保存端到端 wrapper latency 与目标 kernel device time，避免把 host 路由/同步误判为 kernel 性能。

### Phase 4：设计第一版 NPU kernel

第一版目标是“最小正确”，不是极致性能：

- host 侧只做分配、shape/stride 提取和 kernel launch。
- 核心计算必须在 `@triton.jit` 内，AST 检查禁止偷偷回退到 PyTorch。
- 优先保留原算法和正确的 padding/identity 语义。
- 先选一个保守 tile，避免 UB 溢出。
- Grid 映射选择必须写出理由，不默认“GPU grid 原样保留”或“永远固定满核”。

可用两种主要分核模式，必须实测选择：

```python
# 模式 A：物理核数量级的 grid + 核内 grid-stride loop
pid = tl.program_id(0)
nprog = tl.num_programs(0)
for task_id in tl.range(pid, task_count, nprog):
    ...
```

```python
# 模式 B：保留较多逻辑 blocks，使用当前软件栈支持的
# TRITON_ALL_BLOCKS_PARALLEL / auto_blockify 等能力合并调度。
```

模式 A 减少多轮下发，但核内循环、整除/取模和负载不均可能变成瓶颈；模式 B 更接近 GPU 写法，但依赖编译器版本和配置。不要把其中一个写成普适定律。

### Phase 5：全量正确性闸门

验证顺序：

1. AST/静态检查：存在 Triton kernel、wrapper 确实调用它、核心计算没有 PyTorch fallback。
2. 编译：每个会触发不同 specialization 的 case 都编译。
3. 运行：逐 case 独立捕获错误，后一个 case 不因前一个失败而丢失。
4. 数值：shape/dtype/stride 一致；检查 NaN/Inf；按 dtype 和算法设容差。
5. 守卫：能使用 guard/sentinel 时检查越界写；输入可保留副本检查意外 in-place 修改。
6. 报告：`passed_cases == total_cases > 0` 才允许 benchmark。

不要把“进程成功退出”当作精度通过，也不要把 benchmark 的 case 成功数复制成 verify 成功数。

### Phase 6：基线 profiling 与瓶颈分类

先估算理论下界，再看实测流水：

```text
memory_floor ≈ (GM read bytes + GM write bytes) / effective_GM_bandwidth
vector_floor ≈ vector_elements / effective_vector_throughput
launch_floor ≈ measured_empty_or_tiny_kernel_cost
```

然后用 `msprof op`/流水图回答：

| 观测 | 更可能的瓶颈 | 首选实验 |
|---|---|---|
| Block 数远大于物理核、device time/host dispatch 高 | 多轮下发 | 合并 grid、物理核 grid + grid-stride loop |
| Block 数远小于物理核 | 并行度不足 | 减小核间 tile 或增加任务维度 |
| MTE2/MTE3 长、Vector 空洞多 | 访存或流水断流 | 连续搬运、增大 tile、multibuffer、load/compute/store 交织 |
| Scalar/FLOWCTRL 长 | 地址、mask、整除取模、dtype 降级 | 简化索引、维度合并、constexpr、避免 i64/不支持比较 |
| Vector 长且接近理论 | 计算受限 | libdevice、近似数学、pass 融合、减少冗余运算 |
| UB overflow 或编译器无法 multibuffer | live set 过大 | 降 tile、缩短 live range、早 store、减少转换/临时量 |
| 小 shape 极慢 | launch/host 开销 | kernel fusion、shape 特化；不要靠增大单核工作量假装解决 |

### Phase 7：单假设优化循环

每轮保存以下记录：

```json
{
  "round": 7,
  "parent": "best_sha256",
  "hypothesis": "将连续 token 从 1 批到 4，提升 MTE 有效带宽",
  "change": "TOKENS_PER_ITER=1 -> 4",
  "expected_signal": "MTE 空洞减少，UB peak < budget",
  "verify": {"passed": 48, "total": 48},
  "performance": {"median_us": 12.4, "baseline_median_us": 15.1},
  "decision": "KEEP"
}
```

推荐决策规则：

- `VERIFY_FAIL`：不测性能，修正确性或回退。
- `ENV_FAIL`：停止改 kernel，先修环境。
- `PERF_REGRESSION`：discard。
- `NOISE`：增加重复次数或做 A/B 交替测量。
- `IMPROVED`：只有提升超过预设噪声阈值且关键 shape 无不可接受退化才 keep。
- 连续三轮同类失败：进入 diagnosis，重估瓶颈，不再改同一参数。

### Phase 8：多 shape 特化与路由

泛用 kernel 无法兼顾不同形态时，按“性能机理”分组，而不是一 case 一个 kernel：

- reduce-last 与 reduce-non-last。
- 连续、广播、规则跨步、离散索引。
- 小 shape（launch bound）、中 shape、大 shape（带宽/UB bound）。
- 对齐且可安全去 mask，与非对齐尾块。
- dtype 或影响算法的 constexpr 参数。

专用 kernel 的采纳门槛：本组全量精度通过、相对泛用 kernel 有稳定提升、路由开销计入端到端测量；未命中必须回退到泛用 kernel。

### Phase 9：报告和知识沉淀

最终报告至少包含：

- 环境快照和源码哈希。
- 功能契约与完整 case 数。
- baseline/best 的逐 shape 延迟、聚合口径和噪声。
- 最终采用的优化及其证据。
- 尝试但 discard 的优化与原因。
- 未覆盖的架构、dtype、shape、layout。
- profiler 证据和原始产物路径。

知识条目只在“原因已确认 + 修复已验证”后沉淀。表述应包含适用的 SoC、软件版本、触发特征和反例，避免写成无条件规则。

## 4. Ascend Triton 硬件特有技巧

### 4.1 Grid 是物理资源映射问题，不只是逻辑任务编号

GPU Triton 常让 grid 与逻辑 block 数同阶，硬件再调度；Ascend 文档强调 grid 与物理 AI Core 拓扑更强绑定。优化时要同时控制：

- 启动 program 数量是否接近有效核数。
- 每核循环次数是否均衡，front/tail 核工作量差尽量不超过一个 tile。
- 扁平化逻辑维度后 `//`、`%`、分支的 Scalar 成本。
- 2D/3D grid 是否满足目标版本的拓扑约束。

经验上，1D grid + grid-stride loop 是稳健候选，不是必胜答案。规则二维任务若被强行扁平化并产生大量除模，可能比合法的多维映射更慢。

### 4.2 UB 预算看峰值 live set，不看输入总大小

对 192 KB UB 的 910B/A2 场景，85 KB 是“为双缓冲和临时量留余量”的经验值，不是 ABI。通用计算应写成：

```text
usable_per_stage = floor(UB_bytes × safety_factor / pipeline_stages)
tile_count <= usable_per_stage / peak_live_bytes_per_tile
```

其中 `peak_live_bytes_per_tile` 要按程序点做 live-range 分析，计入：

- 同时存活的 load tensor。
- fp16/bf16 转 fp32 后扩大的临时量。
- reduce accumulator、mask、index tensor、broadcast/reshape 临时量。
- store 前仍存活的输入和结果。
- 编译器为 padding、slice、double buffer 引入的空间。

优化方向是缩短 live range：计算完就 store、分阶段复用、避免一次把所有输入都 load 后再统一计算。tile 越大不一定越快；一旦阻断 multibuffer 或造成 spill，性能会陡降。

### 4.3 MTE–Vector–Scalar 三流水要同时观察

Ascend Vector kernel 的目标不是单纯减少指令数，而是让地址生成、GM↔UB 搬运、Vector 计算和写回重叠：

- 多个独立 load 可提前发射，但不要增加无用 live set。
- `load → compute → store` 分批交织，通常比“全部 load → 全部 compute → 全部 store”更利于流水。
- 循环迭代的地址应由 base + iteration × stride 独立计算，避免跨迭代数据依赖。
- 没有核内 tiling/多次搬运时，multibuffer 没有发挥空间。
- 多写流可在结果就绪后尽早 store，减少 UB 占用并给 MTE3 更多重叠机会。

### 4.4 `mask`、`other` 与 `care_padding=False` 必须做数据流证明

三者不能按风格统一替换：

- 首版使用语义正确的 mask 和 identity，先建立精度锚点。
- `other`/默认 padding 可能引入 Vector 填充，从而让 MTE 与 Vector 产生依赖。
- 只有能证明无效 lane 不参与后续结果，或在使用前被可靠覆盖，才尝试 `care_padding=False`。
- sum 的 padding identity 是 0，max 通常是 `-inf`，min 通常是 `+inf`；规约场景不能随意移除 padding。
- store mask 只保护合法输出，不应因输入索引无效而漏写本应定义的输出值。

每次改变 padding 策略都必须回归：尾块、全 mask、极小 shape、reduce identity、NaN/Inf。

### 4.5 避免向量操作退化为 Scalar

高风险来源：

- int64 算术、比较、除法和取模。
- 目标硬件不支持的整数 compare/reduce/arg 操作。
- 数据依赖的离散 mask/地址。
- 内层循环里重复的 shape/stride/address 计算。

可选手段：

- 在数值范围允许时改用 int32；不能为了 vectorize 把超过精确范围的索引盲目转 fp32。
- 把只依赖 shape 的量放到 host 或标成合理的 `tl.constexpr`。
- 合并维度，让地址成为线性表达式，减少 `//`/`%`。
- 把循环不变量的 load 和计算外提。
- 用 profiler 确认 Scalar/FLOWCTRL 确实下降；源码“看起来向量化”不等于后端没有降级。

### 4.6 连续搬运优先，离散访问显式降维处理

- 规则连续的多行数据尽量一次 load，起始地址和搬运长度满足目标 SoC 的有效对齐。
- 规则二维索引用 `tl.arange` 构造，避免从 GM 读取本可计算的索引表。
- 对 `index_select`/gather 类数据依赖离散地址，先按连续行或连续内段分组；二维大 mask 可能被 lower 成大量 Scalar 循环。
- 输入是否必须 `.contiguous()` 属于算子契约：若接口承诺支持任意 stride，就不能在 wrapper 悄悄改变语义或把拷贝耗时排除在端到端指标外。

### 4.7 `tl.constexpr` 是 specialization 预算

适合静态化：

- 用于 `tl.arange`、reshape/slice shape、算法分支且取值集合很小的参数。
- 稳定的 hidden size、head dim、block size。

不适合全部静态化：

- 高频变化的 batch/sequence length。
- 取值空间很大的动态参数。

过度 `constexpr` 会造成编译组合爆炸、缓存膨胀和冷启动变慢。优化报告应同时记录 warm latency 与 compile/cache 成本。

### 4.8 Pass fusion 与 kernel split 双向选择

- 多 pass 反复读同一 GM 数据，且合并后 UB 可承受：优先融合，减少 GM 往返。
- 融合导致 live set 超限、无法流水、编译过慢或不同 shape 的最优策略冲突：拆 kernel 或按 shape 特化。
- 是否融合由“节省的 GM 时间”与“新增 UB/同步/编译成本”共同决定，不能只看 kernel 数量。

### 4.9 Ascend 专用编译选项应纳入 autotune

除了 BLOCK/tile/grid，还可在版本支持时把这些作为实验维度：

- `multibuffer`
- `unit_flag`
- `auto_blockify_size`
- CV kernel 的 vector/cube loop tiling 和 balance 选项

编译选项必须写入环境快照与候选哈希；不同选项产生的是不同候选，不能只保存 Python 源码。

## 5. 优化优先级

推荐按证据驱动，而不是固定扫 13 个技巧：

1. 正确性和越界。
2. Grid/物理核映射与负载均衡。
3. UB 峰值与 tile 大小。
4. 连续 GM 访问和冗余搬运。
5. MTE/Vector/Scalar 流水重叠。
6. Scalar 降级、地址与 mask 简化。
7. pass fusion、循环不变量、load 重排。
8. libdevice/数学近似（需单独数值授权）。
9. autotune。
10. 多 shape 专用 kernel 与路由。

如果 profiler 已显示某一流水接近理论上限，应跳过不相关优化点。固定顺序适合作为漏项 checklist，不适合作为诊断器。

## 6. 建议的产物目录

```text
run-<op>-<timestamp>/
├── environment.json
├── task.yaml
├── semantic_report.md
├── reference.py
├── sketch.md
├── baseline/
│   ├── verify.json
│   └── profile.json
├── rounds/
│   ├── r000/
│   │   ├── candidate.py
│   │   ├── verify.json
│   │   ├── profile.json
│   │   └── round.json
│   └── ...
├── best/
│   ├── kernel.py
│   └── manifest.json
├── history.jsonl
├── state.json
└── report.md
```

`manifest.json` 应把源码、编译配置、环境和数据集哈希绑定起来，保证“同一份性能数据”确实对应“同一份 kernel”。

## 7. 对现有三套材料的具体取舍

### 应直接继承

- 迁移前生成语义报告。
- 首版最小改动、先跑通再优化。
- load/store mask 分责。
- 多 shape 全量验证，严格精度闸门。
- AST 检查 PyTorch fallback。
- 生成、验证、性能数据分别落盘。
- 一轮一优化点，keep/discard，连续失败 diagnosis。
- UB、批量 Token/Block、流水、dtype、离散访存作为优化知识库。
- 泛用 kernel + 专用 kernel + 安全回退的路由结构。

### 应参数化而非照抄

- Vector/Cube 核数。
- UB 容量、85 KB 双缓冲预算和对齐粒度。
- `grid=(num_core,)`、`grid <= core count` 等规则。
- `care_padding=False` 和去除 `other`。
- `torch.compile` 是否支持。
- warmup/repeat 数、0.8/0.3 性能阈值、0.1 ms 路由阈值。
- `pytest`/`msprof` 的文件名和命令模板。
- 精度容差。

### 应补充

- 版本/SoC 能力快照。
- 性能噪声与 A/B 交替测量。
- 参考失败、环境失败、编译失败、精度失败、性能退化的独立分类。
- 对尾块 identity、NaN/Inf、非连续输入和越界写的测试。
- GPU 性能只作参考，NPU 基线作为优化验收对象。
- 源码 + 编译选项 + 环境 + case 集合的统一 manifest。

## 8. 如果进一步封装成 Skill

建议用一个主编排 Skill 加四类可复用资源，不把 400 多行知识全塞进单一 `SKILL.md`：

```text
ascend-triton-operator-workflow/
├── SKILL.md                         # 触发条件、阶段机、强制闸门、路由规则
├── references/
│   ├── semantic-review.md           # load/store/mask/index 审计
│   ├── hardware-model.md            # 运行时探测优先，按 SoC/版本分表
│   ├── optimization-playbook.md      # 本文第 4、5 节的假设库
│   ├── correctness-policy.md         # case、容差、NaN/Inf、越界策略
│   └── failure-taxonomy.md           # ENV/REF/COMPILE/RUNTIME/PRECISION/PERF
├── scripts/
│   ├── detect_environment.py
│   ├── validate_triton_impl.py
│   ├── verify.py
│   ├── benchmark.py
│   └── update_manifest.py
├── schemas/
│   ├── task.schema.json
│   ├── verify.schema.json
│   └── run-manifest.schema.json
└── templates/
    ├── semantic-report.md
    └── final-report.md
```

主 Skill 只决定“现在处于哪一阶段、必须读取哪份 reference、必须产出什么”；硬件知识、故障签名和优化技巧按需加载。这样既保留 AscendOpGenAgent 的强状态机，又避免一个迁移 Skill 同时承担环境、语义、生成、调试和性能优化。

建议的路由边界：

- 源 kernel 尚未在 NPU 正确运行：走 migration/implementation 路径。
- 精度已全过但性能不足：走 profiling/optimization 路径。
- 已缩小为某个具体 Triton/torch_npu/ACLNN 调用的 dtype、shape、layout 故障：交给 operator-debug 类 Skill。
- 需要真实 NPU 执行：先同步确定的本地代码状态，再远端验证；本地 PC 不尝试运行 `torch_npu` 测试。
- 每次执行均生成 manifest；文档知识不能替代目标机器证据。

## 9. 来源与适用边界

主要材料：

- [simple-vector-triton-gpu-to-npu](https://gitcode.com/Ascend/agent-skills/blob/master/community/Op/simple-vector-triton-gpu-to-npu/SKILL.md)
- [vector-triton-ascend-ops-optimizer](https://gitcode.com/Ascend/agent-skills/blob/master/community/Op/vector-triton-ascend-ops-optimizer/SKILL.md)
- [AscendOpGenAgent](https://github.com/Just-it/AscendOpGenAgent)
- [AscendOpGenAgent AutoResearch](https://github.com/Just-it/AscendOpGenAgent/blob/main/autoresearch/AUTORESEARCH.md)

本次分析读取的 Git 快照：`Ascend/agent-skills@155ac37bd169ddb89479af528297cfb2237400aa`、
`Just-it/AscendOpGenAgent@595ce700122e1fea46d5bdc0949578b130ba404f`。

用于交叉核验硬件与软件栈表述的官方材料：

- [昇腾与 GPU 的开发差异](https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/migration_guide/architecture_difference.html)
- [NPU 高性能编程指南](https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/migration_guide/performance_guidelines.html)
- [Triton-Ascend 性能分析方法](https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/debug_guide/profiling.html)
- [Triton-Ascend 架构设计与核心特性](https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/architecture_design_and_core_features.html)

本文中的 192 KB、核数量、对齐、dtype 支持和 compiler option 都不是跨芯片/跨版本保证。实际执行前必须以目标环境的官方文档、运行时探测、编译结果和 profiler 证据为准。
