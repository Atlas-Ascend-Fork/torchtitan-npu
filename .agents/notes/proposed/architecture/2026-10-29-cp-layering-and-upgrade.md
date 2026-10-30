# Agent Note: CP 代码的层位，与它在 v0.4.0 上的去向

Status: proposed（阶段 1–6 已落地并验证；pin 升级与 CP 类替换未开始）
Created: 2026-10-29
Last updated: 2026-10-29
Tracking: 本仓重构（PR !1121）；前移依据为上游 #4617 / #4897 / #4116；升级目标 v0.4.0

## Context

本仓是 TorchTitan 的 NPU 插件仓。上游没有的模型归档在 `models/<model>/`，而 `models/` 的职责按
[架构边界](../../../../docs/architecture.md) 是"模型定义、配置、**并行化**与 checkpoint 适配"。曾经的
`models/common/` 收了 10 个文件、四类东西：CP 的搬运与布局、跨栈共用的 `KernelFrame`、模型构件
（`SharedLinear`、`HalfRotation`）、以及 NPU 运行策略（`compile_policy`、`muon`）。它不是"公共层没收
住"，而是"一个模型家族被拆成两个包，共享部分无处安放"。

上游的 CP 分成五层，这是判断"该放哪"的唯一依据：

| 层 | 上游位置 | 回答的问题 |
| --- | --- | --- |
| 机制 | `torchtitan/distributed/context_parallel.py` | 怎么切、怎么重排、怎么 gather |
| 合约 | `torchtitan/models/common/attention/attention.py` | attention 与框架的接口 |
| 执行 | `torchtitan/models/common/attention/cp_*.py` | 某个 attention family 在 CP 下怎么跑（`cp_X.py` 与 `X.py` 并列） |
| 安装 | `torchtitan/config/transform/context_parallel.py` | 谁把合约层换成执行层 |
| 模型 | `torchtitan/models/<model>/` | 模型自己的定义 |

`patches/` 的规则是另一条轴：它只装**上游已合入、当前 pin 尚未带**的代码前移，并可按 pin 的接口做
适配。没有已合入上游 PR 的自有机制不进 `patches/`。

本仓 CP 的整套机器（`cp_shard` 补丁、`positions` 入参、metadata 上的 `prepare_cp_metadata`、restore 与
carry 的 helper）存在的唯一原因是 pin 早于 #4617 / #4897。v0.4.0（2026-10-28）已带这两条，且只落后
`main` 45 个提交——`v0.3.0 → v0.4.0` 是 689 个提交，但它是一个发布版本，不是移动靶。

## Decision

1. **解散 `models/common/`，按上游的层归位**：
   - CP 执行（restore、`CPMetadata`、ratio-4 carry）→ `models/deepseek_v4/cp_attention.py` 与
     `cp_compressor.py`，各自与 `attention.py`、`compressor.py` 并列；
   - CP 布局（`DocChunkCPLoadBalancer` 与置换数学）→ `models/deepseek_v4/cp_layout.py`；
   - 模型构件（`KernelFrame`、packing、linear、rope、compile_policy）→ `models/deepseek_v4/`；
     Muon 的 `ComputeLayout` 构造只有一处用到，内联进 V4.1 的 registry（V4 本来就是内联写法）；
   - `document_intervals` 是唯一跨家族文件：V4 保留一份，V3.2 内联孪生（两个模型包不得互相依赖）。
2. **布局与机制分家**：CP shard 补丁持有机制——上游 #4617 的 `ContextParallelLoadBalancer(Configurable,
   ABC)`、布局名注册表、`positions` → 边界的推导、`shard_tensors`；模型包持有布局本身（`generate_permutation()`），
   并在 import 时 `register_layout`。`patches/` 里因此没有模型词汇、也不 import 模型包；依赖方向是模型 → 补丁。
3. **布局显式**：`--parallelism.context-parallel-load-balancer` 不再有隐式默认，V4/V4.1 的 registry 写死
   `doc_chunk`，错配在 metadata 边界报错并给出该传的 flag。
4. **未上游的 balancer 不进 `patches/`**：`cp_layout.py` 暂住模型目录；balancer 的上游 PR 合入后，其实现
   作为 backport 迁入 `patches/`，pin 升级后删除。
5. **升级目标是 v0.4.0**，不在中间提交停靠。CP 各件在 v0.4.0 上的去向、以及每个 patch 依据哪条上游 PR
   日落，见下两节。

## 与上游的差异

本仓 CP 与 v0.4.0 的差异只有七处，全部由 pin 的形状限制造成，且都带退出条件：

| 差异 | 上游 v0.4.0 | 本仓（0.3.0 + 本 PR） | 为什么 | 退出条件 |
| --- | --- | --- | --- | --- |
| balancer 构造入参 | `config.build(seq_len=…, attention_metadata=…)` | `Config().build(seq_len=…, cu_seqlens=…, cp_size=…)` | 0.3.0 的 `cp_shard` 拿不到 attention metadata，只能从整行 positions 现算边界 | pin 升级；`positions` 随之消失 |
| 布局解析 | config 系统解析 balancer 类 | 字符串 → 类的注册表 | 0.3.0 的 CLI 字段是字符串 | pin 升级，模型侧改注册 config 类 |
| `shard_tensors` 入参 | `(input_dict, *, input_shardings, permutation)` | `(cp_mesh, tensors, *, permutation, input_seq_dim)` | 0.3.0 的 decoder 交的是显式张量元组，没有 sharding 表可声明 | pin 升级，改用上游实现 |
| 支持矩阵形态 | balancer 类 → inner attention 类（config 校验期） | 栈 → 布局（metadata 边界） | 只有一个布局、一个融合栈族；且融合类在 `override/` 里，模型包不能引用 | pin 升级后并入上游校验 |
| CP 执行归属 | `CPInnerAttention` 子类 + `ContextParallelTransform` 安装 | `CPMetadata` 挂在共享 metadata 上，`Attention.forward` 调 `restore_key_stream` | 0.3.0 没有 `ModelConfigTransform` | 阶段 8 |
| `positions` 入参 | 无（边界从 attention metadata 读） | `cp_shard(..., positions=)` | 见第一行 | pin 升级 |
| `document_intervals` | 无此 API | V4 与 V3.2 各一份孪生 | 两个模型包不得互相依赖 | 不随 pin 消失；上游若合入同类工具再合并 |

## patch 的日落条件

| 文件 | 依据的上游 PR | v0.4.0 已含 | 日落动作 |
| --- | --- | --- | --- |
| `patches/.../models/common/decoder.py` | #4116（`preprocess_inputs` + `_cp_shard`）、#4897、#4617 | 是 | 删文件与其专用测试 |
| `patches/.../models/common/attention_metadata.py` | #4897（metadata 归后端）、#4617 | 是 | 删文件；`prepare_cp_metadata` 迁到 CP inner attention 类 |
| `patches/.../distributed/context_parallel.py` | #4617（`ContextParallelLoadBalancer` / `shard_tensors`） | 是 | 删文件；只留模型侧的注册（改 config 类） |
| `patches/.../distributed/varlen_cp.py` | #3430（尚未合入） | **否** | 等 #3430；若 v0.4.0 的 CP 已原生处理 varlen mask，则退化为 RETARGET 或删除（阶段 7 探测确认） |
| `models/deepseek_v4/cp_layout.py` | 自有，无上游 PR | — | 不随 pin 日落；阶段 9 的提案合入后才迁入 `patches/` |
| `models/deepseek_v4/cp_attention.py` / `cp_compressor.py` | 自有 | — | 阶段 8 迁成 `CPInnerAttention` / `CPCompressor` |

一句话：**v0.4.0 一到，三个 backport 立即日落；两个自有文件等阶段 8/9；`varlen_cp.py` 取决于 #3430。**

## 已执行的阶段

| 阶段 | 做什么 | 完成标志（均已满足） |
| --- | --- | --- |
| 1 | 删布局反查：`cp_layout` / `cp_layout_of` / `resolve_cp_layout` / `layout_needs_positions`；布局由 config 显式命名 | `grep cp_layout\|resolve_cp_layout` 只剩常量；九个示例脚本不改也能跑 |
| 2 | 共享 metadata 只留帧：删 `key_stream`，movement 收进 `CPMetadata`，两栈统一 `cp` 字段 | 共享 metadata 类无搬张量方法；`grep "def key_stream"` 为空 |
| 3 | 解散 `models/common/`（归位表见 Decision），`document_intervals` 内联孪生 | `models/` 下无 `common/` |
| 4 | 布局机制进补丁、本体进模型，模型 import 时注册 | CP 补丁里模型名出现 0 次；未注册名字明确报错 |
| 5 | 对齐 v0.4.0 形状：`ContextParallelLoadBalancer` + `shard_tensors`、删 `build_attention_masks` 过渡分支、两个融合栈的守卫合成一张支持矩阵 | 布局类与上游同名同形；`needs_positions` 消失 |
| 6 | 文档与 Notes 收敛（计划信息只存在于本目录） | `docs/` 下不再有过程性计划 |

每一步都遵守两条纪律：一步一个 commit、可回退；**没有逐位证据就不算完成**。

## 验证口径

单卡对拍的脚手架与环境交接文档由维护者本地保管，不入库；本 Note 只记**判据**，实测数值留在本地
的验证报告里。

| 机器 | 检查 | 判据 |
| --- | --- | --- |
| A5 单卡模拟器 | 同 degree：搬迁前后两棵树，CP=1 | logits 逐位相同 |
| A5 单卡模拟器 | 搬迁后树 CP=1 vs CP=2 | logits 逐位相同（doc-chunk 布局） |
| A5 单卡模拟器 | balancer 改写后 CP=2 vs CP=1 | logits 逐位相同 |
| arch22 真机 | 同 degree：搬迁前后两棵树，CP=2 | step 1 的 `loss` 与 `grad_norm` 打印值相同 |

跨 degree 的逐位判据只在单卡模拟器上取；arch22 上跨 degree 的逐位既有不成立。

## 兼容性变化

- **布局必须显式**：`--parallelism.context-parallel-load-balancer doc_chunk`；缺省不再从 metadata 反查，示例脚本与集成测试入口（`tests/integration_tests/deepseek_v4.py`、`deepseek_v4_1.py`、`run_engram_cp.sh`）已显式传。
- `DocChunkLayout` → `DocChunkCPLoadBalancer`；构造 `(seq_len, cu_seqlens, cp_size)`。
- `models/common/` 下所有模块路径变化（仓库内调用点、脚本与测试已同步）。

## Migration plan

1. **已完成（阶段 1–6）**：见上表。
2. **阶段 7**：pin 升级探测——新 worktree 指向 v0.4.0，只跑 CPU UT，产出"哪些 patch 失效"的清单（一天
   时间盒，超时先报告卡点）。
3. **阶段 8**：`CPCompressor` / `CPCompressedSparseInnerAttention` 吸收 restore 与 carry，由
   `ModelConfigTransform` 安装；`positions` 与 metadata 上的 CP 钩子随 pin 消失。
4. **阶段 9**：上游 PR——布局 balancer（含注册机制）、`seqused` 契约、几何 key 的 metadata 归属。

## Acceptance criteria

- `torchtitan_npu/models/` 下不存在 `common/`，且 `patches/` 的 CP 文件不出现模型名（已达成）。
- 每次搬迁：同 degree 逐位不变（单卡模拟器）+ arch22 真机同 degree 的 step 1 相同（已达成）。
- 阶段 8 后：`Attention.forward` 与共享 metadata 都不含 CP 分支。
- 阶段 7 后：`patches/` 的 CP 三件按清单删除或收窄。

## Alternatives considered

- **先升 pin 再整理**：0.3.0 → v0.4.0 是 689 个提交，代价远大于本重构；而本重构的形状与 v0.4.0 一致，
  升级时删补丁即可，不需要重写上层代码。
- **把 CP 执行直接做成 `CPInnerAttention` 类**：0.3.0 没有 `ModelConfigTransform` 来安装它，硬做会造出
  第二套安装机制，比现状更远。推迟到阶段 8。

## Risks

- 注册依赖"模型包先于 shard 被 import"；被绕过时以明确报错暴露，不做兜底猜测。
- CP 的数值判据在 arch22 上只到同 degree（跨 degree 的逐位既有不成立）；v0.4.0 的内核路径变化需逐 PR 验证。
- `cp_layout.py` 的临时位置若长期不迁，会重新长成"模型目录里的通用机制"——阶段 9 是它的退出条件。
- `spmd_types` 随 pin 变化（0.3.0 依赖 0.2.3），`--parallelism.spmd-backend` 的取值与行为要同步核对。
