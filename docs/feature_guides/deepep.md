# DeepEP - NPU MoE dispatch/combine

## 背景与挑战

MoE 的专家并行（EP）把 token 按路由结果发送到持有目标专家的 rank，再把各专家的输出
还原到原 token 顺序。dispatch 和 combine 同时承担数据重排、跨 rank 通信以及 routing
score 的传递，任一处的张量顺序或 metadata 不一致都可能导致输出错位。

仓库已有标准 all-to-all 路径。dispatch 先调用 `npu_moe_token_permute` 重排 token，交换
token 计数并执行 token all-to-all，再调用 `npu_moe_re_routing` 将收到的 rank-major 布局
重排为 expert-major。EP>1 时，combine 依次调用两次 `npu_moe_token_unpermute`，中间执行
一次返回方向的 all-to-all：第一次把 expert-major 输出恢复为通信前的 rank-major 布局，第二次把
结果恢复到原 token 顺序并完成 top-k 聚合。它在语义上
完整，中小 EP 或通信阶段不构成瓶颈时足够使用，也是默认路径。DeepEP 要解决的不是
「有没有 dispatch/combine」，而是标准路径在每层都要付出的固定通信与 host 侧准备成本：

| 维度 | 标准 all-to-all | DeepEP / ElasticBuffer |
| --- | --- | --- |
| token 计数与 split | 每层先做 count all-to-all，再阻塞 D2H 得到 `input_splits`/`output_splits` 两个 host 列表；据此执行 token all-to-all | 由 buffer 协议完成计数；host 只等待接收 token 数，不生成这两个 split 列表 |
| forward 通信（EP>1） | dispatch 执行 count 和 token all-to-all；开启 score absorption 时再单独执行 score all-to-all。combine 在两次 `npu_moe_token_unpermute` 之间执行一次 all-to-all，将专家输出传回 token 所在 rank。关闭 score absorption 时在该 rank 使用本地保留的 score 加权，无需发送 score | 一次 `deepep::dispatch` 将 token 与 score 发送到专家所在 rank，一次 `deepep::combine` 将专家输出传回 token 所在 rank |
| backward 通信（EP>1） | combine 的梯度经 all-to-all 传向专家所在 rank；dispatch 的 token 梯度经 all-to-all 传回 token 所在 rank。开启 score absorption 且 score 需要求导时，再经单独的 score 梯度 all-to-all 传回；关闭时 score 梯度在 token 所在 rank 本地计算，无需单独的 score 梯度通信。count 不参与求导 | `deepep::combine_backward` 调用 cached dispatch 向专家所在 rank 发送梯度；`deepep::dispatch_backward` 调用 cached combine 传回 token 梯度及所需的 score 梯度。两步仍执行跨 rank 梯度通信；cached handle 复用路由与接收计数等 metadata |
| 接收容量 | 根据本层路由结果动态确定接收 token 数，运行时构造接收 token tensor，并生成对应的 `input_splits`/`output_splits` 列表 | 用 `num_max_tokens_per_rank` 约定每 rank 输入 token 上限，作为 `ElasticBuffer` 申请 buffer 空间的入参；本仓不另行分配 HCCL 通信 buffer |
| 默认状态 | moe_comm_backend="standard" 的默认路径 | 需要 `moe_comm_backend="deepep"` 与 `asc_deepep` |

这些成本在大 EP、token 数大、每层都要重复 count/split 准备和 score 通信时累积。DeepEP
把计数、score 传递和 token 交换收进一个带容量上限的 `ElasticBuffer`：dispatch 直接带 topk
scores 发出，combine 只做 reduction，backward 复用同一 handle 中的 metadata 执行梯度通信。代价是需要给出
`num_max_tokens_per_rank` 上限、依赖 CANN `ElasticBuffer`，并接受 buffer 生命周期和容量
配置的约束；HCCL 通信 buffer 实际由 `ElasticBuffer` 内部底层算子按这组参数计算并申请，本仓只传
所需参数。

上游 torchtitan 的 DeepEP dispatcher 通过 CUDA DeepEP
包创建通信 buffer，这条初始化路径不能在 Ascend NPU 上直接复用。NPU 适配需要以 CANN
`cann_ops_transformer.ElasticBuffer` 替换 buffer 和通信算子，同时继续提供上游
`DeepEPTokenDispatcher` 约定的 dispatch/combine metadata，才能接入现有 MoE 和反向路径。

## 特性概述

DeepEP 是 MoE 专家并行（EP）训练中的 token dispatch/combine 路径。模型配置选择
`moe_comm_backend="deepep"` 后，`torchtitan_npu.override.common.token_dispatcher.asc_deepep`
把上游 `DeepEPTokenDispatcher` 替换为基于 CANN `cann_ops_transformer.ElasticBuffer` 的
`AscDeepEPTokenDispatcher`，由 `deepep::dispatch` / `deepep::combine` 自定义算子完成 NPU 侧
token 收发交换。该路径只在模型配置选择 DeepEP 并启用 override 时生效，且要求
`expert_parallel_degree > 1`；当前只有 A5 入口默认选择它，A3 入口与 graph_trainer A5 入口都
不使用。

- **CANN ElasticBuffer 实现**：`AscDeepEPTokenDispatcher` 使用
  `cann_ops_transformer.ElasticBuffer` 完成 compact（非缓存、可反向）的训练 dispatch/combine。
- **Pre-W2 score absorption**：routed experts 在 SwiGLU 之后、W2 之前乘 routing score；
  combine 不再重复加权。
- **专用 EP 通信域**：`get_deepep_group()` 创建独立 HCCL group，与其他通信域隔离。
- **训练 recipe**：DeepSeek-V4 的 `deepseek_v4_flash_deepep`、
  `deepseek_v4_flash_43layers_16experts_deepep`、
  `deepseek_v4_pro_61layers_32experts_deepep`、
  `deepseek_v4_pro_61layers_12experts_deepep`；DeepSeek-V4.1 的
  `deepseek_v4_1_flash_deepep`、`deepseek_v4_1_flash_text_deepep`、
  `deepseek_v4_1_flash_40layers_24experts_multimodal_deepep`。

## 启用方式

DeepEP 由模型配置和 override 两处共同选择：

| 配置 | 作用 |
| --- | --- |
| `moe_comm_backend="deepep"` | 让模型工厂生成 `DeepEPTokenDispatcher.Config`；recipe 已内置 |
| `asc_deepep` override | 把 dispatcher 换成 `AscDeepEPTokenDispatcher`；未启用时走上游 CUDA DeepEP，NPU 上不可用 |

DeepEP 要求 `expert_parallel_degree > 1`。`hidden_dim` 与 `num_max_tokens_per_rank` 由配置
阶段的 `update_from_config()` 按固定训练参数填写（公式与校验见下文「通信域与容量」）；绕过该
阶段手工构造配置时必须自行提供。`seq_len` 必须能被
`context_parallel_degree * tensor_parallel_degree` 整除。

当前默认选择 DeepEP 的入口是 A5 脚本：DeepSeek-V4 的 `deepseek_v4_flash_cpt_4k_a5.sh`、
`debug/deepseek_v4_flash_8p_cpt_4k_a5.sh`、`debug/deepseek_v4_flash_8p_cpt_4k_a5_hif8.sh`、
`debug/deepseek_v4_pro_32p_cpt_4k_a5.sh`、
`debug/deepseek_v4_pro_8p_cpt_4k_a5_cpuoffload.sh`，DeepSeek-V4.1 的
`deepseek_v4_1_flash_cpt_4k_a5.sh`、`deepseek_v4_1_flash_cpt_1m_a5.sh`、
`debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh`，以及它们的 qat / sft / 1M 派生入口。这些脚本
默认把 `CONFIG` 指向对应的 `*_deepep` recipe，并在 `CLI_OVERRIDES` 中带上 `asc_deepep`：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh
```

`USE_GOLDEN=1` 时这些入口回到标准 dispatcher 的 reference 组合：`CONFIG` 换回非 deepep
recipe，`CLI_OVERRIDES` 不再包含 `asc_deepep`。

graph_trainer A5 入口暂不开启 DeepEP：它们用 `graph_trainer_*` 配置，并在自己的
`CLI_OVERRIDES` 里显式选择标准 all-to-all dispatcher 的 `token_dispatcher.asc`。LoRA 只增加
adapter 参数，与 dispatcher 的选择无关。

## 架构

```text
model config (moe_comm_backend="deepep")
  → DeepEPTokenDispatcher.Config
      ├─ absorb_router_scores（模型工厂默认 True）
      └─ update_from_config(): hidden_dim / num_max_tokens_per_rank / EP>1 校验
  → asc_deepep override
  → AscDeepEPTokenDispatcher
      ├─ get_deepep_group(ep_mesh) ── 独立 HCCL group（缓存）
      ├─ ElasticBuffer(group, num_max_tokens_per_rank=..., hidden=..., num_topk=...)
      ├─ dispatch → deepep::dispatch ── hidden_states / recv_scores / handle
      ├─ routed experts ── SwiGLU 后、W2 前乘 score（absorb=True）
      └─ combine → deepep::combine ── 纯 reduction
```

`AscDeepEPTokenDispatcher` 不走上游 `DeepEPTokenDispatcher.__init__`（该构造函数会导入 CUDA
DeepEP 包），而是直接初始化 `BaseEPTokenDispatcher` 并接入 CANN buffer；`deepep::*` 的反向
复用 ElasticBuffer 的 cached dispatch/combine。

## 组件清单

| 模块 | 职责 |
| --- | --- |
| `override/common/token_dispatcher.py` | `AscDeepEPTokenDispatcher`、`ElasticBuffer` 生命周期、`asc_deepep` |
| `patches/torchtitan/models/common/token_dispatcher.py` | 为 DeepEP 配置补 `absorb_router_scores`；保留上游 metadata 契约 |
| `patches/torchtitan/models/common/moe.py` | `_RouterScoreAbsorbingRoutedExperts`、`HashMoE` |
| `ops/ascendc/deepep.py` | `deepep::dispatch/combine` 及反向、opaque handles、fake/meta |
| `extensions/torchtitan/ep_process_group.py` | `get_deepep_group`、专用 EP group、dispatcher wiring |
| `models/deepseek_v4/config_registry.py`、`models/deepseek_v4_1/config_registry.py` | DeepEP recipe |
| `examples/deepseek_v4/`、`examples/deepseek_v4_1/` 下的 A5 脚本 | 默认入口与 qat / sft / 1M 派生入口 |

## 数据路径与 score absorption

**dispatch**：`AscDeepEPTokenDispatcher.dispatch` 调 `torch.ops.deepep.dispatch`，输入
`x_TD`、`topk_idx`、`topk_weights`、专家数、本地专家数和 buffer handle。op 内使用
`ElasticBuffer.dispatch(..., do_cpu_sync=True)` 的 compact 非缓存路径：先在 host 上等待
接收 token 数，再分配接收 token tensor；返回 `hidden_states_RD`、`recv_scores`、每专家 token
数和 `DispatchHandle`。`DispatchState` 保存 `recv_scores`、`handle`，并在
`absorb_router_scores=True` 时把 `recv_scores` 作为 `routed_scores_R` 传给专家。

**score absorption**：当前模型工厂默认 `absorb_router_scores=True`，
`_RouterScoreAbsorbingRoutedExperts` 把 `routed_scores_R` 传给`_ClampGroupedExperts`；
后者在 SwiGLU 之后、W2 之前乘 score。`combine` 在吸收模式下只做纯 reduction（`topk_weights=None`），
不再重复加权；`recv_scores` 复用 dispatch 已收到的数据，不增加额外通信或拷贝。
`absorb_router_scores=False`则保留 combine 前按 `recv_scores` 加权的路径。

**反向路径**：`deepep::dispatch` 与 `deepep::combine` 都注册了 fake/meta 和 autograd，并复用
同一个 handle：combine 的反向实际调用 cached dispatch，dispatch 的反向实际调用 cached combine，
分别向专家所在 rank 发送梯度、向 token 所在 rank 传回 token 梯度及所需的 score 梯度。
cached handle 保存路由与接收计数等 metadata，梯度数据仍需跨 rank 通信；其中 cached dispatch
复用已知的接收计数，并设置 `do_cpu_sync=False`，跳过非缓存 dispatch 中等待接收计数的 host 同步。
`deepep::combine` 额外接收 `num_permuted_tokens_src`，它只用于反向恢复 routed 行数——combine
前向 kernel 不读取其值，这样反向不必从未保存的专家中间结果推导行数。

## 通信域与容量

`get_deepep_group(ep_mesh)` 为 DeepEP 创建独立 HCCL group：要求 EP>1，使用与源 EP group
相同的 ranks，`group_desc="asc_deepep"`、`use_local_synchronization=True`，并按源 group name
缓存。`AscDeepEPTokenDispatcher` 设置 `_uses_custom_ep_process_group=True`，避免通用 EP
wiring 在拿 buffer 之前替换掉这个 group。

`ElasticBuffer` 是进程级单例（classvar），由 `AscDeepEPTokenDispatcher.get_buffer_and_group()`
调用 `ElasticBuffer(deepep_group, num_max_tokens_per_rank=..., hidden=..., num_topk=...)` 构造
（`override/common/token_dispatcher.py`）。这四个参数都传给底层算子，用于计算并申请 buffer
空间：`hidden` 是 token 宽度、`num_topk` 是每 token 的 top-k、`num_max_tokens_per_rank` 是
每 rank 的输入 token 上限。本仓自身不另行分配通信 buffer；`HCCL_BUFFSIZE`/`hccl_buffer_size`
不是这条路径的容量入口，`get_moe_ep_ccl_buffer_size()` 的默认值也不代表实际窗口。细节见
[EP 通信域分离](ep_process_group.md)。

`num_max_tokens_per_rank` 由模型配置的 `update_from_config()` 调用上游
`update_ep_token_dispatcher_config()` 按固定训练参数换算：

```text
num_max_tokens_per_rank = local_batch_size * seq_len
                          // (context_parallel_degree * tensor_parallel_degree)
```

`hidden_dim` 取模型维度。`seq_len` 必须能被 CP×TP 整除，且要求
`expert_parallel_degree > 1`；配方显式给出该值时不得小于上式结果，否则配置阶段报错，留空则
按上式填写。例如 1M 入口（`MBS=1`、`seq_len=1048576`、CP=128、TP=1）得到 8192。

## 支持组合

| 组合 | 状态 | 说明 |
| --- | --- | --- |
| DP shard + EP | 已验证 | debug 脚本（8 卡、EP8、DP shard 8、CP/TP/PP=1）|
| CP > 1 | 已验证 | 配置阶段校验 `seq_len % (CP*TP)` |
| TP > 1 / PP > 1 | 未验证 | Muon 暂不支持 TP>1 或 PP>1 |
| torch.compile | 已验证 | ops 提供 fake/meta 与 autograd |
| `absorb_router_scores=false` | 已验证 | 保留 combine 前加权路径 |

## 与其他特性的关系

| 特性 | 关系 |
| --- | --- |
| EP 通信域分离 | DeepEP 使用 `get_deepep_group` 创建的独立 HCCL group；见 [EP 通信域分离](ep_process_group.md) |
| GraphTrainer EP overlap | GraphTrainer 的 EP chunk 调度，不走 DeepEP dispatch/combine；见 [GraphTrainer EP overlap](graph_trainer_ep_overlap.md) |
| torch.compile | DeepEP ops 提供 fake/meta；DeepEP 编译已验证 |

## 限制

- 需要 Ascend NPU、`torch_npu` 和 CANN `cann_ops_transformer`（`ElasticBuffer`/`EPHandle`）；
  缺失时导入失败。
- 要求 `expert_parallel_degree > 1`；`seq_len` 必须能被 CP×TP 整除；上限不足会在配置阶段
  报错。
- 只实现 compact 训练 dispatch/combine；不支持上游 expand / CUDA Graph inference 布局。
- 量化方向：CANN `ElasticBuffer.dispatch` 支持 FP8 token 加 scales；`ElasticBuffer.combine` 的输入与输出暂只接受 bf16 / fp16。
- `ElasticBuffer` 是进程级单例；同一进程内多个 DeepEP group/模型不能各自持有独立 buffer。
- compact 非缓存 dispatch 需要 host 等待接收 token 数，存在 host 同步开销。
- `absorb_router_scores` 当前默认 True，没有 recipe 级关闭开关。
- 只有 A5 入口默认选择 DeepEP；A3 与 graph_trainer A5 入口不走 DeepEP。
