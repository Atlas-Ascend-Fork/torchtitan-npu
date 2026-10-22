# DeepSeek-V4 TND 适配

DeepSeek-V4（DSV4）在模型层使用文档打包（varlen）的 `[B, S, ...]` 接口；压缩 KV 以容器网格 `[B, S//r, D]` 表达，仅在进入 AscendC 融合内核时转换为 TND。本文说明当前实现、启用方式和已知边界。

## 启用方式

融合路径要求完整的 Ascend NPU 运行环境，以及与 CANN 版本匹配的 `cann_ops_transformer`。推荐使用仓内脚本：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```

DeepSeek-V4 的示例 wrapper 有两种可切换的 recipe（`USE_GOLDEN` 选择；`--override.imports` 是显式覆盖入口）。`scripts/run_train.sh` 只负责通用启动和参数透传，不定义这些 recipe：

```text
USE_GOLDEN=1  # wrapper 里的参考组合（模型默认内核 = attn-gym gather_attn）：
              #   override.common.rope.workaround
              #   (the MoE is the normal clamped bf16 path on both sides)
USE_GOLDEN=0  # wrapper 里的默认测试组合：
              #   override.common.rms_norm.asc
              #   override.common.rope.asc_complex
              #   override.deepseek_v4.sparse_attn.asc（并带 geometry，见 fused_ops 指南）
```

参考路径（模型默认内核：Attention Gym 的 `gather_attn`，eager 逐文档、FP32）用于数值校验：
参考路径与 patched transformers 做逐 step 容差对比，AscendC 融合内核与参考路径在容差内一致
这些组合定义在 `examples/deepseek_v4/*.sh` 中；通用 launcher `scripts/run_train.sh` 只接收并透传 `--override.imports`，不会自动注入 DeepSeek-V4 override 组。

## 数据流

```text
模型张量 [B, S, ...]
        │
        ├─ positions + VarlenMetadata
        │      └─ DeepSeekV4Metadata（attention backend 的 config 每 microbatch 构建一次公共契约；
        │         AscendC 记录由 AscCompressedSparseInnerAttention.Config 一并注入）
        │
        ├─ Compressor：按统一契约（gather_indices/block_positions/first_indices）压缩 → 池化键流；CP 下 Compressor 内部 borrow 原始行（borrow_x），调用侧按 plan 的 kept_indices/out_width 装入容器网格（pack_container）
        │
        ├─ 默认内核（CompressedSparseInnerAttention）
        │     gather_attn(q, swa_k, cmp_k 容器, 选择) + sink；选择由 LightningIndexer 给出
        │
        └─ AscendC 融合内核（AscCompressedSparseInnerAttention）
             q/swa_k flatten → [T, N, D] / [T, 1, D]
             cmp_k 容器 → TND：container_flat[:n_blocks]（B=1 下等价于恒等映射）
             LI / SMLA / SMLAG / SLIG
             output [T, N, D] → [B, S, N, D]
```

TND 只存在于 NPU 融合内核的局部计算中，不会成为模型公共布局。

| 符号 | 含义 |
| --- | --- |
| `B` | 模型输入的 batch size |
| `S` | 模型输入的序列长度 |
| `T` | TND 下的 token 总数，等于 `B * S` |
| `T_cmp` | 某个压缩比下的完整压缩块总数 |
| `N` | 注意力头数 |
| `D` | 单头维度 |
| `r` | 压缩比；当前 DSV4 配置使用 1、4 和 128 |

## 压缩布局：DeepSeekV4Metadata

[`metadata.py`](../../torchtitan_npu/models/deepseek_v4/metadata.py) 定义公共注意力契约 `DeepSeekV4Metadata`。每个 microbatch 由 attention backend 自己的 config 构建一次（upstream PR #4897 的接缝）：`CompressedSparseInnerAttention.Config.build_attention_metadata` 从 `positions` 调出 `VarlenMetadata`，再用 `build_compressed_varlen_metadata` 生成这个几何的契约；融合路径的 `AscCompressedSparseInnerAttention.Config` 在其上按 `AicpuGeometry` 生成 AscendC 记录（LI record 也由它在同一入口构建）。参考契约的 `prepare_cp_metadata` 直接拒绝 CP；融合路径的 hook 调 `shard_compressed_varlen_metadata`（`build_cp_plan`）生成 rank-local varlen、压缩计划与窗口计划，再按 rank-local 行重建记录。`Decoder` 每个 microbatch 为每个几何构建一次，供所有同几何 DSA 层共享：

- `varlen`：`VarlenMetadata` 或 CP 下的 `CPVarlenMetadata`，`cu_seq_q` 是序列边界的唯一权威来源。`_build_plan` 只构建普通流计划，并明确要求 `cu_seq_q == cu_seq_k`；CP 的 rank-local 计划由 `build_cp_plan` 构建，不会把 CP metadata 传给 `_build_plan`。
- `batch_size` / `seq_len`：容器网格形状。DSV4 打包场景使用 `local_batch_size == 1`，因此 `batch_size == 1`、`seq_len` 等于总 token 数（`cu_seq_q[-1]`），元数据完全由 varlen 流推导，不依赖 positions。
- `plan`：这个 entry 自己的压缩比（1、4、128 之一）对应一个 `CompressedBlockLayout`：
  - `cu_seqlens_cmp_k`、`block_remainder`：打包压缩块流边界与每段尾部余数（对齐后 plain plan 恒 0，CP 计划只在 slab 截断文档时非 0），直接喂给 AscendC 算子；
  - `gather_indices`、`block_positions`、`first_indices`：统一的 Compressor 契约（块 token gather、文档内块起点 RoPE、文档/段首块位置——重叠 borrow 掩码），CP 与非 CP 以同一字段名提供。非 CP 字段由 `_build_plan` 生成；CP 下的 plan 由 `build_cp_plan` 生成，并额外携带 dispatcher 所需字段。
- `asc`：只有融合 entry（`AscV4Metadata`）才有，按 kernel 分组保存 `sparse_attn.fwd/bwd` 与 `lightning_indexer.fwd/bwd`；模型目录的 `DeepSeekV4Metadata` 不携带这些张量。

一个 batch 里可以打包多条序列，所以 `B` 与序列条数无关。例如：

```text
positions  = [[0, 1, 2, 0, 1]]
B          = 1
cu_seqlens = [0, 3, 5]   # 一行里打包了 2 条序列
```

每条序列必须从位置 0 开始并连续递增。该约束保证压缩、因果 mask 和稀疏索引不会跨越序列边界。

## 压缩与索引规则

文档在进入行拼接前按 `lcm(compress_ratios)`（V4 为 `128`）对齐：dataloader 在 EOS 之后补 pad token
（label 为 `-100`，位置继续累加），因此每篇文档都是所有压缩比的整数倍，尾部余数恒为 0，压缩块不会跨文档。
对齐由 `dataloader.per_doc_alignment` 驱动，registry 从 `compress_ratios` 推出该值（`AlignedHuggingfaceDataloader`），
与压缩器、内核看到的块网格同源。布局构建器把这条契约当作前提：`build_kernel_layout` 对非空且按
`lcm(compress_ratios)` 对齐的流，直接把这一行按 ratio 除出计划（边界 `cu_seq_q // ratio`、gather 是恒等、
块位置是一条等差、余数恒 0），不再逐文档扫描、也没有「尾部不足一块」的分支；破约的流（含空文档或长度不是对齐
倍数）在构建处直接报错。CP 下仍以 `seq_len / cp_size` 切 slab，块跨 rank 的补块逻辑属于 CP 路径自己的事。

同一对齐值也约束后续两条路径：`seq_len` 必须是 `per_doc_alignment` 的倍数（loader 在构建时校验），
SFT 的 chat 数据集按同一个值补 pad（`chat_dataloader.sft` 把对齐值透传到 `DeepSeekV4ChatDataLoader`）。

补出的 pad 只影响它自己的位置：压缩槽位只在所属块的**最后一个** token 上可见，而被 pad 补完的尾块最后一个位置必然是 pad，
因此真实 token 因果上看不到任何含 pad 的压缩键（ratio-4 的选择器同样只会从因果可达的整块里挑），pad 的 label 又是
`IGNORE_INDEX`，所以对齐不改变单篇文档真实 token 的前向与 loss。打包密度变化（pad 占掉 row 容量）才会改变同一 step
见到的文档集合——这是 ST 基线平移的原因，不是数值路径变化；`test_padding_a_document_tail_is_invisible_to_its_real_tokens`
把这条不变量钉在测试里（只改 pad 内容，真实位置的 pool、输出与 CE 逐位不变，pad 位置输出改变且 CE 贡献为 0）。

[`compressor.py`](../../torchtitan_npu/models/deepseek_v4/compressor.py) 只处理完整压缩块，且**绝不跨文档压缩**。文档先按 `lcm(compress_ratios)` 对齐（见上），因此对每篇文档 `compressed_len = L / r` 是整数、`residual = 0`；`block_remainder`（内核的 `cmp_residual_k`）仍按内核契约传入，只是 plain plan 里恒为 0。例如 `cu_seq_q = [0, 128, 384]`（两条文档 128 与 256），`r = 4`：

```text
cu_seqlens_cmp_k = [0, 32, 96]   # 压缩块：32 + 64
block_remainder  = [0, 0]        # 对齐后恒 0；CP 计划的非零值来自 slab 截断
gather_indices   = [0..383]      # 恒等 gather（[96, 4]）
block_positions  = [0, 4, ..., 124, 0, 4, ..., 252]  # 文档内相对位置
overlap_valid    = [F, T x 31, F, T x 63]  # 文档起始块没有前驱
```

非零 `block_remainder` 只出现在 CP 计划里：rank 的 slab 截断文档时，段的 K 前缀可能停在块中间。

- `r=1`：不生成压缩 KV，仅执行原始 KV 的滑窗注意力。公共 `plan` 不包含压缩块，融合 entry 仍生成对应的 `asc` 记录，融合算子使用 `has_cmp_kv=False`。
- `r=4`（CSA）：启用重叠压缩和 LightningIndexer；前驱块必须属于同一文档（`overlap_valid`）。
- `r=128`（HCA）：生成连续压缩 KV，不执行 LightningIndexer TopK。

## 参考路径（默认内核）

模型目录默认内核是 `CompressedSparseInnerAttention`：它只读公共 metadata 的 packed 偏移（`cu_seq_q` 与 `plan.cu_seqlens_cmp_k`），用 Attention Gym 的 `gather_attn` 在 `cat([swa_k, cmp_k 容器])` 上做滑窗 + sink，并按 ratio 取选择——`r=4` 走 `LightningIndexer` 给出的逐文档 TopK，`r=128` 直接取本查询本自然段内因果可达的全部压缩槽。选择与 `gather_attn` 都按“文档内”下标寻址（见 `models/common/document_intervals.py`），因此不再需要文档感知的 BlockMask，也不再需要额外的 metadata 扩展来预计算参考 tier。

默认内核就是唯一的参考路径，不再有独立的 golden override：`test_dsv4.py` 以形状/有限值和 pool 梯度的可复现性守护它，融合内核与它做容差对照。

参考组合不训练 indexer：`r=4` 层的 `LightningIndexer` 只提供 TopK 选择（不可导），它的 KL 蒸馏目标由融合组合的 `asc_li` / SLIG 承担，参考核心本身不产生 indexer loss。因此用参考组合与融合组合对比 `loss` / `grad_norm` 时，indexer 相关参数在参考组合下没有梯度属于预期；`test_dsv4.py` 与 `dsv4_golden_*` 用例只校验注意力数值，不覆盖 indexer 的训练目标。

因此参考组合目前只能用 AdamW：`DistMuon` 在 step 前要求每个配置的参数都有梯度，会以 `missing gradients: layers.<n>.attention.indexer.*` 报错（`USE_GOLDEN=1` 搭配示例里的 Muon 参数即属此情况）。

## AscendC 融合路径

[`sparse_attn/ascendc.py`](../../torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py) 中的 `AscCompressedSparseInnerAttention` 执行以下步骤：

1. 校验模型输入与 `DeepSeekV4Metadata` 一致，直接读这一层的 `plan`。
2. 把 query、原始 KV 转成 TND；压缩 KV 与 Indexer key 取容器网格头部 `n_blocks` 槽位转成打包流。
3. 调用 LightningIndexer 生成序列内局部 TopK（使用 `asc.lightning_indexer.fwd`）。
4. 调用 SparseFlashMLA；`_SparseFlashMLATND` 在反向中调用 SMLAG 与 SLIG（使用 `asc.sparse_attn.bwd`、`asc.lightning_indexer.bwd`）。SLIG 的 Indexer KL 梯度按 `indexer_loss_coeff` 缩放；`sparse_attn.asc` 和 `sparse_attn.pypto` 公开 override 入口的有效默认值均为 `1.0`，显式传入 `{"indexer_loss_coeff": 0.0}` 可关闭该梯度。LI 损失值累积到 `_indexer_loss_acc` 缓冲；Indexer auxiliary loss 已从模型目录移除，随上游机制落地后另行设计。

`cann_ops_transformer` 包负责内核的算子注册（schema、meta/fake 实现），本仓只在融合路径内做前反向桥接，不再自建 `torch.library` 封装。

| 参数 | 来源 |
| --- | --- |
| `cu_seqlens_q` / `cu_seqlens_ori_kv` | `varlen.cu_seq_q`（无 CP 时相同） |
| `cu_seqlens_cmp_kv` | `plan.cu_seqlens_cmp_k` |
| `cmp_residual_kv` | `plan.block_remainder` |
| `layout_q` / `layout_kv` | 固定为 `TND` |
| 四个 `*_metadata` 张量 | metadata 的 `asc`（融合 backend 的 config 在 metadata 边界预计算；CP 下按 rank-local frame 重建） |

## 当前边界

| 项目 | 当前状态 |
| --- | --- |
| `positions` | 必须提供 `[B, S]`，且每条序列从 0 连续递增 |
| Padding | varlen 流必须覆盖本 rank 的全部 `B * S` 个 token，metadata 不表达 padding token |
| Context Parallel | 支持（AscendC 融合路径，CLI 指定 `NGPU=2 bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh --parallelism.context-parallel-degree 2 --parallelism.spmd-backend spmd_types`）；CP 与非 CP 共用统一的 `CompressedBlockLayout`（两部分：统一契约 + dispatcher 字段），CP 全部逻辑在 `token_dispatcher.py`（纯全局上下文 plan + `BaseEPTokenDispatcher` 形态的 dispatcher），`cp_plan` 只提供 `borrow_swa`/`borrow_x`/`pack_container` |
| Tensor Parallel | DSA 路径固定 TP=1（indexer score 对 head 维求和），TP 方案待 CP 之后另行设计 |
| 空压缩流 | doc-packing 场景要求每条序列至少产生一个完整压缩块；融合路径在 metadata 边界对 `ratio > 1` 且无完整块直接报错（AscendC CSA/HCA 不接受空 `cmp_kv`） |
| 运行环境 | 需要 `torch_npu`、CANN、HCCL、Ascend NPU，以及匹配的 `cann_ops_transformer` |
