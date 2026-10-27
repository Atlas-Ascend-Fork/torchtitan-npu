# DeepSeek-V4 系列上下文并行（CP）

DeepSeek-V4（DSV4）与 DeepSeek-V4.1（DSV4.1）共用同一套上下文并行语义：**查询按文档内部分成
`cp` 份，KV 与压缩 KV 保持整行**。融合内核在边界处把本地 slab 声明式 all-gather 成整行，再用
frame 的 `seqused = reach` 把可见范围限制在本 rank 的 chunk 之前。两栈共用
`models/common/query_exchange.py` 的 plan/exchange 与 `models/common/metadata.py` 的
`KernelFrame`；差异集中在压缩器的 CP 处理、内核如何锚定 mask，以及 frame 由哪一层 metadata 携带。

本文是 V4 与 V4.1 的合并指南（原 V4 CP 指南的内容已并入）。旧的 `build_cp_plan` /
`CPTokenDispatcher` / `WindowPlan` / `ExchangePlan` / 容器打包 / borrow-strip 路径均已删除。

## 1. 共同契约

### 1.1 坐标

设一行有文档 `d`、长度 `L_d`，CP 度 `R = cp_size`，rank `r`，slab 宽度 `C = N / R`
（`N` 是整行 token 数），chunk 宽度 `Q_d = L_d / R`：

```text
rank r 的输入 slab：        [r·C, (r+1)·C)
查询帧 q：                 每篇文档的第 r 个 chunk（query exchange 搬过去）
滑窗 KV 帧 swa_k：         整行；seqused = reach_d = (r+1)·Q_d
压缩 KV 帧 cmp_k：         整行按 ratio 折叠；seqused = reach_d // ratio，residual = reach_d % ratio
positions_q：              搬过来的查询自己的文档内位置
```

`q.cu_seqlens = cu_seq_q // R`（每篇文档 chunk 前缀和），`q.seqused = Q_d`；
`swa_k.cu_seqlens = cu_seq_q`，`cmp_k.cu_seqlens = cu_seq_k`（entry 自己按 ratio 折好的整行网格）。
因此 `seqused_ori_kv - seqused_q = r·Q_d` 正好是本 rank 在该文档内的 chunk 偏移，内核的滑窗与
压缩因果上限都锚在这个偏移上。

`cu_seq_q` / `cu_seq_k` 永远是**整行**网格：rank 化只发生在 frame 的 `cu_seqlens` / `seqused`
上，metadata 本身不随 CP 改变形状。

### 1.2 plan 与 exchange

`cp_plan(cu_seqlens, cp_size, rank, seq_len)`（`models/common/query_exchange.py`）只从全局文档
边界与本 rank 推导一次移动，零通信：

- `send_index` / `send_split`：本 rank slab 里每一行要去的 peer（按 peer 连续、peer 内按槽位升序）；
- `recv_split`：本 rank 从每个 peer 收到的行数；接收 buffer 已经是 held-row 顺序，不需要二次
  scatter；
- `positions_q`：搬过来的行自己的文档内位置（RoPE 在文档边界重启，不能用原 slab 的位置）。

`QueryExchange` 是执行这段移动的模块：`parallelize` 从框架拿到 CP mesh，前向
`permute` = `all_to_all_single(x[send_index], ...)`，`unpermute` 是它的逆（置换，所以 adjoint
就是逆，不累加）。没有 plan（`metadata.exchange is None`）时两者都是恒等，非 CP 路径因此不需要
分支，也不需要进程组。V4 与 V4.1 的 `Attention.parallelize` 都通过模块递归把 mesh 接到
`query_exchange` 上。

### 1.3 frame 与 metadata

一行由三个 `KernelFrame` 描述：`q`、`swa_k`、`cmp_k`（不池化的几何没有 `cmp_k`）。CP 下由融合
entry 的 `prepare_cp_metadata` 用 rank-local chunk 重建这三个 frame 以及按它们算出的 AI_CPU
记录；参考 entry 的 hook 显式拒绝 CP（它只有一条连续文档流的契约）。frame 的归属两栈不同：

| 栈 | 参考 metadata | CP 重建 |
| --- | --- | --- |
| V4 | `DeepSeekV4Metadata` 自带 `q`/`swa_k`/`cmp_k`（`build_v4_metadata` 用整行边界构造） | `AscV4Metadata.prepare_cp_metadata` → `shard_asc_v4_metadata` |
| V4.1 | `DeepSeekV41Metadata` 只有整行边界与位置；frame 由融合 entry `AscV41Metadata` 携带 | `AscV41Metadata.prepare_cp_metadata` → `shard_asc_v41_metadata` |

两栈的重建都从 `cp_shard_context()` 读 pre-shard `positions` 与 CP mesh，都拒绝
`--parallelism.context-parallel-load-balancer`（除 `None` 外的任何取值）：这些 frame 是每篇文档的
连续 slab，load balancer 的置换需要同时折进 chunk 网格、query exchange 与 engram halo，当前不支持。

## 2. 启用方式

```bash
# DeepSeek-V4
NGPU=2 \
  bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh \
  --parallelism.context-parallel-degree 2 \
  --parallelism.spmd-backend spmd_types \
  --dataloader.per_doc_alignment 256 \
  --training.steps 2
```

```bash
# DeepSeek-V4.1
NGPU=2 \
  bash examples/deepseek_v4_1/deepseek_v4_1_flash_1p_2k_a5.sh \
  --parallelism.context-parallel-degree 2 \
  --parallelism.spmd-backend spmd_types \
  --dataloader.per_doc_alignment 256
```

- `--parallelism.context-parallel-degree > 1` 启用 CP；必须配合
  `--parallelism.spmd-backend spmd_types`（声明式 `S(1) -> R` all-gather 的前提）。
- 只支持连续 slab 切片：`--parallelism.context-parallel-load-balancer None`（上游默认 `headtail`
  会被融合 entry 的 `prepare_cp_metadata` 拒绝）。
- 每篇文档按 `lcm(compress_ratios) × cp` 填充：V4 为 `128 × cp`（CP=2 即 256），V4.1 为其
  `compress_ratios` 的 lcm × cp。这条约束是**刻意严格**的：CP 把每篇文档切成 `cp` 个等长
  chunk，除了 `doc % (cp × ratio) == 0`，**每个 chunk 自身**也必须是 ratio 的整数倍，否则该
  rank 的压缩前缀会停在组中间，而部分压缩块不在已验证的内核契约内——因此不放松。例如
  cp=2、ratio=128、doc=128 会得到 64-token chunk（不是 128 的倍数）并被明确拒绝。
  `--dataloader.per_doc_alignment` 必须是对应 lcm × cp 的倍数：`Trainer` 启动时由
  `require_per_doc_alignment` 检查，每个 geometry 的 entry 再由
  `require_document_alignment(cu_seqlens, cp_size=cp, ratio=r)` 检查；示例脚本已按 `${CP}`
  传入 `--dataloader.per_doc_alignment`。
- 需要完整 NPU 环境（`torch_npu`、CANN、HCCL、匹配的 `cann_ops_transformer`）。只有 AscendC 融合
  路径支持 CP；模型目录的参考 metadata hook 显式拒绝 CP。
- 量化后端（V4 的 `QuantV4Metadata`、V4.1 的 `_V41QuantAttentionMetadata`）目前不重建 rank-local
  metadata，`prepare_cp_metadata` 直接拒绝：量化路径只能跑 `context-parallel-degree 1`。
- DeepSeek-V4 的融合注意力在 CP=1 时支持 `local_batch_size > 1`，QueryExchange 前把 B/S 展平，交换后恢复 `[B, S]`。CP>1 暂不支持 `local_batch_size > 1`，`CarryExchange` 在交换或边界计算前显式报错；DSA 路径 TP 固定为 1。

## 3. 两栈差异总览

| 维度 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| 压缩器与 CP | ratio-4 重叠块需要前驱 rank 的 carry：`CarryExchange`（与压缩器同级的模块）做交换/prepend，调用点压缩后再裁剪 | 压缩器 CP-agnostic：文档按 `cp × ratio` 对齐后，本 rank 的 slab 起点就是组边界，本地池化即可；无 carry |
| 压缩流取得方式 | 本地流（含 carry）池化后，内核边界 `S(1) -> R` gather 成整行 | 本地 slab 池化后同样在核心边界 `S(1) -> R` gather |
| 内核锚定 | 融合内核把 mask 右对齐到**它拿到的 key frame 末尾**；selector 完全不读 `seqused_k`，因此非末尾 rank 还要把两路 key 流压成 rank-local 前缀池 | 内核读 `seqused_q` / `seqused_ori_kv` / `seqused_cmp_kv`，窗口与压缩因果上限直接锚在 reach 上；不需要前缀池 |
| rank-local 前缀池 | `selector_k`/`selector_k_gather`（压缩池）与 `swa_k_prefix`/`swa_k_gather`（滑窗） | 无 |
| frame 归属 | 模型 metadata（`DeepSeekV4Metadata`） | 融合 entry（`AscV41Metadata`） |
| indexer teacher 归一化 | `_LightningIndexerTND.backward` 除以整行 query 数 `cu_seq_q[-1]` | `num_global_queries`（本 step 全部 query 数，按 DP 求和） |
| 参考 metadata | `DeepSeekV4Metadata.prepare_cp_metadata` 拒绝 | `DeepSeekV41Metadata.prepare_cp_metadata` 拒绝 |
| 量化 metadata | `QuantV4Metadata` 拒绝 | `_V41QuantAttentionMetadata` 拒绝 |

## 4. DeepSeek-V4

### 4.1 数据流

```text
Attention.forward
  x (本地 slab)
    ├─ QueryExchange.permute -> x_q (本 rank 的文档 chunk)
    │     ├─ q 投影 / q_norm / q_head_norm / rope(positions_q)
    │     ├─ indexer idx_q / idx_w（查询帧）
    │     └─ wo_b -> QueryExchange.unpermute -> 本地 slab
    ├─ swa_k 投影 + rope(local positions)              [CP=S(1)]
    ├─ compressor(x)（见 4.3，ratio 4 带 carry）        [CP=S(1)]
    └─ indexer compressor(x_keys)                      [CP=S(1)]

inner_attention / lightning_indexer 的 ShardingConfig
  swa_k, cmp_k（核心）与 idx_k（selector）: cp S(1) -> R（内核边界 all-gather 整行）
  q, idx_q, idx_w:      保持本 rank 的 chunk（attention 直接调用这两个模块）
```

`AscV4Metadata` 的 `q`/`swa_k`/`cmp_k` frame 被 SMLA/SMLAG 与独立的 LI override（前向
`lightning_indexer`、反向 SLIKG）直接读取；不再有 `_assemble_tnd`、`cmp_k_global_gather_indices`、
`compressed_rows`、`first_indices` 或容器网格。

### 4.2 rank-local 前缀池

压缩流与滑窗流在 gather 后都是整行，但 V4 的融合内核把 mask 右对齐到**它拿到的 key frame 末尾**：
selector 完全不读 `seqused_k`；SMLA/SMLAG 的压缩侧还要求 key 张量、网格与 `sparse_indices` 在
同一坐标 frame 里。因此 `shard_asc_v4_metadata` 为非末尾 rank 构造两个前缀池：

- 压缩池 `selector_k`：`seqused = reach // ratio`（每篇文档前缀块数）、`cu_seqlens =
  cumsum(prefix)`；`selector_k_gather` 是把整行压缩池压成该布局的 `index_select`。每个 ratio > 1
  的几何都有：ratio-4 的选择器按它选（下标仍是文档内编号），更宽 ratio（HCA）层的 full-causal
  压缩分支直接读它。`AscCompressedSparseInnerAttention` 用该 `index_select` 把 `cmp_kv` 压到池上，
  SMLA/SMLAG/SLIKG 全部读这个池。
- 滑窗池 `swa_k_prefix`：每篇文档前 `reach` 个 token，`cu_seqlens = cumsum(reach)`、
  `seqused = reach`；`swa_k_gather` 是同样的 `index_select`，SMLA/SMLAG 的 ori 记录按该帧重建。
  chunk-`r` 查询的全部窗口 key 都在这个前缀里，右对齐窗口因此落在 chunk `r` 的末尾。

两路 gather 都可微：反向经 `index_select` scatter 回整行，再由 `S(1) -> R` 的 reduce-scatter 回到
本 rank 的 chunk。`cp=1` 与最后一个 rank 的 `reach` 就是整篇文档，此时两个前缀帧都是 `None`
（等同整行帧），行为与之前逐位一致。

### 4.3 压缩器：carry / prepend / trim

压缩器本身**不感知 CP**：它只看到一条本地流、这条流自己的 `positions`，以及（仅 ratio-4 重叠）
这条流的 `cu_seqlens`；池化是扁平的 `ratio` 折叠，重叠的左半块在每个 segment 起点被 mask
（score `-inf`、kv `0`）。carry / prepend / trim 都在压缩器外的 `CarryExchange`
（`models/common/carry_exchange.py`）里，CP 的 KV all-gather 仍由 `sharding.py` 在融合内核边界
声明式完成。

ratio-4 的**重叠前驱**由 `CarryExchange` 处理：

1. 每篇文档长度按 `cp × ratio` 对齐，slab 边界落在压缩网格上；但 ratio-4 的块首池化需要前一块的行。
2. rank `r > 0` 把本 rank 最后 `ratio` 个 token **及其 positions** 通过两次 `all_to_all_single`
   （token、positions）发给 rank `r+1`，同时接收 rank `r-1` 的。
3. 把收到的 block prepend 到本地 slab 前面；`CarryExchange` 用「整行边界裁剪到增强窗口
   `[slab_start-ratio, slab_end)` 再 rebase」构造增强 `cu_seqlens`：若 slab 起点在某文档内部，
   窗口里没有落在接缝上的边界，carry 自然与该文档的第一段合并；若起点正好是文档边界，整行边界
   被保留，carry 是前一篇文档的独立 segment。两种情况都只靠「mask 每个 segment 起点」这一条规则，
   carry 自己的输出块会被裁掉，因此不需要特判。
4. 用增强后的 `cu_seqlens` 调压缩器（融合实现 `coff=2`），输出比本地块多一块；丢掉第一块
   （carry block），保留本地 slab 的 `C/ratio` 块。
5. RoPE 使用 `positions[..., ::ratio]`（保留块的文档内位置）。
6. `CarryExchange` 只产出增强流（`CarryStream`：`x` / `positions` / `cu_seqlens` / `trim`）；
   融合实现的 `_state_inputs` 在内部按增强流的形状符号构造全 0 的 kernel tiling 表
   （`state_block_table` 不在模型层接口上）。ratio=128 无重叠，不做 carry，压缩器直接用本地
   `cu_seqlens`。

rank 0 只发送、不接收；最后一个 rank 只接收、不发送；其余 rank 两者都做。所有 rank 都进入同两次
集合通信（空 split 也参与），避免死锁。

非 CP 路径把整行 `cu_seqlens` 直接传给压缩器；参考实现与融合实现共用同一接口
`compressor(x, positions, *, cu_seqlens=None)`（`cu_seqlens` 是 keyword-only：带 sharding
config 的模块走框架的位置参数契约，位置参数个数必须与签名一致）。
输入 `x` 为 `[B, L, D]`，每行 token 数必须是 `ratio` 的整数倍，positions 为 `[B, L]`。
CP=1 时可将整批展平成 `[1, B * L, D]` 打包流执行压缩，再还原为 `[B, L // ratio, D]`；
每行 positions 从 0 开始，上游 `create_varlen_metadata_for_document` 将这些行首记为文档边界，
避免 ratio-4 的重叠跨行。CP>1 的 carry/prepend/trim 仍要求 B=1。

### 4.4 参考路径

`DeepSeekV4Metadata.prepare_cp_metadata` 明确拒绝 CP：参考内核（Attention Gym 的 `gather_attn`）
只有一条连续文档流的契约。参考**压缩器**本身支持 CP（carry 在 `CarryExchange` 里，与压缩器
实现无关），
所以 debug model（参考压缩器 + 融合注意力）也能跑 CP。V4 的数值验收在融合路径的 CP=1/CP=2
之间做。

## 5. DeepSeek-V4.1

### 5.1 数据流

```text
Attention.forward
  x (本地 slab)
    ├─ QueryExchange.permute -> x_q (本 rank 的文档 chunk)
    │     ├─ q 投影 / q_norm / rope(positions_q)
    │     └─ indexer：idx_q / weights 在 chunk 上；idx_k 由本地压缩 latent 投影
    ├─ swa_k 投影 + rope(local positions)              [CP=S(1)]
    └─ compressor(x, positions)（本地池化，无 carry）    [CP=S(1)]

Sparse core / selector（sharding 声明）
  swa_k, cmp_k, idx_k: cp S(1) -> R（整行）
  q, idx_q, weights:   保持本 rank 的 chunk
  topk_indices, sink:  replicated

输出：rope(positions_q, inverse) -> wo_a -> wo_b -> QueryExchange.unpermute -> 本地 slab
```

V4.1 的参考 metadata（`DeepSeekV41Metadata`）只带整行边界与 `positions_q`；CP 的 frame 与记录由
融合 entry 重建，消费者（`AscV41SparseAttention`、`AscSelector`）直接读
`attention_metadata.q` / `.swa_k` / `.cmp_k` 与 `aicpu_metadata`。

### 5.2 rank-local frame 重建

`AscV41Metadata.prepare_cp_metadata`（`override/deepseek_v4_1/sparse_attn/ascendc.py`）从
`cp_shard_context()` 取 pre-shard `positions` 与 mesh，调用 `shard_asc_v41_metadata`：

```text
seqused = diff(cu_seq_q)
alignment = cp_size * max(ratio, 1)          # 每篇文档必须对齐
chunk = seqused // cp_size                   # Q_d
reach = (cp_rank + 1) * chunk                # K_d
exchange = cp_plan(cu_seq_q, cp_size, cp_rank, seq_len)
q       = Frame(cu_seq_q // cp_size, chunk)
swa_k   = Frame(cu_seq_q, reach)
cmp_k   = Frame(cu_seq_k, reach // ratio, residual=reach % ratio)   # ratio > 0
positions_q = exchange.positions_q.reshape(B, seq_len)
```

并检查 `sum(chunk) == batch_size * seq_len`（边界描述的必须是调用方切出来的那条行）。frame 变
则记录必须重算：entry 用 `build_aicpu_metadata(q, swa_k, cmp_k, geometry)` 重建
`sparse_attn` 与 `lightning_indexer` 两组 `*_metadata`。

V4.1 的 `compress_ratio` 可以是 0（window-only，没有任何压缩 frame）、1（整分辨率第二路 KV）
或 2（压缩共享 KV）；`cmp_k` 只在 `ratio > 0` 时存在，`ratio == 1` 的 `residual` 为 `None`
（轴没有折叠，内核对非零值报错）。

### 5.3 压缩器与 sharding

V4.1 的 `Compressor`（`models/deepseek_v4_1/compressor.py`）没有 CP 分支：source layer 在本地
slab 上做 `ratio` 折叠（ratio==1 是普通投影），因为文档已按 `cp × ratio` 对齐，本 rank 的 slab
起点就是组边界，`C/ratio` 是整数块。压缩输出（`[B, C/ratio, D]`）在 sparse core 的边界被声明式
`cp: S(1) -> R` gather 成整行；`cmp_k.seqused = reach // ratio` 让内核只读本 rank 之前的块。
ratio-4 那种跨 slab 的 carry 在 V4.1 不存在，因此没有 `CarryExchange`。

`models/deepseek_v4_1/sharding.py` 声明 CP 的输入/输出放置：

- sparse core：`q` 保持 `S(2)`（本 rank chunk），`swa_k` / `cmp_k` 在消费侧为整行
  （`in_src_shardings` 是 replicated、`in_dst_shardings` 是行放置，即 `S(1) -> R`），
  `topk_indices` 与 `attn_sink` replicated；
- selector：`idx_q` / `weights` replicated，`idx_k` 做同样的 `S(1) -> R`（选择器用整行 index keys
  给本 rank 的 chunk 打分）。

### 5.4 indexer teacher 的归一化

V4.1 的 selector 在反向把 teacher 除以 `Selector.num_global_queries`：本 step 全部 accumulation
microbatch 的 query token 数（在 CP 切分之前统计，DP 维再求和）。这与 V4 用
`cu_seq_q[-1]`（整行 query 数）是同一个量的两种来源，都是为了避免 CP/DP 下参数梯度被
rank-local 分母放大。

### 5.5 拒绝 CP 的路径

`DeepSeekV41Metadata.prepare_cp_metadata` 拒绝参考内核；`_V41QuantAttentionMetadata`
（`extensions/adaptor/torchao/converter.py`）拒绝量化 attention。两者都在任何层执行之前报错，
不会静默地拿着全局 metadata 跑 rank-local 行。

## 6. 组件

| 文件 | 职责 |
| --- | --- |
| `models/common/query_exchange.py` | `ExchangeMetadata` / `QueryExchange` / `cp_plan`（V4、V4.1 共用） |
| `models/common/metadata.py` | `KernelFrame`（两栈共用的 frame 原语） |
| `models/deepseek_v4/metadata.py` | `DeepSeekV4Metadata`（整行边界、池网格、q/swa_k/cmp_k frame、查询位置）与 `build_v4_metadata` |
| `models/deepseek_v4/attention.py` | 查询交换、q/idx_q 在 `positions_q` 上 rope、selector + 内层核心直调（无 wrapper）、输出 `unpermute`；参考内核读 `q` / `cmp_k` frame 与预计算的 HCA 选择 |
| `models/common/carry_exchange.py` | `CarryExchange` / `CarryStream`：ratio-4 carry 的 ring（tokens + positions）、增强 `cu_seqlens` 与 `trim`；cp=1 为 no-op |
| `models/deepseek_v4/compressor.py` | 参考压缩器（CP-agnostic，CP/非 CP 同一实例）；`Indexer` 的查询帧与 key slab 分离 |
| `models/deepseek_v4/sharding.py` | `swa_k` / `cmp_k` / `idx_k` 的 `cp: S(1) -> R` 声明 |
| `override/deepseek_v4/metadata.py` | `AicpuMetadata` / `AscV4Metadata.prepare_cp_metadata`（frame 归模型层，entry 只加记录） |
| `override/deepseek_v4/sparse_attn/ascendc.py` | `shard_asc_v4_metadata`（q/swa/cmp 帧 + 记录重建）、SMLA/SMLAG 的 frame 参数与 rank-local 前缀池 |
| `override/deepseek_v4/lightning_indexer/ascendc.py` | LI 前向 / SLIKG 反向（选择器自己的 frame，`selector_k` 前缀池、teacher 边） |
| `override/deepseek_v4/lightning_indexer/pypto.py` | PyPTO 版 selector：独立实现同一契约（输入准备、排序、teacher 边），只替换 LI/SLIKG kernel 对 |
| `override/deepseek_v4/compressor/ascendc.py` | CANN 融合压缩器（无 carry、无 CP 分支；`_state_inputs` 按 stream 形状内部构造零 tiling 表） |
| `models/deepseek_v4_1/attention.py` | V4.1 查询交换、本地 KV/压缩、输出 `unpermute` |
| `models/deepseek_v4_1/compressor.py` | V4.1 本地池化压缩器（CP-agnostic，无 carry） |
| `models/deepseek_v4_1/sharding.py` | V4.1 sparse core / selector 的 `S(1) -> R` 声明 |
| `models/deepseek_v4_1/model.py` | `DeepSeekV41Metadata.prepare_cp_metadata` 拒绝 CP；teacher 成对启用检查 |
| `override/deepseek_v4_1/sparse_attn/ascendc.py` | `AscV41Metadata` / `shard_asc_v41_metadata` / `build_fused_metadata` |
| `override/deepseek_v4_1/lightning_indexer/ascendc.py` | V4.1 BF16 selector（整行 index keys × 本 rank chunk） |

## 7. 验证

CPU：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q \
  tests/unit_tests/models/deepseek_v4 tests/unit_tests/override/deepseek_v4
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest -q \
  tests/unit_tests/models/deepseek_v4_1 tests/unit_tests/override/deepseek_v4_1
```

V4 覆盖 `test_v4_cp_frames.py`（帧坐标、对齐拒绝、`CarryExchange` + 压缩器对整行 oracle 的
输出）与 `test_compressor.py::test_compressor_cp_carry_trim_matches_the_global_pool`（融合压缩器
经 `CarryExchange` 的 carry/trim 输出与梯度）。V4.1 覆盖
`tests/unit_tests/models/deepseek_v4_1/test_context_parallel.py`（`cp_plan` 的行移动、闭式接收
split、frame 不变量）与 `tests/unit_tests/override/deepseek_v4_1/test_cp_frames.py`
（`shard_asc_v41_metadata` 的 rank-local 帧与记录）。

NPU：加载同一固定初始权重 checkpoint，CP=1 / CP=2 各跑相同步数，比较逐 step `loss` 与
`grad_norm`；每次使用全新 `--checkpoint.folder`。数值差异按 CANN batch 一致性解释（SMLA 批次
布局、LI tie、SMLAG kv 部分和），不要求逐位一致。V4 的对齐权重实测：每层 hidden 相对误差
0.000% / 0.017% / 0.054% / 0.253%，`loss` 差 1.9e-4。

## 8. 相关文档

- [DeepSeek-V4 TND 适配](deepseek_v4_tnd.md)
- [融合算子使用与开发](fused_ops.md)
- [数值对齐](../developer_guides/numerical_alignment.md)
- [MTP](mtp.md)
