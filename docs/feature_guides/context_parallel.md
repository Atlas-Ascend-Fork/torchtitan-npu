# DeepSeek-V4 系列上下文并行（CP）

DeepSeek-V4（DSV4）与 DeepSeek-V4.1（DSV4.1）共用同一套上下文并行语义：**查询按文档内部分成
`cp` 份，KV 与压缩 KV 保持整行**。融合内核在边界处把本 rank 的 flatten 段声明式 all-gather 成整行，再用
frame 的 `seqused = reach` 把可见范围限制在本 rank 的 chunk 之前。两栈共用
`torchtitan_npu/models/deepseek_v4/metadata.py` 的 `KernelFrame`，也共用同一份输入布局：
`torchtitan_npu/models/deepseek_v4/cp_attention.py` 的静态 doc-chunk 布局由 **CP 切分本身**施加——`cp_shard`
在切之前按布局给出的置换重排整行，所以每个 rank 拿到的输入就是本 rank 的 chunk，不需要
任何集合通信；每层只在需要"本 rank 窗口"的消费方内部做恢复（压缩器池化前、核心取 swa 键前，
同一个 `restore_key_stream`），layer 里不再有 `x_k`。要说明次数：不池化层是一次（swa 键）；池化层是两次
（swa 键 + 主压缩器的 `x`）；池化且带 indexer 的几何是三次，因为 indexer 自己的压缩器对同一份 `x`
再恢复一次——这是「key 侧自己管移动」的代价，各压缩器不必知道别的模块已经恢复过它。ratio-4
重叠的 carry 由 `torchtitan_npu/models/deepseek_v4/cp_compressor.py` 的 `CarryExchange` 在调用点处理，
两栈的压缩器本身都不感知 CP。

本文是 V4 与 V4.1 的合并指南（原 V4 CP 指南的内容已并入）。

## 1. 共同契约

### 1.1 坐标

设一行有文档 `d`、长度 `L_d`，CP 度 `R = cp_size`，rank `r`，flatten 宽度 `C = N / R`
（`N` 是整行 token 数），chunk 宽度 `Q_d = L_d / R`：

```text
rank r 的输入（doc-chunk）：   每篇文档的第 r 个 chunk，文档顺序（cp_shard 的置换结果）
本 rank 的 flatten 段：     [r·C, (r+1)·C)，由消费方按需从上面的 chunk 行恢复（压缩器内部、核心之前的 swa 键）
查询帧 q：                 每篇文档的第 r 个 chunk（输入布局已放好，查询不移动）
滑窗 KV 帧 swa_k：         整行；seqused = reach_d = (r+1)·Q_d
压缩 KV 帧 cmp_k：         整行按 ratio 折叠；seqused = reach_d // ratio，residual = reach_d % ratio
positions：                 查询自己的文档内位置（输入自带，查询侧不再单列字段）
position_k：               本 rank flatten 段自己的文档内位置（entry 顶层字段）
```

`q.cu_seqlens = 整行边界 // R`（每篇文档 chunk 的前缀和），`q.seqused = Q_d`；
`swa_k.cu_seqlens = 整行边界`，`cmp_k.cu_seqlens = 整行边界 // ratio`。因此
`seqused_ori_kv - seqused_q = r·Q_d` 正好是本 rank 在该文档内的 chunk 偏移，内核的滑窗与压缩
因果上限都锚在这个偏移上。

整行网格只由 frame 承载：行网格 = `swa_k.cu_seqlens`，查询网格 = `q.cu_seqlens`，压缩网格 =
`cmp_k.cu_seqlens`，ratio-1（不折叠的压缩轴）的 key 网格 = `swa_k.cu_seqlens`。key 流的文档内位置是 entry 顶层的
`position_k`：degree=1 时就是输入的 `positions`，CP 下是本 rank flatten 段的。两栈都不单列查询位置
字段——查询的行就是输入的行，它的位置由输入自带；两栈也都没有 `cu_seq_q` / `cu_seq_k` 别名，网格一律
从 frame 读（行 = `swa_k.cu_seqlens`，查询 = `q.cu_seqlens`，压缩 = `cmp_k.cu_seqlens`），V4.1 的
`DeepSeekV41Metadata` 只是 V4 那个类的空子类，只多一句拒绝参考栈跑 CP 的话。

### 1.2 plan 与 exchange

输入侧的移动就是 CP 切分的置换，只有一种：

- `DocChunkCPLoadBalancer.generate_permutation()` 给出整行的全局 token 序：把每篇文档切成 `cp_size`
  块、按「第 `r` 块、文档顺序」拼起来，于是置换后 rank `r` 落在 `[r·C, (r+1)·C)` 的窗口正是「每篇文档
  的第 `r` 个 chunk，文档顺序」。
- 布局按上游的方式接进切分：`DocChunkCPLoadBalancer`（`models/deepseek_v4/cp_layout.py`）是上游
  `ContextParallelLoadBalancer` 的子类，模型包 import 时把它注册到 `doc_chunk` 这个名字下。
  `Decoder._cp_shard` 用 `build_load_balancer(doc_chunk, seq_len=…, cp_size=…, attention_metadata=…)`
  构造它，`Config().build` 拿到的是**第一层 attention 的那个 entry**（上游同样按
  `first_attention` 选），边界即该 entry 的 `q.cu_seqlens`；`generate_permutation` 把每篇文档切成
  `cp_size` 块、按「第 r 块、文档序」拼出整行顺序，再由 `shard_tensors`
  交给 `_context_parallel_shard`：先重排**复制在每 rank 的整行**，再均分切片，所以这一步零通信，
  反向是 `argsort` 的逆置换（autograd 自己处理）。调用方不预先算任何下标，也不用把 positions
  单独递进去——与上游 `HeadTailCPLoadBalancer` 的差别只剩「边界从哪读」。
- 还需要从 positions 反推边界的只有一处：两个栈共用的 `DeepSeekV4Metadata.from_positions`
  （`rows - positions` 的 `unique_consecutive`，不经过上游的 `VarlenMetadata`）。1M 配方的 Engram
  history 与 V4 的 MTP 深度都不需要——它们复用 `_cp_shard` 算出的置换，不自己再建 balancer。
- 布局里的"本 rank 的数"（`rank`、`cp_size`、flatten 宽度）一律是 **python int**：rank 是运行的 host
  值，写成 device 标量只会让张量算术无法区分常量与本 rank，并把整条路径绑到单进程模拟上。
- `ExchangeMetadata.from_permutation(perm, cp_size=…, cp_rank=…)` 求 shard 置换的逆，产出本 rank 的交换表：
  - `send_index` / `send_split`：本 rank flatten 段里每一行要去的 peer（按 peer 连续、peer 内按槽位
    升序），前者是重排下标，后者是 `input_split_sizes`；
  - `recv_split`：本 rank 从每个 peer 收到的行数（`output_split_sizes`）。接收 buffer 已经是
    held-row 顺序，所以没有 scatter 下标、没有二次重排；每篇文档的 chunk 是一个全局连续区间、
    至多跨两个 flatten 段，接收侧因此只需 split。
  - flatten 宽度由置换的长度除以度数推出，调用方不传；`dst` 是「本 rank 这一行要去哪个 peer」，`src` 是「本 rank 持有的这一行来自哪个 peer」。
- `apply(tensor, group=…)` 是 doc-chunk -> flatten 的精确逆（先换两边的 split 再按 `send_index`
  scatter，置换的反向就是另一次 all_to_all）；`restore_key_stream(cp, x, group=…)` 在 `cp is None`
  时恒等返回、不碰进程组，所以同一条前向路径服务两个 degree，调用点不需要分支。
- 本 rank flatten 段自己的文档内位置由 `doc_relative_positions(cu_seqlens, 行号)` 算出（纯 device），
  它填的就是 entry 的 `position_k`。
- rank-local entry 由 `DeepSeekV4Metadata.doc_chunk_entry(cp_size=…, cp_rank=…)` 产出（V4.1 继承
  同一个方法），两个 port 的 CP 钩子只在它之上重建自己的 AI_CPU records。

**host 同步的位置**：`ExchangeMetadata` 的 `send_split` / `recv_split` 是其中一处。集合通信按 python
list 接收 split，这是框架的签名；它们由 plan 推导，而 plan **每个 microbatch 每个 geometry 只建一次**
（不是每层），所以这部分同步是 O(microbatch × geometry)。同一频率上还有一处：布局本身——
`_document_chunks` 要按每篇文档的起止切 `arange`，边界是 device 张量，切之前得 `.tolist()` 一次
（每个 microbatch 一次，不是每层）。

每层内的路径上没有 D2H：ratio-4 的 carry ring 用 host rank 直接构造两条 split（`_ring_split`，每层
0 次同步），`position_k` 与 carry 的边界裁剪都是 device 算术。arch22 的 key
压实每个 microbatch 同步一次（`nonzero` 的行数是 device 值，见 §4.2）。

### 1.3 frame 与 metadata

一行由三个 `KernelFrame` 描述：`q`、`swa_k`、`cmp_k`（不池化的几何没有 `cmp_k`）。CP 下由融合
entry 的 `prepare_cp_metadata` 用 rank-local chunk 重建这三个 frame 以及按它们算出的 AI_CPU
记录；参考 entry 的 hook 显式拒绝 CP（它只有一条连续文档流的契约）。frame 的归属两栈不同：

| 栈 | 参考 metadata | CP 重建 |
| --- | --- | --- |
| V4 | `DeepSeekV4Metadata` 自带 `q`/`swa_k`/`cmp_k` 与 `xchg: ExchangeMetadata`（`DeepSeekV4Metadata.from_positions` 用整行边界构造） | `AscV4Metadata.prepare_cp_metadata` → `doc_chunk_entry` → `AscV4Metadata.build_aicpu_metadata` |
| V4.1 | `DeepSeekV41Metadata` 只有整行边界与位置；frame 由融合 entry `AscV41Metadata` 携带 | `AscV41Metadata.prepare_cp_metadata` → `shard_asc_v41_metadata` |

两栈的重建都从 `cp_shard_context()` 读 pre-shard `positions` 与 CP mesh，并要求
`--parallelism.context-parallel-load-balancer doc_chunk`：这些 frame 描述的就是 doc-chunk 布局，切分用
的布局与 frame 描述的不一致会静默错位，所以在 entry 里直接拒绝。V4 的 `doc_chunk_entry` 重算 frame
与 `restore`、`position_k`；V4.1 的 `shard_asc_v41_metadata` 同样重建这三样，只是入口不同。

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

V4.1 的 CP 配方是 `examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_1m_a5.sh`：脚本默认
`CP=128`、`EP=128`、`--dataloader.per_doc_alignment 256`，启动时按 `EP` 校验拓扑（`WORLD_SIZE`
与 `384` 都要能被 `EP` 整除）；`CP` / `EP` 可用同名环境变量覆盖。

- `--parallelism.context-parallel-degree > 1` 启用 CP；必须配合
  `--parallelism.spmd-backend spmd_types`（声明式 `S(1) -> R` all-gather 的前提）。切分与 metadata
  重建之后，`preprocess_inputs` 统一施加输入 SPMD 标注，这一步只在 `spmd_types` 后端执行。
- 布局必须是 doc-chunk：`--parallelism.context-parallel-load-balancer doc_chunk`（上游默认 `headtail`
  会被融合 entry 的 `prepare_cp_metadata` 拒绝）。输入在切之前按该置换重排，零通信；示例脚本已按
  该值传入。布局由 config **显式**命名（两栈的 registry 都写 `doc_chunk`），没有"留空则从 metadata
  反查"的默认——一个 rank-local entry 只对一种行排列成立，错配在 metadata 边界直接报错。
- 每篇文档按 `lcm(compress_ratios) × cp` 填充：V4 为 `128 × cp`（CP=2 即 256），V4.1 为其
  `compress_ratios` 的 lcm × cp（当前 lcm 为 2，1M 配方用 256 = 2 × 128）。这条约束是**刻意严格**
  的：CP 把每篇文档切成 `cp` 个等长 chunk，除了 `doc % (cp × ratio) == 0`，**每个 chunk 自身**也
  必须是 ratio 的整数倍，否则该 rank 的压缩前缀会停在组中间，而部分压缩块不在已验证的内核契约内
  ——因此不放松。例如 cp=2、ratio=128、doc=128 会得到 64-token chunk（不是 128 的倍数），该 rank 的
  压缩前缀会停在组中间，压缩器与内核按契约直接失败。示例脚本已按该值传入
  `--dataloader.per_doc_alignment`；运行期不再校验这个 flag，也不再逐 geometry 校验行——对齐是**数据
  侧的契约**（配置 registry 用栈内 ratio 的 lcm 推出 `per_doc_alignment` 该填多少），违反它时由压缩器的
  `view`/`unflatten` 与内核的网格检查暴露。
- 需要完整 NPU 环境（`torch_npu`、CANN、HCCL、匹配的 `cann_ops_transformer`）。只有 AscendC 融合
  路径支持 CP；模型目录的参考 metadata hook 显式拒绝 CP，V4.1 的 `update_from_config` 还会要求
  同时请求 `override.deepseek_v4_1.sparse_attn.asc` 与
  `override.deepseek_v4_1.lightning_indexer.asc`。
- 量化后端（V4 的 `QuantV4Metadata`、V4.1 的 `_V41QuantAttentionMetadata`）也按本 rank 的 frames
  重建自己的 AI_CPU metadata：V4 只替换量化 selector 的 forward 记录，V4.1 重建整套，两者都接受
  `context-parallel-degree > 1`。
- DeepSeek-V4 的融合注意力在 CP=1 时支持 `local_batch_size > 1`（输入布局按批展平，attention 输出保留
  `[B, S]`）。CP>1 暂不支持 `local_batch_size > 1`，`CarryExchange` 在交换或边界计算前显式报错；DSA 路径
  TP 固定为 1。

## 3. 两栈差异总览

| 维度 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| 输入布局 | 同一份：`cp_shard` 的 doc-chunk 置换，tokens / labels / positions / MTP 深度一次到位 | 同左 |
| 压缩器与 CP | ratio-4 重叠块需要前驱 rank 的 carry：`CarryExchange`（压缩器的子模块）做交换/prepend，调用点压缩后再裁剪 | 压缩器 CP-agnostic：文档按 `cp × ratio` 对齐后，本 rank 的 flatten 段起点就是组边界，本地池化即可；无 carry |
| 压缩流取得方式 | 本地流（含 carry）池化后，内核边界 `S(1) -> R` gather 成整行 | 本地 flatten 段池化后同样在核心边界 `S(1) -> R` gather |
| 内核锚定 | arch35 读 `seqused_q` / `seqused_ori_kv` / `seqused_cmp_kv`，窗口与压缩因果上限直接锚在 reach 上；arch22 的内核不读 key 侧 `seqused`、把 mask 右对齐到 frame 末尾，因此非末尾 rank 的两路 key 流要压成 rank-local 前缀 | 同 arch35（V4.1 只跑在 arch35 上） |
| rank-local key 压缩（仅 arch22） | `A3KeyWorkaround`（`override/a3_workaround.py`）：`swa_k_gather` / `cmp_k_gather` 两个可微 `index_select`，由 `AscV4Metadata.build_aicpu_metadata` 在 `if not _IS_A5:` 里加在 `AscV4Metadata` 上 | 无 |
| frame 归属 | 模型 metadata（`DeepSeekV4Metadata`，含 `xchg: ExchangeMetadata`） | 融合 entry（`AscV41Metadata`） |
| indexer teacher 归一化 | `_LightningIndexerTND.backward` 除以整行 token 数 `swa_k.cu_seqlens[-1]`（量化 selector 同） | `num_global_queries`（本 step 全部 query 数，按 DP 求和） |
| 参考 metadata | `DeepSeekV4Metadata.prepare_cp_metadata` 拒绝 | `DeepSeekV41Metadata.prepare_cp_metadata` 拒绝 |
| 量化 metadata | `QuantV4Metadata` 重建（只换 selector 的 forward 记录） | `_V41QuantAttentionMetadata` 重建（整套记录） |

## 4. DeepSeek-V4

### 4.1 数据流

```text
Decoder.preprocess_inputs（每个 microbatch 一次）
  整行 [1, N] --cp_shard(doc_chunk 置换)--> 本地 chunk 行 [1, C]
    tokens / labels / positions 随同一次切分到位（零通信）
    随后按 spmd_backend == spmd_types 施加 annotate_input_spmd_types
DeepSeekV4Model.preprocess_inputs 只额外准备 MTP 深度，并让它们走同一次 cp_shard

Attention.forward（每层）
  x（doc-chunk 行）
    ├─ wq_a / q_norm / wq_b / q_head_norm / rope(positions)   [查询不移动]
    ├─ swa：wkv / kv_norm / rope(positions)（chunk 行）      [CP=S(1)]
    │    └─ restore_key_stream（ExchangeMetadata.apply）-> 本 rank flatten 段，供核心边界 gather
    ├─ compressor(x)（见 4.3：内部 restore + ratio-4 carry）  [CP=S(1)]
    └─ indexer：idx_q / idx_w 在 chunk 行上投影；idx_k 由它自己的压缩器产出
  wo_a / wo_b 输出留在 doc-chunk 行（不 unpermute）

inner_attention / lightning_indexer 的 ShardingConfig
  swa_k, cmp_k（核心）与 idx_k（selector）: cp S(1) -> R（内核边界 all-gather 整行）
  q, idx_q, idx_w:      保持本 rank 的 chunk（attention 直接调用这两个模块）
```

`preprocess_inputs` 的 CP 切分就是本布局：基类 seam 用布局的 `generate_permutation` 生成的置换
（`--parallelism.context-parallel-load-balancer doc_chunk`）先重排整行再均分，tokens、labels 与
positions 一次到位；`DeepSeekV4Model.preprocess_inputs` 只把 MTP 的每个深度也用同一次 `cp_shard`
切好，保证预测与被预测的行一起换位。切完由 `annotate_input_spmd_types` 重新施加输入 SPMD 标注。
没有 CP（或 `cp_size == 1`）时整段跳过，batch 原样进入前向。

`Attention.forward` 收到的是 doc-chunk 行，查询侧直接投影、在随行搬来的 `positions` 上做 RoPE。key
侧同样在这批行上投影、旋转、池化——每一步都是逐 token 的，因此与移动可交换——需要"本 rank 窗口"的
消费方各自恢复：压缩器在池化前恢复（`restore_key_stream(attention_metadata.cp, …)`，degree=1 即恒等），核心在取
swa 键前恢复（同一函数，窄张量）。layers 里没有 `x_k`：移动只发生在拥有它的模块内部。输出仍留在
doc-chunk 行：`wo_b` 是逐 token 投影，残差、MoE 与 loss 都按同一布局计算，loss 的 labels 也是同一份
plan 搬过的。

`AscV4Metadata` 的 `q`/`swa_k`/`cmp_k` frame 被 SMLA/SMLAG 与独立的 LI override（前向
`lightning_indexer`、反向 SLIKG）直接读取；V4 的 metadata 只有 `ratio`、三个 frame、`position_k`、
`xchg`，没有 `cu_seq_q`/`cu_seq_k`，也没有单独的重切函数（rank-local entry 由
`doc_chunk_entry` 产出）。不再有 `_assemble_tnd`、`cmp_k_global_gather_indices`、`compressed_rows`、
`first_indices` 或容器网格。

### 4.2 arch22 的 key 压缩

压缩流与滑窗流在 gather 后都是整行，其 `seqused` 是本 rank 的 reach。arch35 的内核直接读它；arch22
的内核（`SparseFlashMla` / `LightningIndexer` 系列）不读 key 侧 `seqused`，而是把 mask 右对齐到
**它拿到的 key frame 末尾**，SMLA/SMLAG 的压缩侧还要求 key 张量、网格与 `sparse_indices` 在同一
坐标 frame 里。因此 `AscV4Metadata.build_aicpu_metadata`（`override/deepseek_v4/asc_metadata.py`）在
`if not _IS_A5:` 里把两路 key 换成"reach 自己写成的帧"，并把压行下标存在 entry 的
`a3_workaround`（`override/a3_workaround.py`，与 `aicpu_metadata` 并列）上：

- `a3_key_frame(cu_seqlens, seqused, residual)`：把 reach 写成帧（`cu_seqlens = cumsum(reach)`，每篇
  文档的前缀即一段），`seqused` 保持 reach，ratio > 1 时 `residual` 为空则写全零（arch22 的 tiling
  不接受空残块）；
- `a3_key_gather(cu_seqlens, seqused)`：在 device 上算出把整行压到该帧的 `index_select` 下标
  （`repeat_interleave` + 比较 + `nonzero`；`nonzero` 的行数是 device 值，每个 microbatch 每条流
  一次同步，代价见 §7 的性能注）；
- 两个 gather 都可微：反向经 scatter 回整行，再由 `S(1) -> R` 的 reduce-scatter 回到本 rank 的
  chunk；`cp=1` 与最后一个 rank 的 reach 就是整篇文档，此时两个 gather 都是 `None`。

entry 的 `swa_k` / `cmp_k` 就是内核帧，forward / backward / selector 各取一个 frame 与一个 gather，
只有一处 `if not _IS_A5:` 用它压行；行自身的网格仍在 `q.cu_seqlens` / `restore` 里，与 arch35 一致。

**arch22 + CP 下不编译**：压实的行数是 device 值（`nonzero` 的行数），它进入
`cann_ops_transformer` 的 meta 函数后变成 unbacked 符号，图在 `Could not guard on data-dependent
expression` 处失败（CP=1 时没有压实，同一份代码正常编译）。这条路径也没有地方断图：torchtitan 每个
block 用 `fullgraph=True` 编译，而全量 activation checkpoint 会把 block 放进 `checkpoint` 这个
higher-order operator 里 trace，HOP 内部不允许图断点（`Skip calling torch.compiler.disable()d
function`）。

因此**栈自己在 `parallelize` 里做这个判断**（`models/deepseek_v4/compile_policy.py`：`_IS_A5 or cp <= 1`
才编译），判为不可编译时打一条告警并把 `compile_config.enable` 关掉，让训练照常跑 eager：
`--compile.enable` 因此可以留在同一份配方里——A5 上生效，arch22 + CP 上退化为 eager 而不是报错。
判断放在模型的 parallelize 而不是 patch 里，是因为它只取决于「哪个 arch × 多少度」，与上游
`apply_compile` 的接口无关。

**代价要写在这里**：eager 是逐算子直接跑，吞吐低于编译后的水平。原本在 A3 上开着 `--compile.enable`
跑 CP 的作业会因此变慢，日志里只有一条 warning；评估性能时按「arch22 + CP>1 不编译」这一条来读，
不要把 CP=1 的编译吞吐当成同一份配方在两个 degree 上的预期。

### 4.3 压缩器：restore / carry / prepend / trim

压缩器拿到的是**布局原样产出的流**（degree=1 是整行，CP 下是本 rank 的 chunk 行）、key 流自己的
`positions`（entry 的 `position_k`）以及**整行**的 `cu_seqlens`。池化本身仍是扁平的 `ratio` 折叠，
重叠的左半块在每个 segment 起点被 mask（score `-inf`、kv `0`）；CP 的两件额外工作在模块内部完成：

- **restore**：`restore_key_stream(attention_metadata.cp, x, group=…)`（degree=1 即恒等），把 chunk 行换回本 rank 的
  flatten 段，使池化的块网格与整行一致；
- **carry**：ratio-4 的前驱块由 `CarryExchange`
  （`torchtitan_npu/models/deepseek_v4/cp_compressor.py`）交换、prepend、池化后由压缩器自己裁掉。

调用方只交 `x` 与 `position_k`，看不到这两步。CP 的 KV all-gather 仍由 `sharding.py` 在融合内核
边界声明式完成。

ratio-4 的**重叠前驱**由 `CarryExchange` 处理：

1. 每篇文档长度按 `cp × ratio` 对齐，flatten 段边界落在压缩网格上；但 ratio-4 的块首池化需要前一块的行。
2. rank `r > 0` 把本 rank 最后 `ratio` 个 token 通过一次 `all_to_all_single` 发给 rank `r+1`，
   同时接收 rank `r-1` 的。**位置不参与交换**：它是前驱 flatten 段末块的位置，由整行边界直接算出
   （`doc_relative_positions`），而且 carry 块一定会被裁掉、overlap 读的是 RoPE 之前的投影态，数值上惰性。
3. 把收到的 block prepend 到本地 flatten 段前面；`CarryExchange` 用「整行边界裁剪到增强窗口
   `[flatten_start-ratio, flatten_end)` 再 rebase」构造增强 `cu_seqlens`：若 flatten 段起点在某文档内部，
   窗口里没有落在接缝上的边界，carry 自然与该文档的第一段合并；若起点正好是文档边界，整行边界
   被保留，carry 是前一篇文档的独立 segment。两种情况都只靠「mask 每个 segment 起点」这一条规则，
   carry 自己的输出块会被裁掉，因此不需要特判。
4. 用增强后的 `cu_seqlens` 调压缩器（融合实现 `coff=2`），输出比本地块多一块；丢掉第一块
   （carry block），保留本地 flatten 段的 `C/ratio` 块。
5. RoPE 使用 `positions[..., ::ratio]`（保留块的文档内位置）。
6. `CarryExchange` 只产出增强流（`CarryStream`：`x` / `positions` / `cu_seqlens` / `trim`）；
   融合实现的 `_state_inputs` 在内部按增强流的形状符号构造全 0 的 kernel tiling 表
   （`state_block_table` 不在模型层接口上）。ratio=128 无重叠，不做 carry。

rank 0 只发送、不接收；最后一个 rank 只接收、不发送；其余 rank 两者都做。所有 rank 都进入同两次
集合通信（空 split 也参与），避免死锁。

调用点在每个 degree 都传 `metadata.swa_k.cu_seqlens`（整行边界）；`cp = 1` 时 `CarryExchange`
直接返回同一份流，压缩器拿到的是整行边界。参考实现与融合实现共用同一接口
`compressor(x, positions, *, cu_seqlens=None)`（`cu_seqlens` 是 keyword-only：带 sharding
config 的模块走框架的位置参数契约，位置参数个数必须与签名一致）。
输入 `x` 为 `[B, L, D]`，每行 token 数必须是 `ratio` 的整数倍，positions 为 `[B, L]`。
CP=1 时可将整批展平成 `[1, B * L, D]` 打包流执行压缩，再还原为 `[B, L // ratio, D]`；
每行 positions 从 0 开始，上游 `create_varlen_metadata_for_document` 将这些行首记为文档边界，
避免 ratio-4 的重叠跨行。CP>1 的 carry/prepend/trim 仍要求 B=1。

### 4.4 参考路径

`DeepSeekV4Metadata.prepare_cp_metadata` 明确拒绝 CP：参考内核（Attention Gym 的 `gather_attn`）
只有一条连续文档流的契约。参考**压缩器**本身支持 CP（carry 在 `CarryExchange` 里，与压缩器
实现无关），所以 debug model（参考压缩器 + 融合注意力）也能跑 CP。V4 的数值验收在融合路径的
CP=1/CP=2 之间做。

## 5. DeepSeek-V4.1

### 5.1 数据流

```text
Attention.forward
  x (本 rank 的 chunk 行，输入布局已放好，查询不移动)
    ├─ q 投影 / q_norm / rope(positions)
    ├─ swa_k 投影 + rope(positions)（chunk 行）             [CP=S(1)]
    │    └─ restore_key_stream（ExchangeMetadata.apply）-> 本 rank flatten 段，供核心边界 gather
    ├─ compressor(x, position_k)：内部 restore，source 层池化，无 carry   [CP=S(1)]
    └─ indexer：idx_q / weights 在 chunk 行上投影，idx_k 由压缩器的 flatten 段 latent 投影
  输出留在 chunk 行：rope(positions, inverse) -> wo_a -> wo_b

Sparse core / selector（sharding 声明）
  swa_k, cmp_k, idx_k: cp S(1) -> R（整行）
  q, idx_q, weights:   保持本 rank 的 chunk
  topk_indices, sink:  replicated
```

V4.1 的参考 metadata（`DeepSeekV41Metadata`）只带整行边界与 `position_k`（degree=1 时等于输入的
`positions`）；CP 的 frame 与记录由融合 entry 重建，消费者（`AscV41SparseAttention`、`AscSelector`）
直接读 `attention_metadata.q` / `.swa_k` / `.cmp_k` 与 `aicpu_metadata`。

### 5.2 rank-local frame 重建

`AscV41Metadata.prepare_cp_metadata`（`torchtitan_npu/override/deepseek_v4_1/sparse_attn/ascendc.py`）
从 `cp_shard_context()` 取 pre-shard `positions` 与 mesh，调用 `shard_asc_v41_metadata`：

```text
xchg = ExchangeMetadata.from_permutation(perm, cp_size=cp_size, cp_rank=cp_rank)
chunk    = row.q.seqused // cp_size                       # Q_d
reach    = (cp_rank + 1) * chunk                          # K_d
q        = Frame(row.q.cu_seqlens // cp_size, chunk)
swa_k    = Frame(row.q.cu_seqlens, reach)
cmp_k    = None if row.cmp_k is None else Frame(row.cmp_k.cu_seqlens, reach // ratio,
                                               residual=None if ratio == 1 else reach % ratio)
key_rows   = torch.arange(cp_rank * flatten_len, (cp_rank + 1) * flatten_len)
position_k = doc_relative_positions(row.q.cu_seqlens, key_rows)
```

其中 `xchg` 的移动表来自 `ExchangeMetadata.from_permutation`（与 V4 是同一份实现），`position_k` 来自
`doc_relative_positions`（纯 device 算术，与本 rank flatten 段真实所在的行同一组数；degree=1 时它就是
输入的 `positions`）。

并检查 `sum(chunk) == batch_size * seq_len`（边界描述的必须是调用方切出来的那条行）。frame 变
则记录必须重算：entry 用 `AscV4Metadata.build_aicpu_metadata(local, geometry=...)`（读 entry 自己的
`q` / `swa_k` / `cmp_k`）重建 `sparse_attn` 与 `lightning_indexer` 两组 `*_metadata`；
`reach` 这类"本 rank 的数"由 python int 的 `cp_rank` 参与张量运算（rank 是 host 值，见 §1.2）。

V4.1 的 `compress_ratio` 可以是 0（window-only，没有任何压缩 frame）、1（整分辨率第二路 KV）
或 2（压缩共享 KV）；`cmp_k` 只在 `ratio > 0` 时存在，`ratio == 1` 的 `residual` 为 `None`
（轴没有折叠，内核对非零值报错）。

### 5.3 压缩器与 sharding

V4.1 的 `Compressor`（`torchtitan_npu/models/deepseek_v4_1/compressor.py`）没有 CP 分支：source
layer 在本地 flatten 段上做 `ratio` 折叠（ratio==1 是普通投影），因为文档已按 `cp × ratio` 对齐，
本 rank 的 flatten 段起点就是组边界，`C/ratio` 是整数块。压缩输出（`[B, C/ratio, D]`）在 sparse core
的边界被声明式 `cp: S(1) -> R` gather 成整行；`cmp_k.seqused = reach // ratio` 让内核只读本 rank
之前的块。ratio-4 那种跨 flatten 段的 carry 在 V4.1 不存在，因此没有 `CarryExchange`。

`torchtitan_npu/models/deepseek_v4_1/sharding.py` 声明 CP 的输入/输出放置：

- sparse core：`q` 保持 `S(2)`（本 rank chunk），`swa_k` / `cmp_k` 在消费侧为整行
  （`in_src_shardings` 是 replicated、`in_dst_shardings` 是行放置，即 `S(1) -> R`），
  `topk_indices` 与 `attn_sink` replicated；
- selector：`idx_q` / `weights` replicated，`idx_k` 做同样的 `S(1) -> R`（选择器用整行 index keys
  给本 rank 的 chunk 打分）。

### 5.4 indexer teacher 的归一化

V4.1 的 selector 在反向把 teacher 除以 `Selector.num_global_queries`：本 step 全部 accumulation
microbatch 的 query token 数（在 CP 切分之前统计，DP 维再求和）。V4 的融合 selector 反向则除以
`metadata.swa_k.cu_seqlens[-1]`（整行 token 数，即所有 rank 的 chunk 之和），两者是同一个量的两种
来源，都是为了避免 CP/DP 下参数梯度被 rank-local 分母放大。

### 5.5 拒绝 CP 的路径

`DeepSeekV4Metadata.prepare_cp_metadata` 与 `DeepSeekV41Metadata.prepare_cp_metadata` 拒绝模型目录
的参考内核：它们只描述整行的 metadata，CP 下没有一份能交给层去读，所以在任何层执行之前报错，不会
静默地拿着全局 metadata 跑 rank-local 行。融合与量化后端各自覆盖了这个 hook，重建本 rank 的 frames
与 AI_CPU metadata。

Engram 不在这一列。它的 token history 要的是每行**在本文档内**的前驱，doc-chunk 下这些前驱不是
全局行的一段连续区间（`[lo - H, lo)` 是别的文档的尾巴），一个窗口也说不出每行各自的前驱。所以
`DeepSeekV41Model.preprocess_inputs` 在切分之前把整行的前驱算成 `[B, C, H]`（第 `d - 1` 项是这一行
往前第 `d` 个 token），它读的就是 shard 刚用过的那个置换（`_cp_shard` 把它留在 decoder 上），因此不需要任何跨 rank
通信；文档起点那几行没有前驱，仍然由 entry 的 `positions`（`positions >= distance`）掩掉。engram 的
`_shift_tokens` 因此有两种 history 契约：`[B, H]`（连续切分给的一个窗口）与 `[B, L, H]`（按行）。
按行那一支直接取 `token_history[:, :, distance - 1]` 这一片，**不**先分配一个填满 pad 的缓冲再整片写回：
整片写回读起来像一次拷贝，但在分片的派发下写进了错的那一片（A5 单卡 cp=2 实测里 rank 0 有 970/1024
行拿到了 rank 1 的 token，step 1 的 `loss` 从 11.94468 变到 11.94288，而两个 degree 各自仍是可复现的）。
`deepseek_v4_1_stripped` 现在挂 Engram（`_stripped` 结尾走 `_attach_engram`，只挂一层
`layer_ids=(1,)`），所以这条取数约束由单卡对拍与 910 的用例共同覆盖。

## 6. 组件

| 文件 | 职责 |
| --- | --- |
| `torchtitan_npu/models/deepseek_v4/cp_attention.py` | `ExchangeMetadata`（`from_permutation` / `apply`）/ `doc_relative_positions` / `restore_key_stream`：shard 置换的逆与 key 侧的流 |
| `torchtitan_npu/models/deepseek_v4/cp_layout.py` | `PerDocCPLoadBalancer`（边界、度数与 `_document_chunks`）/ `DocChunkCPLoadBalancer`（`generate_permutation`）：文档布局；模型 import 时注册为 `doc_chunk` |
| `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py` | `ContextParallelLoadBalancer`（上游 #4617 的抽象）、布局名注册表、`build_load_balancer`（名字 + 本 batch 的 metadata → 配置对象）、`shard_tensors`：把调用方算好的置换施加到切分上 |
| `torchtitan_npu/models/deepseek_v4/compile_policy.py` | `skip_compile_under_cp`：arch22 + CP 时把编译关掉并告警（见 §4.2） |
| `torchtitan_npu/models/deepseek_v4/metadata.py` | `KernelFrame`（两栈共用的 frame 原语） |
| `torchtitan_npu/models/deepseek_v4/model.py` | `DeepSeekV4Model.preprocess_inputs`：MTP 深度与 batch 走同一次 `cp_shard`（布局由 seam 施加，模型不再搬 batch） |
| `torchtitan_npu/models/deepseek_v4/metadata.py` | `DeepSeekV4Metadata(ratio, q, swa_k, cmp_k, position_k, xchg)` / `KernelFrame` / `DeepSeekV4Metadata.from_positions` / `doc_chunk_entry`（实例方法，V4.1 继承） |
| `torchtitan_npu/models/deepseek_v4/attention.py` | 查询与 key 都在 doc-chunk 行上投影；swa 键在核心前用 `restore_key_stream` 恢复、压缩器内部恢复+carry、selector + 内层核心直调（无 wrapper）、输出不 unpermute；参考内核读 `q` / `cmp_k` frame 与预计算的 HCA 选择 |
| `torchtitan_npu/models/deepseek_v4/cp_compressor.py` | `CarryExchange` / `CarryStream`：ratio-4 carry 的 ring（只换 tokens；位置由 `doc_relative_positions` 算出）、增强 `cu_seqlens` 与 `trim`；cp=1 为 no-op |
| `torchtitan_npu/models/deepseek_v4/compressor.py` | 压缩器（CP/非 CP 同一实例）：`forward` 用 `restore_key_stream(attention_metadata.cp, …)` 恢复窗口、ratio-4 时内部 carry 并裁掉自己那块；`Indexer` 复用它，`idx_q` 在 chunk 行、`idx_k` 是行内的池化块 |
| `torchtitan_npu/models/deepseek_v4/sharding.py` | `swa_k` / `cmp_k` / `idx_k` 的 `cp: S(1) -> R` 声明 |
| `torchtitan_npu/override/deepseek_v4/asc_metadata.py` | `AicpuMetadata` 等类型 / `AscV4Metadata`（`aicpu_metadata` + `a3_workaround`）/ `build_aicpu_metadata`（组合两个 kernel 模块的 AI_CPU metadata）/ `prepare_cp_metadata`（`doc_chunk_entry` + `build_aicpu_metadata`） |
| `torchtitan_npu/override/deepseek_v4/sparse_attn/ascendc.py` | `kernel_options` / `kernel_aicpu_metadata`（SMLA/SMLAG 的实参与记录，同处一模块）/ `_SparseMLA` / `AscSparseMLAMixin`（两栈共用的端口：行/TND 翻译、arch22 上 key 压缩） |
| `torchtitan_npu/override/deepseek_v4/lightning_indexer/ascendc.py` | `kernel_options` / `kernel_aicpu_metadata`（indexer 的实参与 metadata）/ `_LightningIndexerTND` / `AscIndexerMixin`（两栈共用的 indexer 端口，V4.1 形状：`candidates_BL1C` 与三元返回） |
| `torchtitan_npu/override/deepseek_v4/lightning_indexer/pypto.py` | PyPTO 版 selector：独立实现同一契约（输入准备、排序、teacher 边），只替换 LI/SLIKG kernel 对 |
| `torchtitan_npu/override/deepseek_v4/compressor/ascendc.py` | CANN 融合压缩器（无 carry、无 CP 分支；`_state_inputs` 按 stream 形状内部构造零 tiling 表） |
| `torchtitan_npu/models/deepseek_v4_1/attention.py` | V4.1 查询交换、本地 KV/压缩、输出 `unpermute` |
| `torchtitan_npu/models/deepseek_v4_1/compressor.py` | V4.1 本地池化压缩器（CP-agnostic，无 carry） |
| `torchtitan_npu/models/deepseek_v4_1/sharding.py` | V4.1 sparse core / selector 的 `S(1) -> R` 声明 |
| `torchtitan_npu/models/deepseek_v4_1/model.py` | `DeepSeekV41Metadata` 与 `build_v41_frames`（整行边界与三帧）；`prepare_cp_metadata` 拒绝 CP；`preprocess_inputs` 按行构造 Engram 的 token history；teacher 成对启用检查 |
| `torchtitan_npu/override/deepseek_v4_1/sparse_attn/ascendc.py` | `AscV41Metadata`（`AscV4Metadata` 的子类，只改 `indexer_ratios` 与 selector 记录类型）/ `AscV41SparseAttention`（共用 `AscSparseMLAMixin`） |
| `torchtitan_npu/override/deepseek_v4_1/lightning_indexer/ascendc.py` | V4.1 BF16 selector（整行 index keys × 本 rank chunk，参数与记录都取自共用模块） |

## 7. 验证

### 7.1 层次与判据

CP 的验收分三层，越往下越严格：

1. **CPU 契约**：布局的纯函数（`ExchangeMetadata.from_permutation` 的 send/recv split 在 peers 之间一致、移动复现
   doc-chunk 行集合、`apply` 与切分置换互为逆、跨两个 flatten 段的 chunk、`doc_relative_positions`、
   未对齐文档被拒绝、`cp=1` 恒等）、`from_positions` 的边界与 `cu_seqlens` 一致、frame 坐标
   与 `seqused` 锚点、rank-local 前缀池、metadata 重建与拒绝路径。这一层不看数值，只看"谁被放到了
   哪"。
2. **多 rank 集合通信**（`tests/unit_tests/models/deepseek_v4/test_cp_attention_distributed.py`）：两个 gloo
   rank 上真实跑 `all_to_all_single`，验移动、逆向、`position_k` 以及 `apply` 的梯度等于反向
   交换。
3. **单机数值**（A5 单卡 + `--comm.mode local_tensor` 模拟两个 rank）：CP=1 与 CP=2 两个运行比
   logits、`loss` 与梯度。

判据是分层的，因为跨 rank 归约改变了加法树：

| 量 | 判据 | 原因 |
| --- | --- | --- |
| step 1 的输出 logits | **逐位** | 计算图逐算子相同，只是行顺序与按 rank 分段 |
| step 1 的 `loss` | **逐位** | 同上（`loss` 已按 token 加权，跨 rank 求和顺序不同时仍应逐位，因为每个 rank 的 `loss` 是全体 query 的确定性函数） |
| step 1 的梯度 | 相对误差 ~1e-5 | 每个参数分片的梯度是 `cp` 个 partial 之和："两份 fp32 partial 相加"与"单 rank 逐行累加"不是同一棵加法树，而 partial 的量级远大于抵消后的总和 |
| step 2 的 `loss` / `grad_norm` | 同上 | 由上面的梯度差传播而来 |

因此判断**一次改动是否中性**，看 step 1 的 logits 与 `loss`（逐位）；梯度与 step 2 只能给量级
结论，不能作为逐位判据。

### 7.2 取数要点（每一条都曾把一个对比做废）

- **权重必须对齐**：两侧加载同一份 checkpoint（`--checkpoint.enable` + `--checkpoint.load-only`
  + `--checkpoint.folder=<绝对目录>`，且日志里要看到 `Loading the checkpoint from ...`）。只给
  `load_only` 不给 `enable` 时 checkpoint 管理器整个不启用，日志里没有加载行——"看起来对齐"的对比
  是无效的。
- **同一 seed 重新初始化不算对齐**：模型初始化依赖 world size（分片张量的抽样位置不同），world=1
  与 world=2 拿到的是不同权重。
- **单卡模拟器还要手工镜像权重**：模拟器按 Philox 状态的*元素偏移*摆放每个 rank，而 NPU 生成器
  每次 kernel 启动只把 offset 推进 4（与张量大小无关），所以同一 seed 下 `Shard(0)` 的两个 rank
  不是单 rank 那次抽样的并集（`Replicate` 是）。做法是由一侧导出完整张量、按 shard 规则切分后写回。
- **`NGPU` 必须等于 `CP`**：CP=1 的对照必须 `NGPU=1`，否则 `cp=1` 会落到 `dp_shard=2`，两侧看的
  不是同一条样本、也不是同一份权重。
- **`--dataloader.per_doc_alignment` 两侧相同**：示例脚本按 `lcm × CP` 写，跨 degree 对比时要显式
  写同一个值（否则文档边界不同，比的是两份数据）。
- **A5 上需要 `torch_npu.npu.set_deterministic_level(3)`，两侧同时打开**：不加时同一 degree 重复跑
  是逐位可复现的，但不同 degree 会落到不同的 kernel 实现上（形状相关的实现选择），logits 差异可达
  1e-1~1e0；打开后才有逐位结论。level 3 也会改变前向数值本身，所以只能做**同 level** 的对比。
  arch22（910）没有这一档，逐位判据只在 A5 上取。
- **logits 的取法**：把模型输出按 rank dump，用 entry 的 `restore.exchange` 把 doc-chunk 行还原成
  flatten 段的行；CP=2 的各 rank 按 rank 顺序拼起来应与 CP=1 逐位相同。
- **一次改动是否中性，用同一棵树重跑作分母**：先确认"改动前后、同一配置"逐位相同，再谈跨 degree。
- **V4 的 ratio-4 carry 需要模拟器友好的 rank 算术**：单卡模拟器把 `cp_rank` 表示成符号 int，而 carry 的三处主机算术（`_ring_split(peer=rank±1)`、`first = rank * flatten_len - carried_rows`、按 `x_carry.size(1)` 切片）会把它当成普通整数。不打补丁时 CP=2 与 CP=1 只在一个压缩块（rank 1 的首块，carry 接缝）上不同，机制上等价于 ring 到了错误的 rank。V4.1（ratio 0/1/2）不走 carry，无需这类补丁。
- **V4.1 的融合路径目前只在 A5 上可跑**：arch22 的 `aclnnSparseFlashMlaMetadata` 在 `has_cmp_kv=True` 且 `cmp_topk≠0`（CSA 带 `cmp_sparse_indices`）时要求 `cmp_ratio == 4`，而 V4.1 的 CSA2 是 ratio 2，因此在 910 上无法表达该几何（该限制在 `cann_ops_transformer` 内，换 CANN 包无效；`ref` 同样失败）。

### 7.3 910（arch22）上的补充约定

- 不开 `set_deterministic_level`：arch22 上没有这一档。V4 在 arch22 上跨 degree 的前向**不**逐位
  （既有的 kernel/归约选择差异，改动前就存在）；V4.1 在同配置下实测跨 degree 仍逐位，但那是这一
  套内核在该形态下的结果，不能当成 arch22 的普遍结论。因此 910 上的主要判据仍是"同一 degree、
  改动前后逐位相同"，跨 degree 的逐位判据以 A5 上的结果为准。
- 长跑（50 step）用来确认不漂移：两个 degree 的 `loss` 曲线应保持常数级间距（该间距来自上面的
  实现选择差异），不随 step 放大。
- `--compile.enable` 在 CP=1 下照旧编译（开启后 step 1 的数值应与 eager 一致）；CP>1 下跳过编译并
  告警，跑 eager（§4.2）。

### 7.4 Engram 的取数方式

挂 Engram 的 `deepseek_v4_1_debugmodel_text` 过不了融合内核（`SparseFlashMla` 要求 q head dim 为
512、RoPE 表 512），所以这一路用 `deepseek_v4_1_stripped`：released 几何 + debug 的 Engram 表（只挂
一层，`layer_ids=(1,)`）。它同时是两个 degree 的精度基线，`examples/deepseek_v4_1/debug/
deepseek_v4_1_stripped_2p_cp2_a5.sh` 就是这条配方（NGPU=2、CP=2、融合算子、完全重计算）。

拼好之后的判据与其它用例相同，两条都实测过（A5 单卡，level 3，同一份镜像权重）：

- **整模型**：CP=1 与 CP=2 的 logits 逐位相同（`0/10485760`），step 1 的 `loss` / `grad_norm` 两侧
  都是 `11.94468` / `18.9753`；各自重跑一次也逐位复现。
- **单模块**：把 Engram 模块单独拿出来，用整行前向的结果当基准，再按各 rank 的行 + 按行 history 跑
  一遍，输出同样逐位相同——文档长度不等、`cp=4`、`history` 2/3 都成立。换成"连续一个窗口"的 history
  就不逐位，这正是 §5.5 那条按行 history 存在的理由。

镜像权重、`set_deterministic_level(3)` 两侧同开这两条对 Engram 一样成立；Engram 的表是
`LocalTensor`（没有 `device_mesh`），脚手架在 `_piece` 里要把它当作未切分处理，否则 CP=2 加载镜像
时会崩在取 `device_mesh` 上。

## 8. 相关文档

- [DeepSeek-V4 TND 适配](deepseek_v4_tnd.md)
- [融合算子使用与开发](fused_ops.md)
- [数值对齐](../developer_guides/numerical_alignment.md)
- [MTP](mtp.md)
