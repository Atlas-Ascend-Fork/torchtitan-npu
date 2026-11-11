# DeepSeek-V4 系列上下文并行：切分原理

DeepSeek-V4（DSV4）与 DeepSeek-V4.1（DSV4.1）的 CP 在 forward 前按文档预排序 query，层内保持其布局。
压缩 KV 与 index KV 在消费点 all-gather 成整行，再用 `seqused` 限制因果可见范围；SWA 只交换缺失的
窗口前缀，不做整行 all-gather。布局有两种：

- `doc_chunk`（默认）：rank `r` 取每篇文档的第 `r` 片；
- `doc_zigzag`：每篇文档切 `2 × CP` 片，rank `r` 取第 `r` 片和镜像片，均衡因果负载。

CP 仅适用于 AscendC 融合路径、单个 packed batch row（`B = 1`）和 `spmd_types` 后端。

## 1. 坐标与不变式

设 CP 度 `R`、rank `r`、文档长度 `L_d`、flatten 宽度 `C = N / R`。一行由三个 `KernelFrame` 描述：

| frame | 网格 | `seqused` |
| --- | --- | --- |
| `q` | 每篇文档本地 query 片的前缀和 | 每篇文档的 query 数 |
| `swa_k` | 本地 query 片及前方最多 `W−1` 个 key 的文档边界 | 每个局部窗口的完整长度 |
| `cmp_k` | 整行边界 `// ratio` | `reach_d // ratio`，`residual = reach_d % ratio` |

`position_k` 是**本 rank flatten 段**的文档内位置；查询位置随输入的 `positions` 走，不单列字段。

三条不变式：

- 压缩 KV 与 index KV 保持完整平坦布局；SWA 窗口按 query 的文档顺序拼接。
- `cu_seqlens` 始终保存全局文档边界；consumer 读取其操作数的 `q` / `swa_k` / `cmp_k` frame，
  不再通过 `swa_k` 推断全局网格。
- A5 内核直接读 `seqused`；A3（arch22）内核不读 key 侧 `seqused`，由算子封装把 key 前缀压紧并改写
  `cu_seqlens`，反向再散射回整行（`A3KeyWorkaround`）。

## 2. 切分：布局与置换

`Decoder._cp_shard` 从第一层 attention 的 entry 构造布局对象，拿到 `generate_permutation()` 后由
`shard_tensors` 先重排**复制在每个 rank 的整行**、再均分切片：切分本身零通信，反向是置换的逆。
tokens、labels、positions 各切一次；MTP 的 rolled tokens 先在全局行上生成，再用同一份置换
（`_cp_permutation`）单独 `cp_shard`，所以 future-token shift 不会跨 rank 边界。

每篇文档的对齐值是布局的前提：

- `doc_chunk`：`lcm(正压缩比) × R`（V4 为 `128 × R`，V4.1 为 `2 × R`）；
- `doc_zigzag`：`2 × R × lcm(正压缩比)`（V4 CP2 为 512，V4.1 CP2 为 8）。

每个 chunk 自身也必须是 `ratio` 的整数倍，否则压缩前缀会停在组中间；对齐由
`--dataloader.per_doc_alignment` 在数据侧保证，违反时 metadata 构建或内核网格检查直接报错。

`doc_zigzag` 的物理顺序是「所有文档的 Head | 所有文档的 Tail」，两半等长。rank `r` 的 Head 取第 `r`
片、Tail 取第 `2R-r-1` 片，因此同一 rank 同时拥有短前缀和长前缀的 query：

设 query 片为文档内 `[a, b)`，其 SWA 操作数是 `[max(0, a−W+1), b)`，query 仍靠右对齐。
Head/Tail 的 query 网格相同，SWA 网格各自描述实际窗口；`cmp_k.seqused` 仍由各自的原始可见前缀决定。

## 3. metadata：rank-local frame

- `build_attention_metadata` 在切分前构建全局 frames 与后端记录，每个 geometry 每个 microbatch
  一次，同 key 的层共享。
- `prepare_cp_metadata` 读 `CPShardContext`（CP mesh + 布局名），调用 `rebuild_cp_metadata`。
- `rebuild_cp_metadata` 统一走 `per_doc_cp_metadata(zigzag=…)`：先切本 rank 的 `position_k`，按文档分段从
  permutation 读取实际 query 区间；`doc_zigzag` 保留独立的 `head` / `tail`，最后挂 `xchg.cmp`
  （压缩输入的逆置换）与 `xchg.swa`（SWA 窗口路由）。
- 构建由 `local.head is None` 决定：`doc_chunk` 只重建外层；`doc_zigzag` 只重建 `head` / `tail`
  的 AI_CPU 记录，外层保留全局记录供 KV 生产。

ZigZag 外层保留全局 `q` / `swa_k` / `cmp_k`；`head` / `tail` 的 `position_k` 都是
本 rank 的 KV slice。量化 adaptor 同样按每个 entry 的 frames 重建自己的记录，候选长度由
`position_k` 的形状按布局折半得到，不读 frame 的内容。

## 4. KV 恢复与通信

`xchg` 统一承载 CP 交换元数据，非 CP 时为 `None`。`xchg.cmp` 描述压缩器所需的连续全局 token 区间，
`xchg.swa` 描述 Head/Tail 所需的窗口。两者共用 `_xchg_metadata`：输入 token 的物理位置及每个 rank 的需求，
输出发送索引、收发数量和按来源 rank 排列的接收 token；它不解释布局名称。

`DocumentRanges` 保存每篇文档的全局起止位置，由 permutation 和 query 分段构造。query frame、SWA frame 与
窗口交换使用同一份区间，因此 `WindowMetadata.build` 无需 `zigzag` 参数。Head/Tail 即使相邻也分别保留。
`send_split` / `recv_split` 直接对应 collective 的输入 / 输出分段；压缩器用 `recv_index` 恢复连续顺序，
SWA 的 `head_index` / `tail_index` 从「本地 key + 接收 key」组装窗口。metadata 不持有进程组或执行通信。

`ContextParallelConverter` 按 pinned 上游的 `ModelConfigConverter` 接口，在模型配置阶段选择独立的
CP attention/compressor。普通模型不持有 CP 进程组或通信流程；已选的量化 compressor 子类通过继承保留，
参数模块不增加包装层，因此参数 FQN 不变。inner attention/indexer 仍使用 override 选中的后端。

- `ContextParallelCompressor` 在池化前通过 `_restore_flat_input` 还原布局；ratio-4 再通过 `_prepend_carry` 交换前驱块，
  池化后裁掉 carry 输出。其他压缩比只裁剪本地文档网格。V4.1 的 CP compressor 只恢复输入布局。
- `CPAttention._exchange_window` 按 `WindowMetadata` 一次 all-to-all 获取远端 key；本地 key 直接复用。
  同一 peer 请求的重复 key 只发送一次，Head/Tail 的重复使用在反向累加，短文档的窗口可跨多个 rank。
- `CPAttention` / `CPIndexer` 接管 inner 的 sharding 声明，压缩 KV / index KV 在两次调用外各 gather 一次；
  `CPAttention` 在内部组装 SWA 窗口，再按 Head/Tail 的 frames 调用后端并拼接输出。

这与上游 [SelectiveGather CP 草案 #5746](https://github.com/pytorch/torchtitan/pull/5746) 的
「attention 持有窗口交换、路由与执行分离」一致。我们的 per-document 置换随 batch 改变，
因此路由随 metadata 重建；不沿用其固定全局分片缓存，也不依赖 CUDA copy-engine 后端。

与上游 `AllGatherKVVarlen` 一样，CP 模块拥有通信与两次调用，后端只消费 metadata。区别在于本实现
按每篇文档切 query，并在压缩前恢复平坦输入；上游沿全局序列切分、全量 gather 后裁剪 KV。
SWA 前缀交换是本仓扩展。

布局还原、carry 与窗口交换各自独立：前者是置换，carry 只补一个压缩块，窗口交换允许重复使用 key。
平坦化与窗口交换共用路由构造；carry 直接使用邻居关系。SWA 无需按窗口宽度额外 padding。

arch22 + CP 下 `A3KeyWorkaround` 的压实行数是 device 值，与 `fullgraph=True` 的编译不兼容；栈在
`parallelize` 里按 `is_a5() or cp <= 1` 关闭编译（`compile_policy.py`），退化为 eager 并告警。

## 5. 启用

```bash
--parallelism.context-parallel-degree 2 \
--parallelism.context-parallel-load-balancer doc_zigzag \
--parallelism.spmd-backend spmd_types \
--dataloader.per_doc_alignment 512
```

布局必须显式命名；上游默认的 `headtail` 不在本仓的布局注册表里，融合 metadata 的 query 区间按
`doc_chunk` / `doc_zigzag` 的块数读置换，其他名字不会得到正确的 rank-local frame。CP>1 仅支持
`local_batch_size = 1`，DSA 路径 TP 固定为 1。

## 6. 验证方法

CP 的验收分三层，越往下越严格：

1. CPU 契约：布局的纯函数（置换覆盖、send/recv 一致、`apply` 为逆、frame 与 `seqused`、拒绝路径）；
2. 双 rank Gloo：真实 `all_to_all_single`，验移动、逆向与梯度；
3. 单机数值：A5 单卡 + `--comm.mode local_tensor` 模拟 CP2，比较 CP1/CP2 的 logits、loss 与梯度。

判据分层：首步 logits 与 loss 用逐位；梯度与 step 2 只给量级，因为 CP 轴的 reduce-scatter 改变了加法树。

取数要点：

- 两侧加载同一份 checkpoint（`--checkpoint.enable` + `--checkpoint.load-only`），日志中确认加载行；
  只给 `load_only` 不给 `enable` 等于没对齐。
- 同一 seed 重新初始化不算对齐：world size 会改变分片抽样位置；单卡模拟器还要按 shard 规则手工镜像权重。
- `NGPU` 必须等于 `CP`；两侧 `--dataloader.per_doc_alignment` 相同。
- A5 两侧同时打开 `set_deterministic_level(3)`；不加时不同 degree 可能落到不同 kernel 实现。
- logits 按 rank dump，用 entry 的 `restore` 还原成 flatten 行后拼接比较。
- 判断一次改动是否中性，先用同一棵树确认「改动前后、同一配置」逐位相同，再谈跨 degree。

arch22（910）没有确定性等级：跨 degree 不作为逐位判据，主要看「同一 degree、改动前后逐位」以及长跑 loss
间距是否稳定；CP>1 时 `--compile.enable` 会被 `compile_policy` 关闭并告警。

Engram 的取数用 `deepseek_v4_1_stripped`（released 几何 + debug Engram 表，只挂一层），同样需要镜像
权重与两侧 level 3；逐行 history 由 shard 置换生成，跨 rank 不通信。

## 7. 代码入口

| 文件 | 职责 |
| --- | --- |
| `models/deepseek_v4/cp_layout.py` | `doc_chunk` / `doc_zigzag` 的 LoadBalancer、置换与 `DocumentRanges` |
| `models/deepseek_v4/metadata.py` | `KernelFrame`、`CPExchanges`（`xchg.cmp` / `xchg.swa`）、`DeepSeekV4Metadata`、`rebuild_cp_metadata` |
| `models/deepseek_v4/context_parallel.py` | CPAttention 组装 SWA 窗口；CPCompressor 还原 slab 与 ratio-4 carry；配置转换 |
| `models/deepseek_v4/compile_policy.py` | arch22 + CP 时关闭编译 |
| `override/deepseek_v4*/` | 融合后端、量化 adaptor、A3 前缀压紧 |

## 8. 相关文档

- [DeepSeek-V4 TND 适配](../feature_guides/deepseek_v4_tnd.md)
- [融合算子使用与开发](../feature_guides/fused_ops.md)
- [数值对齐](numerical_alignment.md)
- [MTP](../feature_guides/mtp.md)
