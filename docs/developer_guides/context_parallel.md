# DeepSeek-V4 系列上下文并行：切分原理

DeepSeek-V4（DSV4）与 DeepSeek-V4.1（DSV4.1）的 CP 是「查询按文档切分、KV 保持整行」的静态切分：
切分在 forward 之前由布局置换完成，层内不移动 query；KV 与压缩 KV 在消费点声明式 all-gather 成整行，
再用 frame 的 `seqused` 把可见范围限制在本 rank 的查询之前。布局有两种：

- `doc_chunk`（默认）：rank `r` 取每篇文档的第 `r` 片；
- `doc_zigzag`：每篇文档切 `2 × CP` 片，rank `r` 取第 `r` 片和镜像片，均衡因果负载。

CP 仅适用于 AscendC 融合路径、单个 packed batch row（`B = 1`）和 `spmd_types` 后端。

## 1. 坐标与不变式

设 CP 度 `R`、rank `r`、文档长度 `L_d`、flatten 宽度 `C = N / R`。一行由三个 `KernelFrame` 描述：

| frame | 网格 | `seqused` |
| --- | --- | --- |
| `q` | 每篇文档本地 query 片的前缀和 | 每篇文档的 query 数 |
| `swa_k` | 整行文档边界 | 本 rank 的可见前缀 `reach_d` |
| `cmp_k` | 整行边界 `// ratio` | `reach_d // ratio`，`residual = reach_d % ratio` |

`position_k` 是**本 rank flatten 段**的文档内位置；查询位置随输入的 `positions` 走，不单列字段。

三条不变式：

- KV 与压缩 KV 永远是整行；rank 只持有自己的 query 片和 KV 的平坦切片。
- consumer 只读 entry 的 frame：行网格 `swa_k.cu_seqlens`、查询网格 `q.cu_seqlens`、压缩网格
  `cmp_k.cu_seqlens`；没有 `cu_seq_q` / `cu_seq_k` 别名。
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

| entry | `q` 网格 | KV 网格 | `swa_k.seqused` |
| --- | --- | --- | --- |
| 外层 | 全局文档网格（供 KV 生产） | 全局 | 全部 `L_d` |
| `head` | 每文档 `L_d/(2R)` 的前缀和 | 全局 | `(r+1)·L_d/(2R)` |
| `tail` | 与 `head` 相同 | 全局 | `(2R-r)·L_d/(2R)` |

## 3. metadata：rank-local frame

- `build_attention_metadata` 在切分前构建全局 frames 与后端记录，每个 geometry 每个 microbatch
  一次，同 key 的层共享。
- `prepare_cp_metadata` 读 `CPShardContext`（CP mesh + 布局名），调用 `rebuild_cp_metadata`。
- `rebuild_cp_metadata` 统一走 `per_doc_cp_metadata(zigzag=…)`：先切本 rank 的 `position_k`，再按块数
  生成 query 片；`doc_zigzag` 时追加 `head` / `tail` 两个子 entry，最后挂 `xchg`
  （`ExchangeMetadata.from_permutation` 的逆置换）。
- 构建由 `local.head is None` 决定：`doc_chunk` 只重建外层；`doc_zigzag` 只重建 `head` / `tail`
  的 AI_CPU 记录，外层保留全局记录供 KV 生产。

外层保留全局 `q` / `swa_k` / `cmp_k` 与整行 `xchg` 的语义；`head` / `tail` 的 `position_k` 都是
本 rank 的 KV slice。量化 adaptor 同样按每个 entry 的 frames 重建自己的记录，候选长度由
`position_k` 的形状按布局折半得到，不读 frame 的内容。

## 4. KV 恢复与通信

`ExchangeMetadata.from_permutation` 求 shard 置换的逆：`send_index` / `send_split` 是本 rank flatten
段每行要去的 peer，`recv_split` 是应从每个 peer 收到的行数。`doc_zigzag` 持有的是两段文档列表，
另有 `recv_index` 在 all-to-all 前把它们排回平坦 token 序；接收侧仍按 `send_index` scatter。

`restore_key_stream` 在每个需要「本 rank 窗口」的消费方内部恢复，`cp=None` 时恒等：

- 压缩器在池化前恢复；
- 核心在取 `swa_k` 前恢复；indexer 的 `idx_k` 走自己的压缩器；
- V4 的 ratio-4 重叠前驱由 `CarryExchange` 交换、prepend，池化后裁掉；V4.1 对齐到组边界，
  压缩器天然 CP-agnostic，没有 carry。

KV all-gather 由模型边界的一次声明完成：`cp_dispatch.py` 的 `CPAttention` / `CPIndexer` 包装内层
attention/indexer，接管 `_sharding_config`，在 Head/Tail 两次调用外只 gather 一次；AscendC、
PyPTO 与量化后端只跑单块、不感知 CP。`head is None` 时回退为一次直接调用。包装器只包住无参数的
inner 子树，`state_dict` 键与 CP=1 一致。

arch22 + CP 下 `A3KeyWorkaround` 的压实行数是 device 值，与 `fullgraph=True` 的编译不兼容；栈在
`parallelize` 里按 `is_a5() or cp <= 1` 关闭编译（`compile_policy.py`），退化为 eager 并告警。

## 5. 启用

```bash
--parallelism.context-parallel-degree 2 \
--parallelism.context-parallel-load-balancer doc_zigzag \
--parallelism.spmd-backend spmd_types \
--dataloader.per_doc_alignment 512
```

布局必须显式命名；上游默认的 `headtail` 不在本仓的布局注册表里，融合 metadata 在 CP 下没有它的
rank-local 形式。CP>1 仅支持 `local_batch_size = 1`，DSA 路径 TP 固定为 1。

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
| `models/deepseek_v4/cp_layout.py` | `doc_chunk` / `doc_zigzag` 的 LoadBalancer 与置换 |
| `models/deepseek_v4/metadata.py` | `KernelFrame`、`DeepSeekV4Metadata`、`rebuild_cp_metadata` / `per_doc_cp_metadata` |
| `models/deepseek_v4/cp_attention.py` | `ExchangeMetadata`（置换求逆）与 `restore_key_stream` |
| `models/deepseek_v4/cp_dispatch.py` | 模型级 `CPAttention` / `CPIndexer` 包装器与配置转换 |
| `models/deepseek_v4/cp_compressor.py` | V4 ratio-4 的 `CarryExchange` |
| `models/deepseek_v4/compile_policy.py` | arch22 + CP 时关闭编译 |
| `override/deepseek_v4*/` | 融合后端、量化 adaptor、A3 前缀压紧 |

## 8. 相关文档

- [DeepSeek-V4 TND 适配](../feature_guides/deepseek_v4_tnd.md)
- [融合算子使用与开发](../feature_guides/fused_ops.md)
- [数值对齐](numerical_alignment.md)
- [MTP](../feature_guides/mtp.md)
