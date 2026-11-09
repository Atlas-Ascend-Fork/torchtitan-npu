# Dense 副本与专家分片

DeepSeek-V4 和 DeepSeek-V4.1 可在保留 Dense 参数副本的同时，将路由专家权重分片到全部专家数据并行卡上。
使用 `--parallelism.disable-expert-replication` 启用，默认关闭，不改变现有切分方式。

## 配置示例

单机 8 卡，Dense 使用 4 分片、2 副本，路由专家使用 EP4、FSDP2，不保留额外副本：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --parallelism.data-parallel-shard-degree 4 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.expert-parallel-degree 4 \
  --parallelism.disable-expert-replication \
  --compile.no-enable \
  --training.global-batch-size 8 \
  --training.steps 10
```

运行前按脚本要求设置 CANN 环境、tokenizer 和 checkpoint 路径，详见
[快速上手](../user-guides/quickstart.md)。序列长度沿用脚本的 4096。

| 参数范围 | 存储布局 |
| --- | --- |
| Attention、Embedding、LM head、router、shared experts 等 Dense 参数 | 4 分片 × 2 副本 |
| routed experts，即路由专家权重 | EP4 × FSDP2，无额外副本 |

8 张卡仍处理 8 路不同数据，GBS 的含义不变。EP 组内的 token 分发和合并方式不变。
关闭开关时，以上配置的专家布局为 EP4、FSDP1、2 副本。

DeepSeek-V4.1 使用相同的并行参数，例如纯文本参考配置：

```bash
CONFIG=deepseek_v4_1_debugmodel_text \
DATASET=c4_test DATASET_PATH=tests/assets/c4_test \
USE_GOLDEN=1 OPTIMIZER_OVERRIDES="" \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh \
  --parallelism.data-parallel-shard-degree 4 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.expert-parallel-degree 4 \
  --parallelism.disable-expert-replication \
  --optimizer.name AdamW --compile.no-enable \
  --training.seq-len 512 --training.global-batch-size 8 --training.steps 10
```

开关仅改变路由专家权重，V4.1 的 Engram 表继续使用其自身的存储和梯度同步策略。

## 原理与边界

启用后，仅在 FSDP 创建专家存储时，将 `dp_replicate` 和 `efsdp` 两个轴共同声明为
专家的分片轴，并取消专家副本轴。有效专家 FSDP 度为
`dp_replicate * dp_shard / EP`，无需额外手工同步或缩放梯度。
FSDP 在计算时仍会按需聚合专家权重；「没有副本」描述的是分片存储布局。

- 当前用于 DeepSeek-V4/V4.1 的常规 Trainer，以及 DeepSeek-V4 Flash 的同步 TorchFT Trainer，要求 `dp_replicate > 1`、`EP > 1`、`TP=CP=PP=1`。
- 不支持同时开启 `--optimizer.muon-enable-hsdp-replica-dedup`；配置阶段会明确报错。
- 支持 `spmd_types` 和 `full_dtensor` 的专家存储接口；`partial_dtensor` 和 GraphTrainer 会明确报错。
- A3、`spmd_types` 下，V4 Flash 43 层 16 专家、4K、Muon 的 eager、MTP、Inductor compile 和 CPU offload 分别完成短步训练；compile/offload 与同初始化、同输入的 eager 基线满足 loss/grad_norm 数值近似阈值。未验证这些功能全部同时开启。
- Muon 要求计算时保持完整专家矩阵；有效专家分片度乘 EP 大于专家数时，FSDP 会选择矩阵维切分 `Shard(1)`，当前 Muon 明确拒绝这种布局。
- V4.1 纯文本、关闭 Engram 的参考路径完成 50 步同权重对比；默认关闭时逐值一致，开启时满足数值近似阈值。Engram 开启另做 3 步完成性验证，不作为其精度对齐证明。
- V4.1 的验证范围为纯文本；多模态、compile、量化、CPU offload 及切换布局后的 checkpoint 续训尚未验证。
- 参数名称和全局形状不变。不同切分方式可能改变初始化与浮点归约顺序，精度比较应加载同一份完整权重，不能只固定 seed。

本功能不改变 attention、路由和专家计算公式，不承诺性能收益。

## TorchFT 外层副本

每个 TorchFT 副本内部可配置 Dense HSDP 和专家完整分片，外层完整模型副本由 TorchFT 管理。例如两个 8 卡副本，每个副本配置 `dp_shard=4`、`dp_replicate=2`、`EP=4` 并开启本功能，得到内部 Dense 4 分片 × 2 副本和专家 EP4 × EFSDP2。

TorchFT 优化器配置使用 `param_groups` 和 optimizer factory，不接受普通 Trainer 的 `--optimizer.name` 参数。使用 Muon 时需要提供对应模型的 Muon 参数组及计算布局配置。

恢复时保留 Muon 的本地状态和计划重建，将跨 rank 布局校验放到同步 quorum 完成后的共同入口执行，避免仅部分 rank 加载状态时通信参与者不齐。TorchFT + Muon 每步增加一次副本内标量同步，只有检测到状态加载才执行计划校验。

已验证两个 8 卡副本、43 层 16 专家、4K、GBS16、Muon 的真实首次状态同步及一步更新。两副本读取不同数据，对应分片的初始化完整摘要一致，逐参数梯度和更新样本一致，完整梯度与更新张量均有限。另有真实两卡 Muon 单 rank 恢复后下一步更新与连续执行参考逐位一致的验证。上述结果不代表磁盘 checkpoint 保存后重启、长期故障恢复或大规模集群验证。
