# 辅助损失框架（AuxLoss）

torchtitan-npu 通过 `torchtitan_npu/patches/torchtitan/models/common/aux_loss.py` backport 上游 torchtitan 的 `AuxLoss`（对应上游 PR [#3864](https://github.com/pytorch/torchtitan/pull/3864)）。辅助损失的梯度按 carrier 注入前向图，归一化分母取本步全局有效 token 数，与主 loss 同一分母，使各并行度下的贡献可比。使用者是 DeepSeek-V3.2 的稀疏 indexer 损失，以及 DeepSeek-V4/V4.1 的 indexer 蒸馏损失。

## DeepSeek-V4/V4.1 的 indexer 蒸馏损失

`torchtitan_npu/models/deepseek_v4/indexer_loss.py` 的 `IndexerLoss` 让稀疏 indexer 学会预测主
注意力在选中压缩槽上的质量。逐层、逐 query 行，记 `p` 为主注意力在选中槽上的 head 平均概率、
`y = softmax(I)` 为 indexer 在同一批槽上的分布，则

```
L = Σ_j p_j · (log(p_j / Z) − log y_j),   Z = Σ_j p_j  (< 1：窗口与 sink 占走了其余质量)
```

即 `Z · D_KL(t ‖ y)`（`t = p / Z`）。这个形式由算子固定：`sparse_flash_mla_grad` 输出未归一化的
`p`，`sparse_lightning_indexer_kl_loss_grad` 求导 `dI = Z·y − p`；若把 `p` 在 Python 里先归一化，
训的就是 `D_KL(t ‖ y)` 这个不同的目标，所以去往算子的 `p` 不做归一化。

两条路径共用一个目标函数，区别只在 teacher 的来源与梯度到达 indexer 的方式：

| 路径 | teacher 来源 | 梯度如何到 indexer |
| --- | --- | --- |
| 参考（`CompressedSparseInnerAttention`、`CompressedSparseInnerAttention2`） | 算子自己的 log-sum-exp（已含 window、选中槽与 sink）反推 | `AuxLoss.inject`，挂在注意力输出这个 carrier 上 |
| 融合（`sparse_attn.asc`） | `sparse_flash_mla_grad` 反向旁路产出的 `p` | 注意力反向里把 `p` 按 `coeff / 本步 token 数` 写到 `topk_scores` 的梯度，由 indexer 的 kernel 求参数梯度 |

`coeff` 只乘梯度、不进日志，与 `AuxLoss` 的其它使用者一致。V4.1 的 Reuse Mode 让同一次选择服务
多层注意力：每层贡献自己的 `p` 与值，共享的 student logits 收到逐消费者之和，日志仍按层平均。

## 指标口径与已知限制

两条路径的累加时机不同：参考路径在前向的 `_accumulate_and_inject` 里累加，融合路径在反向、
kernel 出值后累加。因此

- 激活检查点策略对日志值的影响不同：`FullAC` 在反向重放所属 block 的前向时，参考路径的累加会
  重复执行，其指标高于融合路径，两条路径的指标不能直接比较；本仓 backport 未引入上游用的保留
  区域（`torch_remat`，`recompute=False`），重复累加只在 `FullAC` 下确认过，不能推广到其它 AC
  配置。跨 AC 策略比较日志时应说明两侧策略。
- 一致性判据用 `loss` 与 `grad_norm`：两者不受 AC 重放影响；indexer 指标只在相同 AC 策略下比较。

graph_trainer 图捕获路径另有一条限制：

- 每步归一化分母保存在类级状态上，模型内部读它。`set_step_denominator` 把新值**写进同一个张量**
  而不是换绑新张量：换绑会让被捕获的区域要么每步重编译、要么把首步的值当成常量；同一张量下
  重放的图每步读到更新后的内容，因此这条路径的梯度注入是有效的。
- 指标的累加是前向的原地写副作用，读取它的却是模型之外的优化器 step pre-hook 与
  `MetricsProcessor.log`；整步被捕获后这条写没有被带出图，指标因此不可用。上游没有这个问题，
  差别在两处：上游把累加包在 `torch_remat` 的保留区里（重算跳过、也不被当死代码丢弃），分母也
  走显式前向入参而非类级状态。要修就照这两条走，属于后续工作。
- 因此该路径上用 `loss` 与 `grad_norm` 判断收敛，不要用 indexer 指标。
