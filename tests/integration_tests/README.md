# TorchTitan-NPU 集成测试指南

本目录遵循 TorchTitan 的 `tests/integration_tests` 架构，负责维护集成测试（Integration Tests）定义、执行入口、数值精度对齐校验以及集群 Nightly CI 调度规范。

根据运行环境与执行规模的不同，仓内集成测试分为两种核心形态：
1. **流水线典型用例冒烟测试（A3 4die环境）**：面向 PR 门禁与日常开发快速回归，在配备 1~4 卡（4-die）的开发机或流水线容器内运行，通过仓内集成的并行 Runner 执行典型短步数用例，支持 Deterministic 数值 Loss 精确逐值比对与核心算子冒烟；
2. **集群 Nightly 全量用例测试**：面向真实的 A3 / A5 物理多机集群（单机 8P / 单机 16P / 双机 16P / 八机 128P），覆盖生产级完整规格、跨机分布式通信拓扑与真实权重资产，通过内网 Lite Actions 调度系统排队执行。

---

## 一、流水线典型用例冒烟测试（A3 4die环境）

### 1.1 运行入口与基本命令

#### 脚本入口
CI 流水线主要通过以下脚本一键拉起集成测试：

```bash
.ci/integration_test.sh
```

该脚本会自动准备 CANN 环境变量与运行解释器环境。

#### Python 原生 Runner 入口
在开发机上手动运行测试时，可直接调用 Python 模块：

```bash
python -m tests.integration_tests.run_tests \
  ./test_reports/integration \
  --test_suite models \
  --ngpu 4
```

> **参数说明**：
> - `./test_reports/integration`：**必填**测试输出目录，运行前必须确保为空；
> - `--test_suite`：指定要执行的测试套件（如 `models`、`deepseek_v4_checkpoint` 等）；
> - `--ngpu`：指定单次测试使用的总卡数；
> - `--test_name`：（可选）仅执行该套件内的单条特定用例；
> - `--no-parallel`：（可选）强制串行执行，默认会开启多用例并行。

---

### 1.2 冒烟测试全景矩阵（1~4 卡）

4die 冒烟测试及专项验证包含以下 Suite：
* **`models`**：DeepSeek-V4、DeepSeek-V4.1、DeepSeek-V3.2 多卡基础训练冒烟与 Loss 对齐校验；
* **`deepseek_v4_checkpoint`**：DeepSeek-V4 分布式断点续训两阶段比对；
* **`deepseek_v4_quantized`**：DeepSeek-V4 Block-FP8 量化（需 Ascend 950 芯片）；
* **`deepseek_v4_1`**：DeepSeek-V4.1 文本和图像 LoRA 训练与 PEFT 导出；同时纳入默认 `models` suite；
* **`deepseek_v4_1_checkpoint`**：DeepSeek-V4.1 AdamW/Muon 断点续训验证，单独运行；
* **`deepseek_v4_1_quantized`**：DeepSeek-V4.1 Block-FP8 LoRA 专项验证，需支持的设备和运行时；
* **`deepseek_v4_1_golden`**：DeepSeek-V4.1 golden 用例，30 步参考 Loss 逐值比对；同时纳入默认 `models` suite；
* **`deepseek_v4_1_fused`**：DeepSeek-V4.1 多模态打包/融合与 Muon 冒烟，需 vision fixture，单独运行；
* **`deepseek_v4_1_engram_hf`**：DeepSeek-V4.1 Engram MXFP8 专项校验与 DCP/HF 权重回环测试。
* **`qwen3_5_quantized`**：Qwen3.5 dense `W4A8_MXFP`/`W8A8_MXFP8` 量化训练冒烟，需支持的 A5 设备、运行时和 TorchAO-NPU。

#### 核心用例清单矩阵

| Case 名称 | 对应模型 | 并行配置 | 卡数 (ngpu) | Check Loss | 核心校验点与说明 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`dsv4_golden_1rank`** | DeepSeek-V4 | 1 Rank 参考配置、Muon | 1 | **是** | 严格 Deterministic 模式，精确逐值比对参考 Loss |
| **`dsv4_golden_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2、Muon | 2 | **是** | 分布式并行环境下的数值 Loss 精确比对 |
| **`dsv41_golden_2p_ep2_fsdp2`** | DeepSeek-V4.1 | Golden recipe、EP2 + FSDP2、Muon、30 steps | 2 | **是** | V4.1 text debug model，30 步参考 Loss 精确逐值比对 |
| **`dsv4_checkpoint_resume_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2 + AdamW NovaSwap | 2 | **动态比对** | 两阶段连续训练 vs 断点恢复训练，逐步比对 Loss 与 Grad Norm |
| **`dsv4_lora_ep2_fsdp2`** | DeepSeek-V4 LoRA | EP2 + FSDP2，训练 LoRA A/B | 2 | 否 | 校验基座参数冻结、Adapter 权重更新及 PEFT 导出完整性 |
| **`dsv41_lora_engram_ep2_fsdp2`** | DeepSeek-V4.1 LoRA | EP2 + FSDP2、FullAC、AdamW | 2 | 否 | 校验基座与 Engram 冻结、adapter 更新及 PEFT 导出 |
| **`dsv41_lora_multimodal_ep2_fsdp2`** | DeepSeek-V4.1 LoRA | EP2 + FSDP2、FullAC、图像 patch + C4 文本 | 2 | 否 | 校验实际 vision forward、vision/Engram 冻结、adapter 更新及 PEFT 导出 |
| **`dsv4_muon_swap_ep2_fsdp2`** | DeepSeek-V4 | NPU 融合算子 + DistMuon NovaSwap | 2 | 否 | NPU 融合算子（RMSNorm/RoPE/SparseAttention）与 Swap 路径冒烟 |
| **`dsv4_anticipatory_recovery`** | DeepSeek-V4 | 1 Rank 原生 Checkpoint 回滚 | 1 | 否 | 一次性扰动训练 Loss，断言 Override 机制与完整恢复周期 |
| **`dsv4_smla_1rank_aot_eager`** | DeepSeek-V4 | 1 Rank，`aot_eager` 编译 | 1 | 否 | SMLA 算子图在单卡下的编译构图与前反向冒烟 |
| **`dsv4_smla_ep2_fsdp2`** | DeepSeek-V4 | EP2 + FSDP2，`aot_eager` | 2 | 否 | SMLA 算子在 EP+FSDP 分布式环境下的执行与 MoE Dispatcher 验证 |
| **`dsv4_smla_cp2_ep2_fsdp2`** | DeepSeek-V4 | CP2 + EP2 + FSDP2 | 4 | 否 | 混合并行（Context Parallel + Expert Parallel）算子冒烟 |
| **`dsv4_mtp_smla_cp2`** | DeepSeek-V4 MTP | CP2（连续切片） | 2 | 否 | 覆盖 MTP forward、chunked loss 及 backward 算子路径 |
| **`dsv3_2_dsa_1rank`** | DeepSeek-V3.2 | 1 Rank，DSA 算子 | 1 | **是** | 校验 DSA attention override 并在 1-rank 下做数值 Loss 比对 |
| **`dsv3_2_dsa_ep2_fsdp2`** | DeepSeek-V3.2 | DSA + EP2/FSDP2 | 2 | **是** | 校验分布式 DSA 算子 30-step 精度对齐 |
| **`dsv3_2_dsa_cp2`** | DeepSeek-V3.2 | DSA + CP2 | 2 | 否 | 校验 DSA 在 CP2 模式下的训练触发与通信 |
| **`dsv4_ema_ep2_fsdp2`** | DeepSeek-V4 | Golden + EP2/FSDP2 + EMA CPU Offload | 2 | 否 | 校验 DCP Checkpoint Metadata 完整包含 `ema_optimizer.*` |

---

### 1.3 核心校验机制与调度原则

#### 1. 精度对齐判定原则：`use_golden` 与 `check_loss`
集成测试严格区分**算子实现选择**与**数值精确对齐**两个独立维度：
* **`use_golden`**：决定模型使用参考实现（DeepSeek-V4 为模型默认内核，即 attn-gym `gather_attn`；V4.1 为同一入口的 reference override 组合），还是使用 SMLA / NPU 专用算子优化 override；
* **`check_loss`**：决定是否启用确定性计算（`--debug.deterministic`）、固定随机种子，并严格读取参考基准 Loss 执行逐值比对。

**判定规则**：
1. 设置 `check_loss=True` 的用例（如 Golden 用例与 DSA 对齐用例），会自动比对 TensorBoard 中的标量 `loss_metrics/global_avg_loss`，**要求执行步骤集合和每一个浮点数值均严格相等（不允许浮点容差）**；
2. 设置 `check_loss=False` 的用例主要用于验证算子图在特定编译模式（如 `aot_eager`）或并行拓扑下的稳定性与执行路径，只要正常完成指定步数且主进程退出码为 0 即判定通过。

#### 2. 核心特色用例设计机制
* **DCP 动态断点续训比对 (`dsv4_checkpoint_resume_ep2_fsdp2`)**：
  采用**两阶段动态自校验**设计，严格验证在分布式 AdamW NovaSwap 场景下的 Checkpoint 保存与恢复一致性：
  - **阶段一（连续基准）**：固定 `seed=42` 开启确定性模式，连续训练 4 步，并在 step 2 保存完整 DCP Checkpoint；
  - **阶段二（续训恢复）**：新进程通过 `--checkpoint.load-step=2` 加载 step 2 权重，继续训练第 3、4 步；
  - **断言比对**：比对两次运行在 TensorBoard 中的 `global_avg_loss` 与 `grad_norm`。第 3、4 步数值必须逐值精确相等，不舍入、不设容差。
  ```bash
  python -m tests.integration_tests.run_tests ./test_reports/lora-resume \
    --test_suite deepseek_v4_checkpoint \
    --test_name dsv4_checkpoint_resume_ep2_fsdp2 \
    --ngpu 2
  ```

* **LoRA 训练与 PEFT 导出 (`dsv4_lora_ep2_fsdp2`)**：
  在模型并行化后冻结 Base 基座权重，仅针对 LoRA A/B 结构训练 2 步：
  - 检查冻结层参数与 routing bias 保持不变；
  - 检查可训练 adapter 权重正常更新；
  - 训练结束后通过真实 Checkpointer 导出 PEFT 权重，自动断言导出文件的 key、shape、数值以及 rank/alpha 配置。

* **Block-FP8 量化场景 (`deepseek_v4_quantized`)**：
  需要在支持 Block-FP8 的 Ascend 950 芯片及对应 CANN 运行时下运行，依赖 `torchao==0.17.0` 与可选扩展包 `experiments/torchao-npu`：
  ```bash
  # 安装扩展包
  pip install ./experiments/torchao-npu

  # 执行量化测试
  python -m tests.integration_tests.run_tests ./test_reports/lora-fp8 \
    --test_suite deepseek_v4_quantized \
    --test_name dsv4_lora_block_fp8_ep2_fsdp2 \
    --ngpu 2
  ```

* **Qwen3.5 `W4A8_MXFP`/`W8A8_MXFP8` 量化场景 (`qwen3_5_quantized`)**：
  ```bash
  pip install ./experiments/torchao-npu

  python -m tests.integration_tests.run_tests ./test_reports/qwen35-w4a8-mxfp \
    --test_suite qwen3_5_quantized \
    --test_name qwen3_5_w4a8_mxfp_1rank \
    --ngpu 1

  python -m tests.integration_tests.run_tests ./test_reports/qwen35-w8a8-mxfp8 \
    --test_suite qwen3_5_quantized \
    --test_name qwen3_5_w8a8_mxfp8_1rank \
    --ngpu 1
  ```

* **Engram HF 专项验证（超节点）**：
  在具备 Engram MXFP8 Host 通信环境的超节点上，验证 V4.1 Engram 参数在原生 DCP 与 FP32 HF 权重格式之间的双向回环保存与恢复一致性：
  ```bash
  HF_ASSETS_PATH=/path/to/v41-tokenizer \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
    --test_suite deepseek_v4_1_engram_hf \
    --ngpu 4 --no-parallel
  ```
  成功执行后控制台输出 `HF_ROUNDTRIP PASS`。

#### 3. 4die 并发调度（`GPUPool`）、实际耗时汇总与通信端口防冲突
当执行包含多个用例的套件时，`run_tests` 默认复用 TorchTitan 的 `GPUPool` 机制并发执行测试，自动按卡号切分：
* **设备池管理**：若环境中预设了 `ASCEND_RT_VISIBLE_DEVICES`，仅在指定集合内切分卡；否则自动通过 `torch.npu.device_count()` 获取总卡数。用例按所需卡数由大到小排序提交调度；
* **超时与安全回收**：支持为用例配置超时时间，超时后会自动向训练进程组发送 `SIGTERM`（限时后 `SIGKILL`），杜绝孤儿 `torchrun` 残留占卡；
* **调度遥测与执行耗时汇总**：
  - 并行调度结束后，Runner 会打印两行调度遥测：池利用率（`[parallel] pool: window/utilization/busy histogram/allocations`）与用例重叠（`[parallel] overlap: sequential vs window、节省时长、并发度直方图`），直观展示任务打包和重叠效果；
  - 全部测试结束时，Runner 最终会输出 `[timing] per-case wall time` 汇总块（串行模式 `--no-parallel` 下同样输出）：逐用例列出用例名称、`ngpu` 和实际运行耗时，并统计所有用例总耗时。即便用例中途失败、超时或提前退出，也会在 `finally` 阶段忠实记录耗时。开发者在 CI 日志末尾一眼就能看出「每个 Case 到底跑了多久」，无需再手动核对日志头尾的时间戳；
* **同机多任务防踩踏重点**：
  > [!IMPORTANT]
  > 如果在同一台机器上手动同时启动多个独立的测试进程或分布式 run，必须为各个运行分配**互不重叠的通信端口范围**：
  > ```bash
  > export HCCL_NPU_SOCKET_PORT_RANGE="60000-60020"  # 任务 A
  > export HCCL_NPU_SOCKET_PORT_RANGE="60021-60040"  # 任务 B
  > ```
  > 否则节点间 HCCL 端口绑定冲突会导致初始化死锁或通信超时。

### 1.4 DeepSeek-V4.1 LoRA checkpoint 验证

`deepseek_v4_1_checkpoint` 单独运行 AdamW adapter-only、Muon 完整 DCP 和 Muon adapter-only 恢复用例。
AdamW 使用固定基座与 adapter-only DCP；Muon adapter-only 保持公共 LoRA 配方的默认保存模式。
完整模型配置见 [A3/A5 示例入口](../../examples/deepseek_v4_1/readme.md)。

#### Muon adapter-only resume

`dsv41_lora_muon_adapter_only_resume_ep2_fsdp2` 保持公共 LoRA 配方的
`periodic_save_adapter_only=True`，使用固定基座、Muon、EP2/FSDP2 和确定性模式。
第一阶段连续训练 10 步并保存 step 5；第二阶段从 step 5 恢复到 step 10。
两阶段均使用 `training.steps=10`，保持学习率调度一致。
TensorBoard 中 steps 6–10 的原始 loss 和 grad_norm 必须逐值精确相等。
每个 rank 在恢复后、首次更新前检查 step 5 的全部 adapter 参数与 optimizer state，
并在 steps 6–10 每次更新后与连续训练快照做零容差比较，包括 Muon momentum。
恢复阶段只读基准快照，不覆盖连续训练的参数或状态证据；不更新 golden 文件。

```bash
python -m tests.integration_tests.run_tests /tmp/v41_muon_adapter_resume \
  --test_suite deepseek_v4_1_checkpoint \
  --test_name dsv41_lora_muon_adapter_only_resume_ep2_fsdp2 --ngpu 2
```

---

## 二、分层训练用例（Lite Actions）

Lite Actions 的用例清单由 [`.github/ci/suites.json`](../../.github/ci/suites.json) 管理，
具体定义位于 [`training_cases/`](training_cases/)。当前只接入已经验证的 `base` 层；`pr.py` 和 `nightly.py` 保留为空占位，后续验证完成后再补充。

| Suite | 当前内容 |
| --- | --- |
| `base` | 5 个基础用例 |
| `pr` | 继承 `base`，暂无新增用例 |
| `nightly` | 继承 `pr`，暂无新增用例 |

### 2.1 当前用例

| 编号 / Case ID | 设备与拓扑 | 组合特性 |
| --- | --- | --- |
| base_01 `dsv4_flash_a3_16p_cp2_ep8_eager` | A3，1 × 16P，CP2 / EP8 | DSV4 Flash 裁剪模型、Muon、eager |
| base_02 `dsv4_flash_a3_16p_cp2_ep8_compile` | A3，1 × 16P，CP2 / EP8 | DSV4 Flash 裁剪模型、Muon、compile |
| base_03 `dsv4_flash_a3_8p_cp1_ep8_graphtrainer` | A3，1 × 8P，CP1 / EP8 | DSV4 Flash 裁剪模型、GraphTrainer、`aot_fx_trace` |
| base_04 `dsv41_flash_a3_8p_multimodal_cpt_eager` | A3，1 × 8P，EP8 | DSV4.1 Flash 裁剪多模态、图文 CPT、eager |
| base_05 `dsv41_flash_a3_8p_multimodal_cpt_compile` | A3，1 × 8P，EP8 | DSV4.1 Flash 裁剪多模态、图文 CPT、compile |

所有当前用例执行 10 步，使用随机初始化；文本用例使用仓内 C4 资产，base_04 和 base_05 需要 CC12M 图文数据及 `DATASET_PATH`。
Lite Actions 的详细 launch、verify、资产注入和跨机调度说明后续补充。

---

## 三、CI 门禁流程与调度安全准入

为了保证内网物理 NPU 集群的执行安全，防止恶意 PR 注入未经审查的代码，CI 调度实行**受控安全准入**机制：

### 3.1 组织仓 PR 自动 CI 触发
1. **自动触发**：所有面向 `torchtitan-npu/torchtitan-npu:master` 提交的 PR（含来自外部 Fork 仓库的 PR）会自动触发组织仓的 GitHub Actions 工作流；
2. **安全防护**：
   - 外部 Fork 的首次 PR 受 GitHub 官方的 Workflow 审批策略保护（需要 Maintainer 批准运行）；
   - 调度机根据 PR API 严格比对 PR 编号、目标分支以及精确的 Head Commit SHA，坚决拒绝过期或关联异常的 Run；
   - 调度机在拉取代码时，仅下载该特定 PR Head SHA 对应的代码，杜绝中间人篡改；
   - 执行机**完全不持有 GitHub Token**，仅内网调度机单点具有回写 Comment 权限。

### 3.2 手动调度（Maintainer 冒烟与调试）
维护者可通过 `Training CI` 的 suite 下拉框或 GitHub CLI 手动发起调度，默认选择 `base`。
PR 固定选择 `pr`，定时事件固定选择 `nightly`，不受手动输入影响。定时与手动 Nightly 共用并发组，不取消正在执行的任务。

```bash
# 基础核心用例；suite 也可选择 pr 或 nightly。
gh workflow run training.yml -R torchtitan-npu/torchtitan-npu --ref master \
  -f suite=base

# 仅检查调度器与 NPU 资源连接，不运行训练。
gh workflow run dispatcher_smoke.yml -R torchtitan-npu/torchtitan-npu --ref master
```

### 3.3 CI 矩阵并发与动态端口分配

在 GitHub Actions CI 流水线中，每个 Matrix Case 是独立的调度与执行单元：
* **并发调度**：Lite Actions 调度机会根据物理集群的空闲 NPU 资源池，将互不冲突的 Case 并发拉起执行（如单机 8P 各用例并发执行，或双机 16P 跨节点独立分配）。
* **动态端口防冲突**：当同机或跨机并发拉起多个任务时，调度机会为每个任务动态分配互不冲突的通信端口（通过注入环境变量 `LITE_CI_MASTER_PORT`、`LITE_CI_HCCL_IF_BASE_PORT`、`LITE_CI_HCCL_HOST_PORT_RANGE` 与 `LITE_CI_HCCL_NPU_PORT_RANGE`）。仓内的 `tools/lite_actions/runner.py` 会自动读取并映射这些变量，覆盖用例的默认端口。开发者编写用例时无需硬编码特殊端口，使用默认值即可。

### 3.4 硬件环境与调度故障排障
若任务长时间卡在排队状态、远端 NPU 报告 Quarantine 或出现 SSH 连接异常，属于物理集群调度机层面的环境状态问题，请参阅 Lite Actions 调度系统运维说明或联系集群管理员排查。

---

## Engram HF 专项验证

在支持 Engram MXFP8 Host 通信的超节点上激活 CANN、torch extension 和训练环境，准备 V4.1 tokenizer assets 后执行：

```bash
HF_ASSETS_PATH=/path/to/v41-tokenizer \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
python -m tests.integration_tests.run_tests /tmp/engram-hf-output \
  --test_suite deepseek_v4_1_engram_hf --ngpu 4 --no-parallel
```

输出目录使用新目录。该 suite 包含 EP4/FSDP4 及 EP2×EFSDP2 联合分片两项用例，均为 V4.1 debug text、四卡、eager、seq512、GBS4，启用 Engram MXFP8 override，关闭普通模型 FP8 和 optimizer CPU offload。使用仓内 C4 数据及自动生成的微型 HF fixture，不需要正式模型权重。

先训练两步并保存同一模型的原生 DCP 和 FP32 HF 权重，再各自通过真实 CheckpointManager 初始化模型，使用相同的新优化器和数据种子训练三步。每个 rank 校验加载后的 Engram 参数及 MXFP8 缓存/scale 有效行字节，随后对比 TensorBoard loss/grad_norm。失败会使测试返回非零，成功输出 `HF_ROUNDTRIP PASS`。不读取固定 golden，不验证优化器状态续训，也不覆盖正式整包非 Engram 量化权重的导入。

仅验证 Engram 联合分片 HF 路径时，在上述命令追加
`--test_name dsv41_engram_mxfp8_hf_ep2_efsdp2`。该用例保持相同的 native/HF
加载后参数、MXFP8 缓存及三步 loss/grad_norm 精确比较，不加入默认 CI smoke。
