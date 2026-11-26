# TorchFT 计划抢占

Ascend 上的同步 TorchFT 训练可以在运行中释放整个副本的 NPU，供其他任务使用。Lighthouse 根据申请的卡数和节点选择副本，在下一次 quorum 成员决策中移出目标实例并终止其 Manager。节点上的 launcher 负责清理该实例的本地进程，资源归还后重新拉起副本。幸存副本继续训练。

以下以双机双副本为例。节点 `node-a` 运行 Lighthouse 和副本 0，节点 `node-b` 运行副本 1，每个副本使用本机的 0、1 号卡。将节点名替换为实际主机名，确认训练节点之间的 Manager、checkpoint 和 HCCL 通信可达。以下命令使用 Bash，从各自的 torchtitan-npu 仓库根目录执行。

## 安装配套 wheel

在各节点激活可运行 DeepSeek-V4 Flash TorchFT 训练的 Python 和 CANN 环境，再安装与该环境匹配、包含 Lighthouse `POST /preemption` 接口的配套 torchft wheel。控制端和 Lighthouse 使用相同构建。配套 wheel 与上游标准包都可能显示为 `0.2.0`，可用下面的 `status` 命令确认实际运行的服务支持抢占。[配套 torchft 实现](https://github.com/chango9543/torchft/pull/1)

节点 launcher 使用 `psutil` 检查和清理训练进程。项目的 `torchft` 可选依赖已包含 `psutil`。如果跳过该可选依赖，直接用下方的 `--no-deps` 命令安装预编译或自行构建的 wheel，请先在每个运行 launcher 的 Python 环境中执行 `python -m pip install psutil`。

```bash
python -m pip install --force-reinstall --no-deps --no-index --find-links https://cann-ai.obs.cn-north-4.myhuaweicloud.com/torchtitan-npu/simple/torchft-suspend/index.html torchft==0.2.0+preempt.1
```

### 从源码构建同一功能的 wheel

需要自行构建时，使用仓内的 [Lighthouse 抢占补丁](../../torchtitan_npu/patches/torchft/torchft-0.2.0-lighthouse-preemption.patch)。补丁在编译 torchft 时应用，将抢占路由和 quorum 成员管理编入 `torchft._torchft`。训练环境只需安装构建产物。

在与部署环境匹配的 Linux 架构和 Python 环境中准备 Rust 工具链、C 链接器、`protoc` 和 `maturin>=1.7,<2.0`。将 `TORCHFT_SRC` 设为独立、干净的 torchft 0.2.0 源码目录（Git 标签 `v0.2.0` 的检出目录或解压后的源码包）。以下命令从 torchtitan-npu 仓库根目录执行。若使用解压包，先执行 `git -C "$TORCHFT_SRC" init -q`，使补丁相对该源码根目录应用。

```bash
(
  set -e
  patch_file="$PWD/torchtitan_npu/patches/torchft/torchft-0.2.0-lighthouse-preemption.patch"
  cd "${TORCHFT_SRC:?请设置原版 torchft 0.2.0 源码目录}"
  git apply --check "$patch_file"
  git apply "$patch_file"
  cargo test --release --lib
  maturin build --release --interpreter "$(command -v python)" --out dist
)
```

将 `$TORCHFT_SRC/dist/` 中生成的 wheel 分发到各节点，在各自的运行环境中用确切文件路径安装，随后用下文的 `status` 命令验证服务。

```bash
python -m pip install --force-reinstall --no-deps "${TORCHFT_WHEEL:?请设置构建产物的绝对路径}"
```

## 配置作业环境

在 Lighthouse、launcher 和发送控制命令的终端分别加载以下设置。将地址改为各节点可访问的 Lighthouse 地址，所有终端输入同一作业的 token。示例使用 HTTP，适用于可信训练网络。

```bash
export TORCHFT_LIGHTHOUSE=http://node-a:29920
read -rs -p 'TorchFT job token: ' TORCHFT_PREEMPT_TOKEN
echo
export TORCHFT_PREEMPT_TOKEN
LAUNCHER=torchtitan_npu/extensions/torchtitan/experiments/torchft/preempt.py
CONTROL=tests/integration_tests/torchft_preemption.py
JOB_DIR="$PWD/outputs/preempt-demo"
mkdir -p "$JOB_DIR"
```

## 启动 Lighthouse

在 `node-a` 创建布局文件。键 `torchtitan_ft_N` 对应训练参数 `replica_id=N`。每个节点的数组列出该副本占用的物理设备 ID，顺序与该节点的 `ASCEND_RT_VISIBLE_DEVICES` 一致。同一设备只分配给一个副本。`TORCHFT_PREEMPT_LAYOUT` 是 Lighthouse 本机的文件路径。

```bash
export TORCHFT_PREEMPT_LAYOUT="$JOB_DIR/layout.json"
cat > "$TORCHFT_PREEMPT_LAYOUT" <<'JSON'
{
  "torchtitan_ft_0": {"node-a": [0, 1]},
  "torchtitan_ft_1": {"node-b": [0, 1]}
}
JSON
torchft_lighthouse --bind node-a:29920 --min_replicas 1 --join_timeout_ms 5000
```

保持 Lighthouse 运行，在另一终端执行 `python "$CONTROL" status`。返回的 `layout` 应与文件内容一致。此时 `quorum` 可以为空。连接失败时检查监听地址和端口；HTTP 401 时核对 token；HTTP 404 时核对运行 Lighthouse 的 wheel 以及启动进程时是否设置了 `TORCHFT_PREEMPT_LAYOUT`。

## 启动训练

两个训练节点各运行一个 launcher。将下面头两行在 `node-a` 设为 `REPLICA=0`、`NODE=node-a`，在 `node-b` 设为 `REPLICA=1`、`NODE=node-b`。每个副本都在一台机器内组成独立的 torchrun 作业。训练开关 `fault_tolerance.enable_preemption` 默认关闭；启用时，由 launcher 给每次拉起的训练进程注入本地登记目录 `TT_PREEMPT_LOCAL` 和逻辑副本名 `TT_PREEMPT_SLOT`。

```bash
REPLICA=0
NODE=node-a
export ASCEND_RT_VISIBLE_DEVICES=0,1
export TRITON_NPU_COMPILER_PATH="${TRITON_NPU_COMPILER_PATH:-$ASCEND_HOME_PATH/tools/bisheng_compiler/bin}"
python "$LAUNCHER" launch --replica "$REPLICA" --node "$NODE" --state-dir "$JOB_DIR/workers" -- \
  torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 --max-restarts=0 \
  --rdzv-backend=static --master-addr="${NODE_IP:-$NODE}" --master-port="$((29921 + REPLICA))" \
  -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 --config deepseek_v4_flash_8p_torchft \
  --dump-folder "$JOB_DIR/replica$REPLICA" --training.steps 200 --training.seq-len 256 \
  --debug.seed 42 --metrics.log-freq 1 \
  --hf-assets-path "$PWD/tests/assets/deepseek_v3" \
  --dataloader.dataset c4_test --dataloader.dataset-path "$PWD/tests/assets/c4_test" \
  --checkpoint.folder "$JOB_DIR/replica$REPLICA/checkpoints" --checkpoint.load-only \
  --fault-tolerance.enable-preemption --fault-tolerance.replica-id "$REPLICA" \
  --fault-tolerance.group-size 2 --fault-tolerance.min-replica-size 1 \
  --fault-tolerance.process-group-timeout-ms 120000 --fault-tolerance.quorum-warmup-timeout-ms 600000 \
  --parallelism.data-parallel-shard-degree 2 --parallelism.expert-parallel-degree 2 \
  --override.imports torchtitan_npu.override.common.rope.workaround
```

`group-size` 统计 TorchFT 副本数，`nproc-per-node` 统计当前 torchrun 作业在本节点的 rank 数。TorchFT 会向对端公布本副本的 TCPStore 地址；如果主机名解析到回环地址或管理网卡，设置 `NODE_IP` 为对端可访问的本机训练网卡 IP。双副本释放一个后还要继续提交训练步，Lighthouse 的 `--min_replicas` 和训练端的 `--fault-tolerance.min-replica-size` 均设为 1。保持 launcher 运行，由它负责计划内重新拉起。

单机运行两个副本时，把两个布局条目都指向同一节点，分配互不重叠的设备列表，并为两个 launcher 分配互不重叠的 `HCCL_NPU_SOCKET_PORT_RANGE`。一个副本跨多台机器时，在该副本的布局条目中列出所有参与节点，每节点运行一个相同 `--replica` 的 launcher，并沿用该副本原有的多机 torchrun rendezvous 参数。

## 抢占与恢复

外部调度器通过 Lighthouse 的 `POST /preemption` 接口申请和归还资源。请求头使用 `Authorization: Bearer <作业 token>` 和 `Content-Type: application/json`。抢占请求示例为 `{"action":"suspend","request":"inference-1","cards":2,"node":"node-b"}`，归还资源时发送 `{"action":"resume","request":"inference-1"}`。同一次操作使用相同的请求 ID。

以下命令使用仓内集成测试工具模拟调度器。节点 launcher 继续通过 `preempt.py launch` 运行。

1. 执行 `python "$CONTROL" status`，等待 `quorum` 同时包含 `torchtitan_ft_0:<实例 ID>` 和 `torchtitan_ft_1:<实例 ID>`。记录副本 1 的实例 ID。
2. 执行 `python "$CONTROL" suspend --request inference-1 --cards 2 --node node-b`。此例要求在 `node-b` 释放两张卡，Lighthouse 会选中副本 1。省略 `--node` 时按整个作业的可释放卡数选择。资源以整个副本为单位释放，实际卡数可能大于申请值。
3. 用 `status` 检查 `lease.targets`。当 `lease.phase` 变为 `released`，目标副本各节点均已确认释放，此时可将对应设备交给抢占任务。训练日志应显示副本 0 继续提交。
4. 抢占任务归还设备后，执行 `python "$CONTROL" resume --request inference-1`。目标 launcher 使用原命令重启训练。用 `status` 确认 `lease.phase` 为 `restored`、副本 1 以新的实例 ID 加入 quorum，再从训练日志确认两个副本均已提交新的训练步。

同一作业一次处理一个成员变更，下一次抢占使用新的请求 ID。若请求返回 `insufficient releasable cards`，检查指定节点的设备数，以及保留最少副本后可释放的卡数。

要自动验证两个单节点副本（可在同一台机器上）的完整流程，可在申请释放设备的节点上执行下列命令。先启动 Lighthouse 和两个 launcher，并预留足够训练步数供副本重新加入。将卡数改为本次申请值，布局中的节点名使用本机主机名。

```bash
python "$CONTROL" verify --cards 2 --after-step 3 --timeout 900 --probe
```

验证依次检查双副本训练、目标实例退出及节点释放回执、幸存副本单独推进。加上 `--probe` 后，独立进程会在每张释放的 NPU 上执行计算。工具随后发送 `resume`，等待新实例加入，并确认两个副本的进度都超过重入时的最高步数。
