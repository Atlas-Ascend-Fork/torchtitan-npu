# Lint 指南

本指南只覆盖代码静态检查、格式检查和开源合规检查，不包含模型训练、模型加载、NPU 算子验证或功能测试。
Lint 可以在 CPU 环境执行，不需要 CANN、HCCL 或可用的 Ascend NPU。Python 类型检查仍需要固定版本的上游源码和被检查代码导入的类型依赖；安装检查工具并不等于这些前提已经满足。

## 检查范围

仓库通过 pre-commit 统一执行检查，具体配置见
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)。不同文件类型会触发不同的 hook：

| 修改内容 | 主要检查 | 是否需要 NPU |
| --- | --- | --- |
| Markdown、YAML、JSON 和配置文件 | 文件格式、尾随空格、冲突标记、拼写和 OAT 合规性 | 否 |
| Python | Ruff 格式与规则检查、Pyrefly 类型检查 | 否 |
| C/C++ | clang-format 格式检查 | 否 |

## 环境与依赖

本地建议使用 Python 3.12，与仓库 CI 的要求保持一致。Windows 环境需要使用 WSL 发行版或
Git Bash，以便执行 OAT hook 使用的 `bash scripts/oat_check.sh`。

### 最小安装

只检查 Markdown、配置文件或其他非 Python 文件时，安装 pre-commit 即可：

```bash
python -m pip install pre-commit
```

修改 Python 文件时，还需要 Pyrefly。可以只安装相关工具：

```bash
python -m pip install pre-commit pyrefly
```

也可以安装仓库提供的开发依赖。该依赖集合提供检查工具和开发测试依赖，未包含 `torch`、`torch_npu`、
`torchtitan` 或 Triton；Python 类型检查的源码与类型依赖前提见下一节：

```bash
python -m pip install -r requirements-dev.txt
```

`scripts/oat_check.sh` 在检测到本地没有 `oat-py` 时会尝试自动安装 `oat-py>=1.0.1`。
`requirements.txt` 记录配套版本；仅安装检查工具时，并未安装其中的类型依赖。

### Python 类型检查前提

`pyproject.toml` 的 Pyrefly 配置使用 `search-path = ["/tmp/torchtitan"]`。执行 Python 或全仓检查前，应先确认该目录是 `requirements.txt` 所声明版本对应的 TorchTitan checkout；版本/tag 推导与源码准备逻辑见 [`.ci/setup_torchtitan.sh`](../../.ci/setup_torchtitan.sh)。仅安装 PyPI 包或把源码放在其他目录，不会创建该 checkout；按根开发规则，还应确认上述固定路径的源码。

Pyrefly 使用当前 Python 环境查找类型导入。当前配置仅对清单中列出的模块忽略缺失导入，并将 `torch_npu` 和 `torch.npu` 替换为 `Any`；它不会忽略所有运行时模块。出现 `missing-import` 时，应按仓库配套版本准备相应 Python 类型依赖。不需要为类型检查启动训练或设备，但不能通过屏蔽检查规则来替代这些前提。

## 执行检查

### 检查本次修改

定位本次修改的格式或局部错误时，可先执行增量检查；仅增量通过不能视为完整 Lint 通过：

```bash
python -m pre_commit run --files <file1> <file2> --show-diff-on-failure
```

例如：

```bash
python -m pre_commit run --files docs/developer_guides/lint_guide.md --show-diff-on-failure
```

部分 hook（例如 Ruff 和 clang-format）可能会自动修复文件。检查完成后，应查看工作区差异并重新运行命令。

### 检查整个仓库

按[根开发规则](../../.agents/AGENTS.md)，提交前从仓库根目录执行完整 Lint；仅此完整命令通过后，才视为 Lint 通过：

```bash
python -m pre_commit run --all-files --show-diff-on-failure
```

该命令仍然只进行静态检查，不会启动训练。

### 安装 Git hook

提交前必须安装 Git hook。安装后，每次提交会自动对增量文件执行检查：

```bash
python -m pre_commit install
```

## 检查失败时

- 根据 hook 输出修复格式、拼写、类型或合规问题，然后重新运行同一条命令。
- 如果提示 `pyrefly` 不存在，安装 `pyrefly` 或完整的 `requirements-dev.txt`。
- 如果 Windows shell 找不到 `bash`，切换到 Git Bash 或已安装 Linux 发行版的 WSL。
- Lint 无法替代 NPU 功能验证。涉及 `torch_npu`、NPU 算子、分布式并行、checkpoint 或模型训练行为的改动，
  仍需在具备相应设备和软件栈的环境中另行验证。

相关配置和开发依赖如下：

- [`requirements-dev.txt`](../../requirements-dev.txt)
- [`pyproject.toml`](../../pyproject.toml)
