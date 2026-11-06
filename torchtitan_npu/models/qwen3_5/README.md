# Qwen 3.5 系列模型（可选安装 FLA）

先完成仓库的[基础安装](../../../docs/user-guides/installation.md)。以下命令均在仓库根目录执行。

Torch 2.14 的首次 JIT 编译需要支持 C++20 的编译器。加载 CANN 环境后，可使用毕昇编译器。

```bash
export CC="$ASCEND_HOME_PATH/tools/ccec_compiler/bin/bisheng" CCC_OVERRIDE_OPTIONS="# +-std=c++20"
```

## 功能调测验证

调测使用 TorchTitan 源码自带的 tokenizer 和示例数据。调测资产已就绪时，可跳过克隆步骤。

```bash
git -C .. clone --depth 1 --branch v0.3.0 https://github.com/pytorch/torchtitan.git
python -m pip install av einops pillow
bash examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh
```

## 训练（FLA 可选安装）

以下以 `qwen35_27b_long_text_sft` 配置为例。运行前，导出 `HF_ASSETS_PATH` 和 `DATA_FILES`。
前者是包含 Qwen 官方 HF 权重和 tokenizer 的模型目录，后者是 JSONL 训练数据文件。数据每行例如
`{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"回答"}]}`。

### 默认训练

直接运行，使用配方自带的 GDN 实现。

```bash
NGPU=16 MODULE=torchtitan_npu.models.qwen3_5 CONFIG=qwen35_27b_long_text_sft \
  bash scripts/run_train.sh \
    --hf-assets-path "$HF_ASSETS_PATH" \
    --dump-folder ./outputs/qwen35-default \
    --checkpoint.enable \
    --checkpoint.initial-load-path "$HF_ASSETS_PATH" \
    --checkpoint.initial-load-in-hf \
    --parallelism.context-parallel-degree 8
```

### FLA 加速训练（可选安装）

按照**以下经过验证的版本配套**进行安装。

| 软件包 | 版本 |
| --- | --- |
| `flash-linear-attention` | `0.6.0` |
| `fla-core` | `0.6.0` |
| `triton` | `3.6.0` |
| `triton-ascend` | `3.6.0.post2` |
| `transformers` | `5.22.1` |
| `tokenizers` | `0.23.2` |

```bash
python -m pip uninstall -y flash-linear-attention fla-core triton-ascend triton
python -m pip install --only-binary=:all: \
  'flash-linear-attention==0.6.0' 'fla-core==0.6.0' \
  'triton==3.6.0' 'triton-ascend==3.6.0.post2' 'transformers==5.22.1' 'tokenizers==0.23.2' \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

以典型的 CP8 x DP2（16路参数分片）算法训练为例，可选安装完成后，运行下列命令启用 FLA。

```bash
NGPU=16 MODULE=torchtitan_npu.models.qwen3_5 CONFIG=qwen35_27b_long_text_sft \
  bash scripts/run_train.sh \
    --hf-assets-path "$HF_ASSETS_PATH" \
    --dump-folder ./outputs/qwen35-fla \
    --checkpoint.enable \
    --checkpoint.initial-load-path "$HF_ASSETS_PATH" \
    --checkpoint.initial-load-in-hf \
    --parallelism.context-parallel-degree 8 \
    --override.imports \
      torchtitan_npu.override.qwen3_5.varlen_attention.asc_cp \
      'torchtitan_npu.override.qwen3_5.fla.operators={"context_parallel":true}'
```

`--override.imports` 会替换配方中的整个列表。上例保留 `varlen_attention.asc_cp`，
用带 `context_parallel=true` 的 FLA 条目替换原 GDN。其他配置已有的 override 也应保留。
未开启 CP 的配置使用 `torchtitan_npu.override.qwen3_5.fla.operators`。
