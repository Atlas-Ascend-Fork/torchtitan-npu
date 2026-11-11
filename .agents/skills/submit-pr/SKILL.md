---
name: submit-pr
description: "在 torchtitan-npu 仓库创建、更新或推送 GitCode Pull Request。用于提交/创建/更新 PR、提 PR、发 PR、推送代码、生成精简 PR 标题或描述、提交前代码自审、精度/性能验证。要求检查各级 AGENTS.md、控制单个 PR 代码变更不超过 1k 行，并遵循 check-pr-docs 与 developer-tests-review。不负责业务实现或 CI 修复。"
---

# torchtitan-npu 提交 PR

只有用户明确要求提交、创建或更新 PR 时，才执行提交、推送和远端写操作。流程：读规则 → 定范围 → 自审 → 验证 → 判断精度影响（仅新建 PR）→ 写描述 → 推送创建 → 回读。

## 规则

- 读根 `AGENTS.md`、`.agents/AGENTS.md` 和改动路径上所有目录级 `AGENTS.md`，最近的目录规则优先。
- `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md` 是唯一模板来源；忽略远程 skill 的固定模板路径。
- 文档和依赖按 `check-pr-docs`，测试按 `developer-tests-review`，数值按 `numerical_alignment.md`，性能按 `metrics_and_debugging.md` 的 Profiling 部分。
- 详细自审项见 [references/review-checklist.md](references/review-checklist.md)。

## 范围与规模

- 新建 PR 时，目标分支使用用户指定分支或仓库默认分支（当前 `master`）；更新已有 PR 时，先读取该 PR 的实际 base/head，不要默认 `master`。
- 用 merge-base 计算 diff，不要用 `HEAD~1`：`merge_base=$(git merge-base "$(git rev-parse <base_remote>/<base>)" HEAD)`，然后执行 `git diff --name-status "$merge_base...HEAD"`；更新已有 PR 时 `<base>` 用该 PR 的实际目标分支。
- 检查 `git status --short`；排除生成物、日志、密钥、`.agents/skills/_remote/` 和远程 skill 软链。
- 不要自行新建分支或 cherry-pick。如果当前分支是目标分支或含无关提交，暂停并让用户处理。
- 代码增删合计不超过 1000 行（纯文档、生成数据和测试数据不计）；超出时拆分 PR，或取得用户或 reviewer 的明确同意。

## 自审与验证

- 满足各级 `AGENTS.md` 的要求：代码放置、依赖方向、导入和注册顺序、命名、断言、参数、注释和测试规则。
- 简化：复用现有实现；删除多余抽象、重复代码、未使用的导入和变量、死代码、注释掉的代码、调试输出和临时开关。
- 注释只保留维度语义、并行梯度 placement、workaround 原因、上游 PR 等必要信息；删除复述代码的注释；TODO 写明限制。
- 检查共享代码的所有调用点，标出 checkpoint、数值、性能或用户行为风险。
- 文档和依赖检查默认只读，未获授权不修改。
- 精简后重新统计行数；改代码后重跑受影响的检查。
- 未执行或未通过的检查必须写明原因，不能写成通过。以下 Lint 和 CPU UT 在 Linux 环境执行；

```bash
# Lint
python -m pre_commit run --all-files --show-diff-on-failure

# CPU UT
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -v --tb=short tests/unit_tests -x
```

- 精度：按 `numerical_alignment.md` 执行验证；修改前后训练对比优先用 `scripts/numerical_alignment/loss_compare.py`（两个 Git ref），需要定位到算子/模块时再用 `.agents/skills/numerics-debugging/`；证据按下方“如何测试 / 精度和性能验证结果”记录。
- 性能：涉及 kernel、并行、编译、显存、融合或通信等性能敏感改动时，按 `metrics_and_debugging.md` 采集；相同环境和配置下对比 baseline，记录吞吐、时延、显存或 MFU；未测量就写“未测量”，且不得给出性能提升或回退结论。以性能优化为目标的 PR，未完成性能验证时同样不得创建可审查 PR。
- NPU：涉及分布式、kernel、override、patch、模型训练或 checkpoint 的改动，执行对应的 smoke 或 integration 测试；CPU UT 不计入 ST 覆盖。

## 提交前判断精度影响

创建新 PR 前，agent 按 `numerical_alignment.md` 第 4 节和 diff 判断并分类；更新已有 PR 时只更新本次修改涉及的描述部分，补充后重新压缩整段描述，删除过期、重复或已完成内容，保持精简，不改标题、draft 状态或 review 状态。

判断为有影响，或改动命中上方验证范围、可能改变模型输出、梯度、loss、参数更新、随机行为时：

- **无影响**：直接创建可审查 PR；PR 描述仍要按 `numerical_alignment.md` / `.agents/AGENTS.md` 写明跳过精度验证的依据，并保留 `精度和性能验证结果` 段。
- **有影响**：只创建草稿 PR（`draft=true`），在 `如何测试` 的 `精度和性能验证结果` 段标注“精度验证待补充”，相关 Checklist 不勾选，不请求 review；由用户自行补充验证后转为可审查 PR。

判断不清时按“有影响”处理，直接创建草稿 PR。

## 标题与描述

- 标题固定为 `[type] 描述`：优先 `[feat]`、`[fix]`、`[refactor]`、`[docs]`、`[test]`；仓库已有其他类型且与改动一致时也可使用；只选一个类型；禁止 `feat: 描述`、`test: 描述` 或无标签标题。
- 严格使用仓库模板，保留全部二级标题：`描述`、`类型`、`Checklist:`、`如何测试`（含 `精度和性能验证结果` 三级标题）、`其他信息`。标题层级和顺序与模板一致；`类型` 和 `Checklist:` 条目逐字沿用模板，只改勾选状态。
- 提交前运行 `python .agents/skills/submit-pr/scripts/validate_pr_body.py --body <pr-body.md>`，失败则修正后重试。
- 描述只保留核心信息（正文 10-15 行，说明为什么改、行为变化和影响范围），不要逐文件复述 diff 或粘贴大段日志、trace、代码；如何测试写实际命令和结果摘要，未执行写原因；精度和性能结果用紧凑表格给出基线/版本、环境、命令、核心指标、判定和证据链接，无验证时写明跳过依据；其他信息只写 checkpoint 兼容性、风险、待完成项、文档和依赖影响、关联 issue。
- 类型默认只勾一个；只有模板或用户明确允许多选时才勾多个。Checklist 只勾真实完成项：Lint 通过才勾代码风格，测试执行才勾已测试，文档实际更新才勾文档，标题格式正确才勾类型标签。
- 隐私：不得包含个人隐私信息、token 或密钥、内网 IP、主机名、内部 URL、客户或未公开数据、带用户名的本地路径；用 `<user>`、`<host>`、`<token>` 等占位符或脱敏链接，敏感原文走仓库认可的私下渠道。

## 推送与创建

- 有写权限时推送到 `origin`，否则推送到个人 fork；使用 `git push -u <remote> <branch>`，不要强制推送；有未提交改动时先确认范围，不要直接用 `git add -A`。
- 只检查 `GITCODE_API_TOKEN` 是否存在，不打印、不写入文件；fork 使用 `parent.full_name`、`head=username:branch`，必要时传 `fork_path`；payload 用 `curl --data-binary`。
- 创建 PR 使用 `POST /repos/{owner}/{repo}/pulls`；已有同源 PR 时用 `PATCH` 只更新本次修改涉及的描述部分，不改 title、draft 或 review 状态，不要重复创建。端点见 [references/gitcode-api.md](references/gitcode-api.md)。
- 有影响时按上节创建 `draft=true` 的草稿 PR，在精度结果段标注“精度验证待补充”，不请求 review；有环境时先跑 `loss_compare.py` 并把结果或链接写进草稿 PR。不要替用户转为可审查 PR。
- 创建后回读 PR，核对标题、分支、描述、draft 状态和隐私保护；报告 `html_url` 和未完成项。
- 本仓 PR pipeline 由 `/compile` 评论触发。

## 完成检查

- [ ] 已读各级 `AGENTS.md` 和 PR 模板
- [ ] diff 中没有无关文件；代码增删合计 ≤1k 行或已获同意
- [ ] 代码已自审、精简，注释只保留必要项
- [ ] 测试、文档和依赖要求已按对应 skill 检查
- [ ] Lint、UT、NPU、性能完成或记录原因；精度影响已判断（更新 PR 只改描述并重新压缩，有影响的新 PR 仅建草稿并在精度段标注待补充）
- [ ] 标题和描述符合模板；类型只勾一个；Checklist 只勾真实项；校验通过
- [ ] 隐私已保护；PR 已回读，链接和未完成项已交付
