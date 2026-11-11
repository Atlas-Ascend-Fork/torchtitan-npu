# GitCode PR API 参考（v5）

## 认证

API 根地址：`https://api.gitcode.com/api/v5`

优先使用请求头：`Authorization: Bearer ${GITCODE_API_TOKEN}`，不要把 token 放进
URL。也支持 `PRIVATE-TOKEN` 头和 `?access_token=` 查询参数，但查询参数容易进入
日志和 shell 历史，只在无法使用请求头时使用。Token 至少需要 `read_repository`、
`write_repository` 和 `read_api` 权限。

只判断 token 是否存在，不要打印，也不要写入仓库文件、提交、PR 描述或日志；同样
不要把带 token 的 URL、个人邮箱或电话、内网地址和本地用户路径作为证据链接。

Windows PowerShell 中 `curl` 可能是 `Invoke-WebRequest` 的别名；使用 `curl.exe`
或在 Git Bash 中执行本文命令。

## 仓库和 fork

```bash
GET /repos/{owner}/{repo}
```

```bash
curl -sS -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}" |
  jq '{fork, parent: .parent.full_name, default_branch}'
```

- `fork=true` 时，目标仓库使用 `parent.full_name`；`head` 使用 `username:branch`；
  跨仓 PR 传 `fork_path=owner/repo`。
- 当前用户：`GET /user`，取 `.login`。
- 解析 remote：

```bash
repo_url=$(git remote get-url origin)
repo_url=${repo_url%.git}
case "$repo_url" in
  *://*) repo_path=${repo_url#*://}; repo_path=${repo_path#*/} ;;
  *@*:*) repo_path=${repo_url#*:} ;;
  *) repo_path=$repo_url ;;
esac
owner=${repo_path%%/*}
repo=${repo_path#*/}
```

不依赖 `sed -E`，兼容 macOS；带端口的 `ssh://` remote 请先转成 `git@host:owner/repo`
形式。

不要假设 `origin` 就是目标仓库；用户指定 `upstream` 时以用户指定为准。

## 创建 PR

```bash
POST /repos/{owner}/{repo}/pulls
```

请求体：`title`、`head`、`base`、`body`，跨仓时加 `fork_path`；可选 `draft`、
`labels`、`assignees`、`testers`。精度证据待补充时可传 `"draft": true`；转为可审查
由用户在 PR 界面手动完成。

用 Python 读取 Markdown 正文，避免 shell 引号和换行问题：

```bash
python - <<'PY'
import json, os, pathlib
payload = {
    "title": os.environ["PR_TITLE"],
    "head": os.environ["PR_HEAD"],
    "base": os.environ["PR_BASE"],
    "body": pathlib.Path("pr-body.md").read_text(encoding="utf-8"),
    "draft": os.environ.get("PR_DRAFT", "false").lower() == "true",
}
if os.environ.get("PR_FORK_PATH"):
    payload["fork_path"] = os.environ["PR_FORK_PATH"]
pathlib.Path("pr-payload.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

curl -sS -X POST \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls" \
  -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @pr-payload.json
```

有精度影响时设置 `PR_DRAFT=true`；创建后 `GET` 回读 `draft` 字段确认。响应中关注
`number` 和 `html_url`；PR 网页地址以响应为准，不要自行拼接。

## 更新 PR

```bash
PATCH /repos/{owner}/{repo}/pulls/{number}
```

更新已有 PR 时只更新描述，payload 只放 `body`；不要带 `title`、`head`、`base`
或 `draft`，避免覆盖标题和草稿状态：

```bash
python - <<'PY'
import json, pathlib
payload = {
    "body": pathlib.Path("pr-body.md").read_text(encoding="utf-8"),
}
pathlib.Path("pr-update-payload.json").write_text(
    json.dumps(payload, ensure_ascii=False), encoding="utf-8")
PY

curl -sS -X PATCH \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${number}" \
  -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data-binary @pr-update-payload.json
```

更新后重新 `GET` 一次确认。只有用户明确要求改标题时，才另做一次只带 `title`
的更新，不要和描述更新混在一起。

## 获取 PR 和文件变更

```bash
GET /repos/{owner}/{repo}/pulls/{number}
GET /repos/{owner}/{repo}/pulls/{number}/files.json
```

```bash
curl -sS -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${number}" |
  jq '{number,title,state,target_branch,source_branch,html_url,web_url}'

curl -sS -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${number}/files.json" |
  jq '.diffs[] | .statistic'
```

## 查找已有 PR

```bash
GET /repos/{owner}/{repo}/pulls?state=open&per_page=100
```

```bash
curl -sS -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls?state=open&per_page=100" |
  jq --arg b "$branch" --arg t "$base" \
    '.[] | select(.source_branch == $b and .target_branch == $t) |
     {number, html_url, state}'
```

找到 head 和 base 相同的 PR 时，用 `PATCH` 更新，不要重复创建。

## 评论

```bash
GET  /repos/{owner}/{repo}/pulls/{number}/comments
POST /repos/{owner}/{repo}/pulls/{number}/comments
POST /repos/{owner}/{repo}/pulls/{number}/discussions/{discussion_id}/comments
```

列表接口的关键字段包括 `id`、`discussion_id`、`body`、`user.login` 和
`comment_type`；行内评论的位置在 `diff_position.start_new_line`，列表接口没有顶层
`path`/`position`。`comment_type` 为 `pr_comment` 或 `diff_comment`。

单条评论接口 `GET /repos/{owner}/{repo}/pulls/comments/{id}`：顶层 `path` 仍可能为
空，实际路径和行号在 `position.new_path` / `position.new_line`，`comment_type` 为
`DiffNote`。

```bash
curl -sS -X POST \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/${number}/comments" \
  -H "Authorization: Bearer ${GITCODE_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"body":"评论内容"}'
```

## 常见错误

| 状态码 | 原因 | 处理 |
| --- | --- | --- |
| `400` | 缺少 `title`/`head`/`base` | 检查 payload |
| `401` | token 缺失或无效 | 重新提供 token |
| `403` | 无权限推送或创建 PR | 检查仓库权限和 fork |
| `404` | 仓库、分支或 PR 不存在 | 检查 owner/repo、分支和编号 |
| `422` | 分支不存在、重复 PR 或字段不合法 | 检查 head/base，查找已有 PR |
| `429` | 速率限制 | 降低频率后重试 |

## 官方文档

- https://docs.gitcode.com/docs/apis/
- https://docs.gitcode.com/docs/apis/post-api-v-5-repos-owner-repo-pulls
- https://docs.gitcode.com/docs/apis/patch-api-v-5-repos-owner-repo-pulls-number
- https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-number
- https://docs.gitcode.com/docs/apis/get-api-v-5-repos-owner-repo-pulls-number-comments
