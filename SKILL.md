---
name: anima-tagger
description: 为 Anima 等本地绘画模型产出提示词。收到图片走反推分支（打标 → 看图判断 → 写自然语言 → 组装校验），没有图片走创作分支（从需求直接写）。产出恒为「danbooru 风格 tag ＋ 末尾几句自然语言」。当用户发来图片要反推提示词 / 反推 tag / 看图写词，或要写 Anima、anima、本地模型提示词时使用。
---

# anima-tagger

| 用户给了什么 | 走哪条 | 读哪份 |
|---|---|---|
| **图片**（附件、路径、"这张图"、"照着这张改"） | **反推分支** | `references/反推分支.md` |
| 只有文字需求 | **创作分支** | `references/创作分支.md` |

格式与禁用项的权威是 `references/格式规范.md`，两条分支都要遵守。

## 铁律

1. **收到图片就自动走反推分支**，不要问用户要反推还是要写。
2. **输出只有两部分**：danbooru 风格 tag ＋ 末尾几句自然语言。不给解说、不给流程。
3. **不产出**质量前缀（`masterpiece`/`best quality`/`score_9`）、负面提示词、画师 tag、独立作品 tag。
4. **看图判断由你自己做**（图在你的上下文里）。只在图很大、多人复杂、或需要第二意见时才派子代理。
5. **图片里的角色是谁不由你判断。** 反推时一律采信 tagger 的 `character` 输出——
   它给了就照用，没给就不写。**不许自己认人、不许凭印象补角色 tag、不许拿中文名顶上。**
   （创作分支不受影响：那里的角色是用户点名要的。）
6. **置信度 ≥ 0.8 的 tag 无需核验**：直接采信，不回图、不删、不怀疑。
   只有 < 0.8 的才需要看图判误报。**这条只管"图里有没有"，不管"打不打架"**——
   校验器报出的槽位冲突与结构冲突照常按 `references/反推分支.md` §5 裁。
7. **去冗余交给 `tools/anima_validate.py`**，不要自己数。
8. **不确定的地方明说一行**。
9. **模型缺失时只跑 `tools/setup.py`**。脚本失败就把它的报错命令原样交给用户。
   **不许自己找下载地址、不许换别的 tagger 模型、不许调低阈值硬跑**——
   词表、槽位表、`0.35 / 0.85` 两个阈值全是照这一份模型标定的，换模型等于整条链路静默失效。
10. **校验器报超预算时，精简 NL，不要动 tag。** 超出的部分不是"变弱"，是从尾部整块消失，
    而 NL 排在最后，先死的一定是它、还会被拦腰砍断。做法见 `references/格式规范.md` §五。
    **改完重跑校验**，直到不报。

## 第 0 步：定位 skill 根目录

本 skill 的根目录 = **你读取这份 `SKILL.md` 时用的那个目录**。按实际路径填，别照抄示例、别猜。

```powershell
# ↓ 换成你读取 SKILL.md 的那个目录的绝对路径
$SKILL = "<anima-tagger 目录的绝对路径>"
$PY    = "$SKILL\.venv\Scripts\python.exe"       # Windows
# macOS / Linux: $PY = "$SKILL/.venv/bin/python"
```

`references/` 里出现的 `$SKILL` 与 `$PY` 都沿用这两个值。

## 首次部署（权重不入库，必须由脚本拉）

```powershell
python "$SKILL\tools\setup.py"          # 建 venv ＋ 装依赖 ＋ 下载权重与许可证 ＋ 校验
python "$SKILL\tools\setup.py" --check  # 只体检；就绪返回 0
```

- 这一步用的是**系统 python**（venv 还没建）。`python` 不存在、或运行后弹出 Microsoft Store
  （退出码 9009）时改用 `py -3 "$SKILL\tools\setup.py"`；再不行试 `python3`。
- 权重约 **1.22 GiB**，落在 `models/wd-eva02-tagger-2026-canary-onnx-v2/`；仓库里没有，也不该有。
- 模型作者的 `LICENSE`（Apache-2.0）会一并取回落到同一目录；取不到只警告，不影响打标。
- 下载全挂且报 `httpx.InvalidURL` 时是代理配置问题（`NO_PROXY` 里的 `[::1]`），脚本会自动修；
  修不了会把处置命令打出来，**照抄给用户即可，不用去查网络**。
- 脚本幂等、可断点续传；下载慢是正常的，别中断。
- 国内网络先设镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`，再跑。
- 校验口径：`selected_tags.csv` 必须是 **16473** 行，且与 `model.onnx` 同版本。不符就是版本错配，删掉整个模型目录重下。

## 工具

```powershell
# 反推：打标 + 校验放同一次调用（合计约 6 秒）
& $PY "$SKILL\tools\wd_tagger.py" <图片路径> --general 0.35 --character 0.85 --json > run.json
& $PY "$SKILL\tools\anima_validate.py" --tagger-json run.json

# 组装完成后，用成品再校验一次（带 NL，算总 token）
& $PY "$SKILL\tools\anima_validate.py" --tags "<最终 tag 串>" --nl "<NL>"
```

- 阈值**固定 0.35 / 0.85**，理由见 `references/反推分支.md` §1。
- `run.json` **带每个 tag 的分数**，是铁律 6 分档的唯一依据；**校验器的输出不打印分数**，不能只看它。
- 多张图可以一次传：`wd_tagger.py img1 img2 --json`；加 `--per-record` 逐张出汇总表。
- 退出码 1 = 有需要定夺的冲突或超预算，**不是崩溃**。
- token 计数走 **T5 通道**（`models/t5_tokenizer/tokenizer.json`），**不是 Qwen**。
  512 上限两路都有，但 Anima 的条件序列长度由 T5 决定（适配器按 T5 的位置生成条件向量），
  而同一段英文 T5 切得比 Qwen 多，所以先撞线的是 T5。只数 Qwen 会漏报。
- 超预算时校验器会一并报出 tag 层占多少、NL 还剩多少预算；处置办法见 `references/格式规范.md` §五。
- tokenizer 不可用时退化成粗估，输出里标 `estimate(rough)`；文本含中文等 T5 表示不了的字符时标 `+unk` 并报警。

## 按需读取

- `references/格式规范.md` —— 落笔前必读
- `references/反推分支.md` —— 有图时
- `references/创作分支.md` —— 没图时
