---
name: anima-tagger
description: 为 Anima 等本地绘画模型产出提示词。收到图片走反推分支（打标 → 看图判断 → 写自然语言 → 组装校验），没有图片走创作分支（从需求直接写）。产出恒为「danbooru 风格 tag ＋ 末尾几句自然语言」。当用户发来图片要反推提示词 / 反推 tag / 看图写词，或要写 Anima、anima、本地模型提示词时使用。
---

# anima-tagger

| 用户给了什么 | 走哪条分支 | 查阅哪份指南 |
|---|---|---|
| **图片**（附件图片、本地路径、"照着这张图做"、"帮我反推"） | **反推分支** | `references/反推分支.md` |
| 只有文字需求（"画个猫耳少女"、"来张赛博朋克风格"） | **创作分支** | `references/创作分支.md` |

格式与禁用规则的终极标准是 `references/格式规范.md`，无论走哪个分支都必须不折不扣遵守。

---

## 十条硬核铁律（必须焊死在心里的操作准则）

1. **见图就推**：只要用户发了图片，**直接默认启动反推分支**开始分析，千万别多嘴反问“你是想要反推还是创作”。
2. **输出只给两部分**：只输出标准的「Danbooru 标签流 ＋ 末尾几句英文自然语言」，**整体打包在一个 \`\`\`text 代码块里**。不给冗长解说，不给过程废话。
3. **四类违禁词碰都不要碰**：坚决不输出质量前缀（`masterpiece` / `best quality` 等）、负面提示词（Negative）、画师名字（Artist tags）、独立作品名标签。
4. **看图判断自己做**：图片就在你的上下文里，单人图直接看整图，千万别无意义地反复切图浪费上下文；只有在超大多人复杂场景或需要第二意见时才调用子代理。
5. **角色是谁绝对别自己猜**：反推图片时，**完全采信打标器的 `character` 输出**——它识别出了角色名就用，没识别出来就当原创角色处理。**绝对禁止凭借个人印象瞎认人、补人名，更不要用中文名糊弄上去！**
6. **置信度 0.8 分水岭（省时核心）**：
   - 打标分数 **≥ 0.8** 的标签：**无条件直接采信**，不用核对原图、不怀疑、不乱删；
   - 只有 **< 0.8** 的低分标签才需要你对照原图辨别：剔除真误报，保留不同属性轴（如颜色与图案并存）。
7. **去重和上位词折叠交给机器**：不用人工肉眼数词去重，统一交给 `tools/anima_validate.py` 自动处理。
8. **拿不准的在末尾留一行附注**：有存疑的地方（比如低置信度的饰品删了），在代码块下方另起一行简要说明，不超过一行。
9. **环境缺失只认 setup 脚本**：找不到模型权重或依赖报错时，统一跑 `python tools/setup.py`。严禁自行乱下其他 tagger 模型，阈值 `0.35 / 0.85` 和 16,473 维词表是深度绑定的。
10. **超预算只砍自然语言，不碰标签**：校验器报超过 512 token 时，**只能精简最后那几句自然语言（NL），坚决不准动前面的 tag**！精简后必须重跑校验，直到安全通过。

---

## 快速上手与环境准备

### 第 0 步：定位环境路径

不管你在什么操作系统上，本工具的 Python 环境都可以快速定位：

```powershell
# 1. 如果你在 Windows PowerShell 下使用本项目的内置环境：
$SKILL = "<本项目根目录的绝对路径>"
$PY    = "$SKILL\.venv\Scripts\python.exe"

# 2. 如果你在 Linux / macOS 下：
# $PY = "$SKILL/.venv/bin/python"

# 3. 如果当前虚拟环境已经激活，或者通过 pip 安装了本工具：
# $PY = "python"
```

### 首次环境部署（模型不进 Git，由脚本自动拉取）

```powershell
# 自动建 venv、装依赖、拉取 1.22 GiB ONNX 权重与词表：
python "$SKILL\tools\setup.py"

# 体检检查（就绪后返回退出码 0）：
python "$SKILL\tools\setup.py" --check
```
*(若国内网络下载缓慢，可先设置镜像：`$env:HF_ENDPOINT = "https://hf-mirror.com"`)*

---

## 工具调用标准范式

### 反推工作流（推荐：无盘管道，零临时文件产生）

```powershell
# 一条管道完成：本地打标 ➔ 流式传输 ➔ 确定性清洗校验
& $PY "$SKILL\tools\wd_tagger.py" <图片绝对路径或CAS路径> --general 0.35 --character 0.85 --json | & $PY "$SKILL\tools\anima_validate.py" --tagger-json -
```

> 💡 **落盘调试规范**：
> 如果需要查看每个 tag 的具体置信度打分，**严禁散落在根目录下**！请统一收敛在 `.work/` 临时目录：
> ```powershell
> New-Item -ItemType Directory -Force -Path "$SKILL\.work" | Out-Null
> & $PY "$SKILL\tools\wd_tagger.py" <图片路径> --general 0.35 --character 0.85 --json | Out-File "$SKILL\.work\run.json" -Encoding utf8
> & $PY "$SKILL\tools\anima_validate.py" --tagger-json "$SKILL\.work\run.json"
> ```

### 终稿全量校验（必须跑！）

在组装好最终 tag 和写好末尾的自然语言（NL）后，正式交付用户前必须跑一次终验：

```powershell
& $PY "$SKILL\tools\anima_validate.py" --tags "<排好序的完整tag串>" --nl "<末尾的自然语言段落>"
```

- 退出码为 **0**：安全通过，放心交付；
- 退出码为 **1**：发现槽位冲突或 Token 突破 512 上限，必须按照 `references/格式规范.md` 砍短 NL 重新校验！

---

## 进阶与分支规范

- 拿到图片该怎么看、怎么挑词、怎么写光影？👉 查看 `references/反推分支.md`
- 只有一两句模糊需求该怎么构思、怎么防串味？👉 查看 `references/创作分支.md`
- 槽位顺序、排版示例、标点规则与违禁清单？👉 查看 `references/格式规范.md`
