---
name: anima-tagger
description: 为 Anima 等本地绘画模型产出提示词。收到图片走反推分支（打标 → 看图判断 → 写自然语言 → 组装校验），没有图片走创作分支（从需求直接写），已有提示词要加细节走扩写分支（三段式长稿，不受 512 token 上限）。反推与创作分支产出恒为「danbooru 风格 tag ＋ 末尾几句自然语言」。当用户发来图片要反推提示词 / 反推 tag / 看图写词，要写 Anima、anima、本地模型提示词，或要把现成提示词扩写 / 加细节 / 增强质感时使用。
---

# anima-tagger

| 用户给了什么 | 走哪条分支 | 查阅哪份指南 |
|---|---|---|
| **图片**（附件图片、本地路径、"照着这张图做"、"帮我反推"） | **反推分支** | `references/反推分支.md` |
| 只有文字需求（"画个猫耳少女"、"来张赛博朋克风格"） | **创作分支** | `references/创作分支.md` |
| **已经有提示词或一串 tag**，要求"扩写 / 加细节 / 增强质感 / 细节拉满" | **扩写分支** | `references/扩写分支.md` |

格式与禁用规则的终极标准是 `references/格式规范.md`，无论走哪个分支都必须不折不扣遵守。

**扩写分支只有两处例外，其余一模一样：**

- 交付形态是**三段式长稿**（标签锚点 ➔ 极致分层细节 ➔ 精炼 Master），不是"tag 流 ＋ 几句 NL"；
- **不受 T5 512 上限约束，不需要校验提示词长度**，也绝不准为了省 token 砍句子。

> 📌 图片 ＋ "照着扩写" ➔ **先走反推分支拿到 tag**，再进扩写分支，角色名以打标器输出为准；用户明确指定角色身份时例外。

---

## 核心规则

1. **分支与交付**：收到图片默认反推；反推与创作交「Danbooru 标签＋英文自然语言」，扩写交三段式长稿，均放在一个 `text` 代码块中。必要说明限代码块下方一行。
2. **角色候选直接保留**：采用打标器 `character` 输出（默认阈值 0.27），不判断身份、不搜索、不筛选或二次确认；无候选就不补人名。用户明确指定身份时，以其指定的英文角色标签为准。
3. **一般标签结合画面整理**：`general` 默认阈值 0.17，分数用于排序和参考；剔除明显误报，保留颜色、花纹、材质等独立属性轴。单人图直接看整图，复杂多人场景再按需裁切或寻求第二意见。
4. **只写可见内容**：将需求和构图意图转成画面内的对象、动作、空间与光影；不写画外拍摄说明、不可见心理或负向句。成品不含质量词、负面提示词、画师名及独立作品名标签。
5. **统一运行环境**：缺模型或依赖时运行 `python tools/setup.py`，默认自动选择：有 CUDA 用 PixAI 原版 GPU 推理（BF16，可选 FP32），无 CUDA 回退 PixAI ONNX CPU FP32 路线；输入均为 1008×1008。可用 `--device cpu` 强制 CPU。
6. **交付前校验**：由 `tools/anima_validate.py` 去重、折叠上位词并检查完整 tag＋NL。普通分支按 T5 精确计数，默认上限 512；超限先精简 NL，再减少非核心标签，保护用户锚点，不以身份判断删角色候选。修改后重跑，校验通过再交付。
7. **扩写保留锚点与形式**：保留原有关键设定，不限制长度；仍须遵守槽位、标签与标点规范，并用 `--no-token-limit` 校验。清除原稿中的禁用内容、`BREAK` 和权重语法。

---

## 工具与助手的分工

工具负责规范化、去重、上位词折叠、移除分级、已知人数/槽位冲突、明确禁用标签、NL 末尾句号和精确 token 计数。工具保持输入顺序，不自动全局排序、不判定全部词表外标签无效，也不猜画师或独立作品名。槽位排列、一般标签语义和画师/作品禁则由助手负责；角色候选直接采用打标器输出，助手不判断身份。`copyright`、`style`、`meta`、`rating` 仅作为分析数据，校验器只读 `character` 和 `general`。

多人任务读取 `references/多人提示词.md`：tag 按「人数段 → 全局段 → 各角色段」排列，末尾 NL 写清位置和动作归属。助手按画面整理人物、按角色归属采纳工具清洗建议，并对最终分段稿精确计数。

普通分支用随附 T5 分词器核算默认 512 token 预算；扩写按 `references/扩写分支.md` 的模板和长度要求执行。

## 快速上手与环境准备

### 第 0 步：定位环境路径

不管你在什么操作系统上，本工具的 Python 环境都可以快速定位：

```powershell
# 1. 如果你在 Windows PowerShell 下使用本项目的内置环境：
$SKILL = "<本项目根目录的绝对路径>"
$PY    = "$SKILL\.venv-pixai-gpu\Scripts\python.exe"

# 2. 如果你在 Linux（NVIDIA CUDA）下：
# $PY = "$SKILL/.venv-pixai-gpu/bin/python"

# 3. 如果当前虚拟环境已经激活，或者通过 pip 安装了本工具：
# $PY = "python"
```

### 首次环境部署（模型不进 Git，由脚本自动拉取）

```powershell
# 自动建 venv、装依赖、拉取 1.81 GiB 原版 权重与词表：
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
& $PY "$SKILL\tools\pixai_tagger.py" <图片绝对路径或CAS路径> --general 0.17 --character 0.27 --json | & $PY "$SKILL\tools\anima_validate.py" --tagger-json -
```

> 💡 **落盘调试规范**：
> 如果需要查看每个 tag 的具体置信度打分，**严禁散落在根目录下**！请统一收敛在 `.work/` 临时目录：
> ```powershell
> New-Item -ItemType Directory -Force -Path "$SKILL\.work" | Out-Null
> & $PY "$SKILL\tools\pixai_tagger.py" <图片路径> --general 0.17 --character 0.27 --json | Out-File "$SKILL\.work\run.json" -Encoding utf8
> & $PY "$SKILL\tools\anima_validate.py" --tagger-json "$SKILL\.work\run.json"
> ```

### 终稿全量校验（必须跑！）

在组装好最终 tag 和写好末尾的自然语言（NL）后，正式交付用户前必须跑一次终验：

```powershell
& $PY "$SKILL\tools\anima_validate.py" --tags "<排好序的完整tag串>" --nl "<末尾的自然语言段落>"
```

- 退出码为 **0**：安全通过，放心交付；
- 退出码为 **1**：按 problems 修正并重新校验；分词器不可用时先恢复精确计数环境。

### 扩写分支的校验（只查形式，不查长度）

```powershell
# 将标签之后的完整长稿写入 .work/expansion-nl.txt，保持三段式模板不变。
& $PY "$SKILL\tools\anima_validate.py" --tags "<扩写稿第一部分的完整 tag 串>" --nl-file "$SKILL\.work\expansion-nl.txt" --no-token-limit
```

- `--no-token-limit` 跳过 512 预算判定：**超长不触发预算问题，退出码不会因长度变成 1**；
- 规范化、上位词折叠、槽位冲突照常生效——该修的照修，该保留的属性轴照留；
- 此时退出码为 **1** 表示已检测到冲突、禁用内容、NL 形式问题或无法表示的字符；查看工具的 problems。

---

## 进阶与分支规范

- 拿到图片该怎么看、怎么挑词、怎么写光影？👉 查看 `references/反推分支.md`
- 只有一两句模糊需求该怎么构思、怎么防串味？👉 查看 `references/创作分支.md`
- 已有提示词怎么扩成极致细节长稿、不受 512 限制？👉 查看 `references/扩写分支.md`
- 两个及以上主体、人物互动或多人分格？👉 同时读取 `references/多人提示词.md`
- 槽位顺序、排版示例、标点规则与违禁清单？👉 查看 `references/格式规范.md`
