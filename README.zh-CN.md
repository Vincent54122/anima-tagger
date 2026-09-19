<h1 align="center">anima-tagger</h1>

<p align="center">专为 Anima 等本地双通道绘画模型设计的提示词生成器与 Agent Skill。<br>输入一张图片或文字需求，输出规范化的「Danbooru 标签流 ＋ 自然语言描述」。</p>

<p align="center">
  <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Target-Anima-purple?style=flat-square" alt="Target: Anima">
</p>

像 Anima 这类现代动漫绘图模型采用「文本标签（Qwen/T5）+ 语义描述」的双通道条件注入。写好提示词的核心挑战是：**标签词表不能瞎编、槽位顺序不能错乱、Token 预算不能超标（512 上限拦截截断）**。

`anima-tagger` 是一套既可作为 **Agent Skill**（遵循 [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills) 标准）挂载给 AI 助手，又能在命令行独立运行的工具包。它打通了双向生成路径：
- **反推分支（图片 ➔ 提示词）**：本地 WD EVA02 跑出高保真 Tag，AI 核验低置信度特征，脚本确定性校验与排版。
- **创作分支（需求 ➔ 提示词）**：AI 扮演编剧/导演/原画角色展开分镜，输出标准槽位 Tag 与空间光影 NL。

## 特性亮点

| 特性 | 说明与优势 |
|---|---|
| **机器打标，拒绝幻觉** | 标签提取自固定的 16,473 维 WD EVA02 词表，杜绝 LLM 凭空编造不存在的 Danbooru tag。 |
| **分档置信度裁决** | 打标得分 ≥ 0.8 直接无条件采信；仅对 < 0.8 的标签由视觉模型回图核验，高效精准。 |
| **确定性清洗与校验** | `tools/anima_validate.py` 自动化处理上位词折叠、反下划线、槽位冲突与规范排序，不依赖随机性。 |
| **T5 通道 Token 预算** | Anima 架构在 512 token 外直接硬件级丢弃。本工具以切词更细的 T5 分词器为基准，严格防止尾部自然语言被截断。 |
| **纯 CPU 极速运行** | ONNX 权重经过优化，单次打标 + 校验仅需约数秒，无需独占绘图显卡显存。 |

## 架构流程

```text
┌─────────────────────────┐               ┌─────────────────────────┐
│     输入：本地图片      │               │     输入：文字需求      │
└────────────┬────────────┘               └────────────┬────────────┘
             │                                         │
             ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│        反推分支         │               │        创作分支         │
│  wd_tagger.py (CPU)     │               │  LLM 展开剧本分镜       │
│  16,473 维固定词表打标  │               │  编剧 → 导演 → 摄影     │
└────────────┬────────────┘               └────────────┬────────────┘
             │ 置信度评分                              │
             ▼                                         │
┌─────────────────────────┐                            │
│    视觉判据 (Agent)     │                            │
│  ≥0.8 直接入选          │                            │
│  <0.8 回图核验真伪      │                            │
└────────────┬────────────┘                            │
             │                                         │
             └────────────────────┬────────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │     anima_validate.py     │
                    │ 1. 规范化与上位词折叠     │
                    │ 2. 槽位排序与冲突检测     │
                    │ 3. T5 通道 Token 计数     │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │      成品提示词交付       │
                    │  Danbooru 标签 + 纯净 NL  │
                    └───────────────────────────┘
```

## 输出示例

无论走反推还是创作，最终交付形态严格统一为**「标准 Tag 串 ＋ 尾部空间/光影自然语言」**：

```text
general, 1girl, solo,

ui (blue archive),

black hair, very long hair, blue eyes, halo,

white serafuku, purple neckerchief, grey cardigan,

standing, looking at viewer, holding book,

cowboy shot, depth of field,

blue background, gradient background,

Place the character slightly right of center, Use soft light from the left, Keep the face sharp against a softly blurred background.
```

> **纯净原则**：本工具拒绝产出无效的质量前缀（`masterpiece`, `best quality` 等）、画师 tag 及负面提示词，专注把控构图与细节生成。

## 快速安装

环境要求：Python 3.11 或更新版本。

```powershell
git clone https://github.com/Vincent54122/anima-tagger.git anima-tagger
cd anima-tagger
python tools/setup.py
```

`setup.py` 会自动初始化独立的 `.venv` 环境、安装依赖并拉取锁定的 ONNX 模型权重（约 1.22 GiB）。
随时可验证安装完整性：

```powershell
python tools/setup.py --check
```

## 快速上手

### 方式一：作为 Agent Skill 挂载（推荐）

将仓库整体置于 Agent（如 Claude Code / DSH）的技能目录下即可。AI 读取 `SKILL.md` 后会自动识别用户意图：
- 发送图片时 ➔ 自动触发打标与反推工作流；
- 提出构思时 ➔ 自动触发创作工序与确定性校验。

### 方式二：命令行独立使用

你可以直接调用虚拟环境中的工具进行批量打标与校验：

```powershell
$PY = ".\.venv\Scripts\python.exe"        # Linux / macOS: ./.venv/bin/python

# 1. 运行打标（输出带评分的 JSON）
& $PY tools\wd_tagger.py photo.png --general 0.35 --character 0.85 --json > run.json

# 2. 自动校验与整理打标结果
& $PY tools\anima_validate.py --tagger-json run.json

# 3. 校验包含自然语言的完整成品
& $PY tools\anima_validate.py --tags "1girl, solo, ..." --nl "Place the character..."
```

## 命令速查

| 命令 | 用途 |
|---|---|
| `python tools/setup.py` | 自动化配置环境、安装依赖与拉取模型 |
| `python tools/setup.py --check` | 检查环境完整性与模型词表（退出码 0 即就绪） |
| `tools/wd_tagger.py <img...> --json` | 本地执行 WD 标签反推，输出每个标签的分数 |
| `tools/anima_validate.py --tagger-json <json>` | 校验打标 JSON，去重、折叠上位词、排序并统计 token |
| `tools/anima_validate.py --tags "..." --nl "..."` | 全面校验最终成品（含 NL 语句），确保未突破 512 token |

## 仓库结构

```text
anima-tagger/
├── SKILL.md                     # Agent Skill 入口规范 (Agent Skills 标准)
├── references/                  # 核心提示词与工作流规范
│   ├── 格式规范.md              # 槽位顺序、Token 预算与禁则
│   ├── 反推分支.md              # 图片打标 ➔ 提示词标准流程
│   └── 创作分支.md              # 文本构思 ➔ 提示词工序
├── tools/                       # 核心可执行工具
│   ├── setup.py                 # 一键环境配置与权重下载
│   ├── wd_tagger.py             # CPU ONNX 标签反推引擎
│   ├── anima_validate.py        # 确定性标签校验与 Token 统计
│   └── requirements.txt
└── models/
    ├── t5_tokenizer/            # 内置 T5 分词器词表
    └── wd-eva02-.../            # 下载的打标权重（setup 脚本自动拉取）
```

## 常见问题排查

- **国内网络下载慢/中断**：在运行安装前配置镜像源：
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python tools/setup.py
  ```
- **Token 提示 `+unk` 警报**：表示提示词中混入了 T5 词表中无法解析的字符（如中文字符或异常符号）。由于 Anima 模型的 T5 无法识别这些标记，请确保提示词全部使用规范英文。
- **Token 预算超标处置**：若报出 `512` 溢出错误，**切勿删减 Tag 层**。Tag 属于关键特征约束，超预算通常是尾部 NL 赘述过多。请优先精简或合并自然语言描述中的修饰词。

## 开源协议

基于 [MIT License](./LICENSE) 开源发布。
第三方资产与模型分发规范参见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。
