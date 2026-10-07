<h1 align="center">anima-tagger</h1>

<p align="center">专为 Anima 等本地双通道绘画模型设计的提示词生成器与 Agent Skill。<br>输入一张图片或文字需求，输出规范化的「Danbooru 标签流 ＋ 自然语言描述」。</p>

<p align="center">
  <a href="./README.en.md">English</a> | <a href="./README.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Target-Anima-purple?style=flat-square" alt="Target: Anima">
</p>

Anima 支持标签、自然语言及混合提示。这个工具按统一结构组织标签与英文描述，使用随附 T5 分词器核算普通分支的默认 512 token 预算。多人提示词逐人整理外貌、服饰与动作，再用明确主语写位置和互动。

`anima-tagger` 是一套既可作为 **Agent Skill**（遵循 [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills) 标准）挂载给 AI 助手，又能在命令行独立运行的工具包。它打通了三条生成路径：
- **反推分支（图片 ➔ 提示词）**：本地 PixAI 原版在 GPU 上输出 Tag，AI 整理一般外观标签，角色候选直接保留，脚本确定性校验与排版。
- **创作分支（需求 ➔ 提示词）**：AI 扮演编剧/导演/原画角色展开分镜，输出标准槽位 Tag 与空间光影 NL。
- **扩写分支（提示词 ➔ 极致细节长稿）**：把现成的提示词或 Tag 串按「标签锚点 ➔ 极致分层细节 ➔ 精炼 Master」三段式扩写，面向 Krea2 等不受 512 限制的长提示词通道，**不做长度校验**。

## 特性亮点

| 特性 | 能帮你做什么 |
|---|---|
| **反推、创作、扩写一体** | 给参考图，提取并整理画面特征；给文字想法，写成完整提示词；给已有提示词，进一步丰富细节。按输入自动选择对应工作流。 |
| **标签与自然语言协同** | 用 Danbooru 标签明确人物、服装、道具和姿势，用英文自然语言补充空间层次、人物互动与复杂光影，按统一结构组织成可直接使用的提示词。 |
| **从简短想法展开完整画面** | 围绕题材、人物、动作、场景、构图和光影逐层构思。明确需求按原意落实，模糊想法补齐画面，开放需求提供不同方向的方案。 |
| **保留原有锚点，分层丰富细节** | 扩写时保留原稿的关键设定，细化构图姿态、发饰、表情、服装、道具、光色与背景，交付「标签锚点 → 七层细节 → 精炼整合」三段式长稿，适配不设 512 上限的长提示词通道。 |
| **只描述画面里看得见的内容** | 把抽象意图转成具体的对象、位置、遮挡、材质与受光关系；避免混入画外拍摄说明、不可见的心理解释和空泛质量词，让描述集中在希望生成的画面上。 |
| **交付前自动清洗与校验** | 自动去重、折叠上位词、规范标签格式，检查已知人数与属性冲突、禁用内容和自然语言形式，并用 T5 分词器精确核算普通分支的 512 token 预算；超限时优先精简赘述，保护关键设定。 |

## 多人提示词

tag 按「人数段 → 全局镜头、取景、场景、天气与光影段 → 角色 A 段 → 角色 B 段」排列，更多角色依次续写。每人段内集中外观、服装、饰品、道具与独立表情姿势，末尾 NL 明确位置和动作归属。具名角色补充基本外貌。

反推时保留打标器角色候选，按画面确定人数、外观与位置。校验器检查形式、冲突与预算，助手核对逐人归属，并对最终分段稿精确计数。

完整写法、三个多人示例与分格建议见 [多人提示词指南](references/多人提示词.md)。

## 架构流程

```text
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   输入：本地图片    │  │   输入：文字需求    │  │  输入：现成提示词   │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│      反推分支       │  │      创作分支       │  │      扩写分支       │
│  pixai_tagger.py GPU │  │  LLM 展开剧本分镜   │  │  LLM 七层细节扩写   │
│  30,877 维固定词表  │  │  编剧 → 导演 → 摄影 │  │  标签→分层→Master   │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │ 置信度评分             │                        │
           ▼                        │                        │
┌─────────────────────┐             │                        │
│   视觉判据 (Agent)  │             │                        │
│   角色候选直接入选     │             │                        │
│  一般标签回图整理  │             │                        │
└──────────┬──────────┘             │                        │
           │                        │                        │
           └───────────┬────────────┴────────────────────────┘
                       ▼
         ┌───────────────────────────┐
         │     anima_validate.py     │
         │ 1. 规范化与上位词折叠     │
         │ 2. 形式规则与冲突检测     │
         │ 3. T5 通道 Token 计数     │
         │ 4. 可跳过 512 长度判定    │
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │      成品提示词交付       │
         │  Danbooru 标签 + 纯净 NL  │
         │  或三段式极致细节长稿     │
         └───────────────────────────┘
```

## 输入图片

<p align="center">
  <img src="./assets/example-input.jpg" alt="示例输入：雨后天台上撑着透明伞的少女" width="420">
</p>

## 输出示例

把上面这张图交给本工具反推，最终交付形态严格统一为**「标准 Tag 串 ＋ 尾部空间/光影自然语言」**（创作分支的输出形态完全相同）：

```text
1girl, solo,

short hair, blue hair, blue eyes, parted bangs, bare legs,

transparent, translucent, see-through clothes, raincoat, coat, hood, long sleeves,
see-through sleeves, blue shirt, pleated skirt, blue skirt, hair ornament, hair flower,
white flower, transparent umbrella, white shoes, sneakers,

standing, standing on one leg, leg up, holding, holding umbrella, outstretched arm,
looking at viewer, light smile, closed mouth,

full body, from below, dutch angle,

outdoors, day, building, cloudy sky, blue sky, cumulonimbus cloud, sunlight,
backlighting, lens flare, rainbow, rain, water, puddle, reflection, water drop, bird,
balloon, plant, overgrown,

Place a girl balancing on one leg under a clear umbrella just after rain, one arm
outstretched toward the rainbow arcing overhead and the other hand gripping the umbrella
handle.
Frame the full body from a low angle, overgrown buildings rising on both sides and small
balloons drifting in the distant sky.
Use strong sunlight bursting from the upper left behind the umbrella, water drops
glittering in the backlight and the wet ground mirroring the sky.
```

> 成品用具体标签与英文描述表达构图、外观、材质、动作和光影。
>
> **所见即所得**：提示词只写最终画面里看得见的东西——用户原话、拍摄方式、机位推理都只是脚手架，得先内化成“画面上是什么”才准落笔；不写镜头外的装置，也不写“画面里没有××”这类负向句。

## 快速安装

环境要求：Python 3.11 或更新版本。默认优先使用 NVIDIA CUDA；无 CUDA（如 Intel/AMD 显卡或仅 CPU）时使用已验证的 PixAI ONNX CPU 路线。GPU 环境安装已验证的 PyTorch 2.8.0 cu128；CPU 环境无需 PyTorch。

```powershell
git clone https://github.com/Vincent54122/anima-tagger.git anima-tagger
cd anima-tagger
python tools/setup.py
```

`setup.py` 会初始化独立的 `.venv-pixai-gpu` 环境，按设备安装依赖与固定版本权重。GPU 部署包含原版模型及 ONNX CPU 回退模型；CPU 部署只下载 ONNX 模型。可用 `python tools/setup.py --device cpu` 强制部署 CPU。
随时可验证安装完整性：

```powershell
python tools/setup.py --check
```

## 快速上手

### 方式一：作为 Agent Skill 挂载（推荐）

将仓库整体置于 Agent（如 Claude Code / DSH）的技能目录下即可。AI 读取 `SKILL.md` 后会自动识别用户意图：
- 发送图片时 ➔ 自动触发打标与反推工作流；
- 提出构思时 ➔ 自动触发创作工序与确定性校验；
- 已有一份提示词、要求加细节时 ➔ 自动触发扩写工序，交三段式长稿（不做长度校验）。

### 方式二：命令行独立使用

你可以直接调用虚拟环境中的工具进行批量打标与校验：

```powershell
$PY = ".\.venv-pixai-gpu\Scripts\python.exe"        # Linux (CUDA): ./.venv-pixai-gpu/bin/python

# 1. 管道化打标与校验（推荐：无盘流式交互，零临时文件产生）
& $PY tools\pixai_tagger.py photo.png --general 0.17 --character 0.27 --json | & $PY tools\anima_validate.py --tagger-json -

# 2. 若需落盘中间结果，统一收敛在 .work/ 临时目录（已配置 .gitignore 忽略）
New-Item -ItemType Directory -Force -Path ".work" | Out-Null
& $PY tools\pixai_tagger.py photo.png --general 0.17 --character 0.27 --json | Out-File .work\run.json -Encoding utf8
& $PY tools\anima_validate.py --tagger-json .work\run.json

# 3. 校验包含自然语言的完整成品
& $PY tools\anima_validate.py --tags "1girl, solo, ..." --nl "Place the character..."
```

## 命令速查

| 命令 | 用途 |
|---|---|
| `python tools/setup.py` | 自动化配置环境、安装依赖与拉取模型 |
| `python tools/setup.py --check` | 检查环境完整性与模型词表（退出码 0 即就绪） |
| `tools/pixai_tagger.py <img...> --json` | 执行 PixAI 原版 GPU 标签反推，输出每个标签的分数 |
| `tools/anima_validate.py --tagger-json <json>` | 校验打标 JSON，移除分级、去重、折叠上位词、检查形式并统计 token（保留输入顺序） |
| `tools/anima_validate.py --tags "..." --nl "..."` | 检查已知形式、冲突和精确 token 预算；语义及槽位排列由助手检查 |
| `tools/anima_validate.py --tags "..." --no-token-limit` | 扩写分支专用：照常折叠上位词、查槽位冲突，但跳过 512 长度判定 |

## 仓库结构

```text
anima-tagger/
├── SKILL.md                     # Agent Skill 入口规范 (Agent Skills 标准)
├── references/                  # 核心提示词与工作流规范
│   ├── 格式规范.md              # 槽位顺序、Token 预算与禁则
│   ├── 反推分支.md              # 图片打标 ➔ 提示词标准流程
│   ├── 创作分支.md              # 文本构思 ➔ 提示词工序
│   ├── 多人提示词.md            # 逐人标签、动作归属与示例
│   └── 扩写分支.md              # 现成提示词 ➔ 三段式极致细节长稿（不受 512 限制）
├── tools/                       # 核心可执行工具
│   ├── setup.py                 # 一键环境配置与权重下载
│   ├── pixai_tagger.py             # PyTorch CUDA 原版推理引擎
│   ├── anima_validate.py        # 确定性标签校验与 Token 统计
│   └── requirements.txt
├── assets/                      # README 示例图片
└── models/
    ├── t5_tokenizer/            # 内置 T5 分词器词表
    └── pixai-tagger-v1.0-original/            # 下载的打标权重（setup 脚本自动拉取）
```

## 常见问题排查

- **国内网络下载慢/中断**：在运行安装前配置镜像源：
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python tools/setup.py
  ```
- **Token 提示 `+unk` 警报**：表示提示词中混入了 T5 词表中无法解析的字符（如中文字符或异常符号）。由于 Anima 模型的 T5 无法识别这些标记，请确保提示词全部使用规范英文。
- **Token 预算超标处置**：若报出 `512` 溢出错误，优先精简 NL；标签本身超限或仍超限时，精简冗余、次要及低置信度标签，保护用户明确锚点。（扩写分支不受 512 约束，加 `--no-token-limit` 即可，无需精简。）

## 开源协议

基于 [MIT License](./LICENSE) 开源发布。
第三方资产与模型分发规范参见 [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)。

默认 `--device auto`：CUDA 原版推理使用 BF16（可选 `--precision fp32`），无 CUDA 自动回退 ONNX CPU FP32；可用 `--device cpu` 强制 CPU。角色候选直接保留。批量调用可共用模型加载；JSON 的 `performance` 区分加载耗时与逐图耗时，并报告 PyTorch 显存分配。
