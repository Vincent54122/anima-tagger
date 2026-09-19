# anima-tagger

给 **Anima** 等本地绘画模型写提示词的 skill。

输入一张图，产出一段 danbooru 风格 tag ＋ 末尾几句自然语言；输入一句需求，同样。产出形态固定，不给解说、不给流程。

```
general, 1girl, solo,

ui (blue archive),

black hair, very long hair, blue eyes, halo,

white serafuku, purple neckerchief, grey cardigan,

standing, looking at viewer, holding book,

cowboy shot, depth of field,

blue background, gradient background,

Place the character slightly right of center, Use soft light from the left,
Keep the face sharp against a softly blurred background.
```

## 它做什么

| 分支 | 触发 | 做什么 |
|---|---|---|
| **反推** | 给了图片 | WD EVA02 tagger（ONNX，CPU）打标 → 按分数分档（≥0.8 直接采信）→ 只对 <0.8 看图判误报与属性轴 → 写自然语言 → 组装校验 |
| **创作** | 只有文字需求 | 按 `编剧 → 监督 → 原画 → 摄影` 四道工序从需求直接写 |

反推的 tag 层是**机器打标**出来的，不是模型编的——`tools/wd_tagger.py` 跑一份 16,473 标签的 ONNX tagger；判断与自然语言才交给 LLM。校验由 `tools/anima_validate.py` 确定性完成：规范化、上位词折叠、槽位冲突、精确 token 计数。

## 快速开始

```powershell
git clone https://github.com/Vincent54122/anima-tagger anima-tagger
cd anima-tagger
python tools/setup.py
```

`setup.py` 会建 `.venv`、装依赖、下载权重与许可证、并校验词表行数，幂等可重复运行。

> 这一步用的是**系统 python**（venv 还没建）。`python` 不存在、或运行后弹出 Microsoft Store（退出码 9009）时，改用 `py -3 tools/setup.py`；再不行试 `python3`。

完成后：

```powershell
python tools/setup.py --check     # 体检，就绪返回 0
```

## 模型获取

**本仓库不包含模型权重。** 单个 `model.onnx` 约 1.22 GiB，超过 GitHub 单文件上限，必须走 Git LFS 或 Release；同时上游会持续更新，让使用者直接从 Hugging Face 取更合适。

`tools/setup.py` 按**钉死的 revision** 拉取，保证 `model.onnx` 与 `selected_tags.csv` 是同一版本：

| 项 | 值 |
|---|---|
| 模型作者 | [`ashen-sensored`](https://huggingface.co/ashen-sensored/wd-eva02-tagger-2026-canary)，**Apache-2.0** |
| ONNX 转换 | `Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2`（只做格式转换，未附 LICENSE） |
| revision | `0a86acfa093b33b8818667820e52fc5eccf27ff8` |
| 文件 | `model.onnx`（约 1.22 GiB）、`selected_tags.csv`（457 KiB，16,473 行）、`LICENSE` |
| 落点 | `models/wd-eva02-tagger-2026-canary-onnx-v2/` |

国内网络先设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python tools/setup.py
```

手动下载也一样——把上面两个文件按原样放进 `models/wd-eva02-tagger-2026-canary-onnx-v2/`，然后跑 `python tools/setup.py --check` 确认。

> ⚠️ **不要换成别的 tagger 模型。** 词表、槽位冲突表、`0.35 / 0.85` 两个阈值全是照这一份标定的。换模型后 `selected_tags.csv` 行数对不上，`wd_tagger.py` 只会报「输出维度 != 标签数」，而阈值和槽位表会**静默**失效——产出看起来正常，其实是错的。

## 用法

### 当作 skill 挂载

把仓库放进 agent 的 skills 目录，入口是 `SKILL.md`：

```powershell
# 示例：DSH / Claude Code 风格
git clone https://github.com/Vincent54122/anima-tagger "$env:USERPROFILE\.dsh\skills\anima-tagger"
python "$env:USERPROFILE\.dsh\skills\anima-tagger\tools\setup.py"
```

挂上之后，直接发图或说需求即可，不用提 skill 名字。

### 直接调脚本

```powershell
$PY = ".\.venv\Scripts\python.exe"          # macOS / Linux: ./.venv/bin/python

# 打标
& $PY tools\wd_tagger.py photo.png --general 0.35 --character 0.85 --json > run.json

# 校验（吃上一步的 JSON）
& $PY tools\anima_validate.py --tagger-json run.json

# 组装完成后，用成品再校验一次（带 NL，一起算 token）
& $PY tools\anima_validate.py --tags "<最终 tag 串>" --nl "<NL>"
```

多张图一次传：`wd_tagger.py img1 img2 --json`；加 `--per-record` 逐张出汇总表。

**退出码 1 = 有需要定夺的冲突或超预算，不是崩溃。**

## 目录结构

```
anima-tagger/
├── SKILL.md                     skill 入口（Agent Skills 标准）
├── references/
│   ├── 格式规范.md              格式与禁用项的唯一权威
│   ├── 反推分支.md              图 → 提示词
│   └── 创作分支.md              需求 → 提示词
├── tools/
│   ├── setup.py                 一键部署
│   ├── wd_tagger.py             ONNX 打标
│   ├── anima_validate.py        确定性校验器
│   └── requirements.txt
└── models/
    ├── t5_tokenizer/            随仓库提供（token 计数用）
    └── wd-eva02-.../            下载落点，仓库里没有（见 models/README.md）
```

## 环境

| 项 | 值 |
|---|---|
| Python | **3.11+**（实测 3.11.9；onnxruntime 当前要求 `>=3.11`） |
| 运行时依赖 | `onnxruntime` / `pillow` / `numpy` / `tokenizers`（`tools/setup.py` 自动装） |
| 仅部署时依赖 | `huggingface_hub`（只有 `setup.py` 下载模型时用） |
| 实测版本 | onnxruntime 1.30.0 · numpy 2.4.6 · pillow 12.3.0 · tokenizers 0.23.2 |
| 算力 | 打标走 **CPU**，不需要显卡 |
| 平台 | 在 Windows 上开发与验证；脚本与工具本身是跨平台的 |

设置 `ANIMA_TOKENIZER` 环境变量可以指定别的 `tokenizer.json`，覆盖自带的那个。

## 常见问题

**下载卡住或失败？** 设 `HF_ENDPOINT=https://hf-mirror.com` 重跑。`setup.py` 支持断点续传，中断了直接再跑一次。

**报 `httpx.InvalidURL: Invalid port: ':1]'`？** 代理工具（Clash 一类）常往 `NO_PROXY` 里写 `[::1]` 这种方括号写法，httpx 解析不了它，会让**所有**下载在发出请求之前就失败，报错也看不出跟代理有关。`setup.py` 会尝试自动去掉方括号条目；若仍失败，手动清掉再跑：

```powershell
$env:NO_PROXY = ""; $env:no_proxy = ""
python tools/setup.py
```

**`anima_validate.py` 输出里出现 `estimate(rough)`？** 说明 tokenizer 没生效，token 数是粗估（宁可高估）。检查 `models/t5_tokenizer/tokenizer.json` 是否存在。

**输出里出现 `+unk` 并报警？** 说明文本里有 T5 词表表示不了的字符（中文、日文等），那一段在 T5 通道会被丢成 `<unk>`，token 数也不代表它的真实占用。Anima 的提示词应当是英文。

**为什么数 T5 而不是 Qwen？** Anima 的文本编码器确实是 Qwen3-0.6B，但提示词要同时进两条通道：Qwen 产出语义（context），T5 只负责切出 token 位置，模型里的 LLM Adapter 按 **T5 的位置**生成条件向量：

```python
x = self.in_proj(self.embed(target_input_ids))   # 长度 = T5 切出来多少
context = source_hidden_states                   # 内容 = Qwen 读出来的
```

所以条件序列有多长由 T5 决定，而两条通道各自独立截断到 512。同一段英文，T5 切得比 Qwen 多（实测 tag 串 +2%、自然语言 +12%），**先撞线的是 T5**。只数 Qwen 会漏报——报「480/512 安全」时 T5 那路可能已经砍掉了尾巴。

**超预算了怎么办？** 精简 NL，不要动 tag。超出的部分是从尾部整块消失，而 NL 排在最后。详见 [`references/格式规范.md`](references/格式规范.md) 第五节。

**`selected_tags.csv 有 N 行，应为 16473 行`？** 模型与词表版本错配。删掉整个模型目录重下，不要改期望值去迁就。

## 第三方资产

本仓库不含任何模型权重、不含任何 Danbooru 图片。随仓库分发一份 T5 tokenizer。逐项来源、许可与归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## License

本仓库自身的代码与文档以 [MIT](LICENSE) 授权。第三方资产另按各自许可，见 THIRD_PARTY_NOTICES.md。
