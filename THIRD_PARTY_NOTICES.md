# 第三方资产与归属

本仓库（`SKILL.md`、`references/`、`tools/`）自身的代码与文档以 [MIT](LICENSE) 授权。
下面列出的第三方资产**不适用 MIT**，各自按本文件所述许可与条件使用。

本仓库**不包含任何模型权重**，**不包含任何 Danbooru 图片或 Danbooru 数据集**。

---

## 1. 随本仓库分发的资产

### T5 tokenizer

| 项 | 值 |
|---|---|
| 文件 | `models/t5_tokenizer/tokenizer.json` |
| 大小 | 2,424,069 bytes |
| SHA256 | `2432e8aa83414a69884a5cbd54a68f528c042d1f30780affbefcbb85091f022e` |
| 上游 | [`google/t5-v1_1-xxl`](https://huggingface.co/google/t5-v1_1-xxl) |
| revision | `3db67ab1af984cf10548a73467f0e5bca2aaaeb2` |
| 许可 | Apache-2.0（上游 metadata 声明；**该仓库内未附 LICENSE 文件**，故此处不列版权人） |

用途：`tools/anima_validate.py` 借它核算 Anima 的 512 token 预算。

上游提供的是 SentencePiece 格式的 `spiece.model`（791,656 bytes，
SHA256 `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86`）。
本仓库分发的是**由它转换而来的 `tokenizer.json`**，目的是复用已有的 `tokenizers`
依赖，不再额外引入 `sentencepiece`。

转换结果已核对：与官方 `T5Tokenizer` 对同一批文本（tag 串、自然语言、权重语法、
括号、下划线、标点、空串）的编码结果**逐个 token id 相同**。

> 该词表只覆盖拉丁字母语言。中文、日文等会落到 `<unk>`，内容在这一通道直接丢失；
> 校验器检测到会标 `+unk` 并报警。

> **为什么是 T5 而不是 Qwen**：Anima 的文本编码器确实是 Qwen3-0.6B，
> 但提示词同时进两条通道——Qwen 产出语义（context），T5 只切 token 位置，
> 模型里的 LLM Adapter 按 T5 的位置序列生成条件向量。两条通道各自截断到 512，
> 而同一段英文 T5 切得更多，所以 **T5 先撞线**。详见 [models/README.md](models/README.md)。

---

## 2. 不随本仓库分发、由使用者自行获取的资产

### WD EVA02 Tagger 2026 Canary（模型本体）

| 项 | 值 |
|---|---|
| 作者 | [`ashen-sensored`](https://huggingface.co/ashen-sensored/wd-eva02-tagger-2026-canary) |
| revision | `c45a59a3f17c0ca6066072b1c213e0c12a90e242` |
| 许可 | **Apache-2.0**（上游 metadata 声明 `license:apache-2.0`，且随附 `LICENSE` 文件） |
| 说明 | 16,473 标签（2,205 character ＋ 3,794 general 为新增） |
| 训练数据截止 | 2026-05-18（danbooru ID 11403645） |
| 标签元数据来源 | [`u-haru/danbooru-tags-20260518`](https://huggingface.co/datasets/u-haru/danbooru-tags-20260518) |
| 基座模型 | [`SmilingWolf/wd-eva02-large-tagger-v3`](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)（Apache-2.0） |

### ONNX 导出（本项目实际加载的文件）

| 项 | 值 |
|---|---|
| 上游 | [`Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2`](https://huggingface.co/Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2) |
| revision | `0a86acfa093b33b8818667820e52fc5eccf27ff8` |
| 文件 | `model.onnx`（1,309,248,037 bytes，SHA256 `fd78fbdf9390cbd163e4dd28f754a5bbf83bc7a111c4d20270f22415a0f66c95`）<br>`selected_tags.csv`（467,782 bytes，SHA256 `3f78c28ee0d50779edb320733f76aeaf4184694cbd09c631deef6889865f9178`） |
| 许可 | 沿用模型本体的 **Apache-2.0** |
| 性质 | 对上述模型的 **ONNX 格式转换**；本项目加载的就是这一份 |

> 该 ONNX 仓库未附 `LICENSE` 文件，其创作者做的是格式转换、并非模型作者，
> 因此授权以**模型本体**（`ashen-sensored`，Apache-2.0）为准。
> `tools/setup.py` 会把模型作者随附的 `LICENSE` 文本一并取回落进模型目录。
> 使用者直接从上游获取权重，与本项目之间不存在再分发关系。

---

## 3. 关于标签词表

`selected_tags.csv` 中的标签名派生自 Danbooru 的 tag 体系，随上述 ONNX 导出一并提供。
本仓库不主张对标签名本身的任何权利，也不分发 Danbooru 的图库或数据集。

---

## 4. 上游 NOTICE

Apache-2.0 第 4(d) 条：若上游随作品提供了 `NOTICE` 文件，再分发时需一并保留。
本文件核对过的四个上游仓库——`google/t5-v1_1-xxl`、`SmilingWolf/wd-eva02-large-tagger-v3`、
`ashen-sensored/wd-eva02-tagger-2026-canary`、`Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2`
——**均未附 `NOTICE` 文件**，因此目前没有需要转抄的 NOTICE 内容。
其中只有 `ashen-sensored/wd-eva02-tagger-2026-canary` 附了 `LICENSE` 文件。
若上游后续补充 `NOTICE`，本仓库的使用者应一并遵守。
