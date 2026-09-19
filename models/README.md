# models/

放模型资产。**权重不入库**——它们由 `tools/setup.py` 从 Hugging Face 拉取。

```
models/
├── t5_tokenizer/
│   └── tokenizer.json                        ← 入库（2.4 MiB，token 计数用）
└── wd-eva02-tagger-2026-canary-onnx-v2/      ← 不入库，下载落点
    ├── model.onnx                            （约 1.22 GiB）
    ├── selected_tags.csv                     （457 KiB）
    └── LICENSE                               （Apache-2.0，取自模型作者仓库）
```

## t5_tokenizer/tokenizer.json

入库。**Anima 的 512 上限按这条通道计。**

来源 `google/t5-v1_1-xxl` 的 `spiece.model`（SentencePiece），转成 `tokenizer.json`
格式以便复用已有的 `tokenizers` 依赖、不再额外引入 `sentencepiece`。
转换后已逐条核对：与官方 `T5Tokenizer` 对同一批文本的编码结果**逐个 token id 相同**。

| 项 | 值 |
|---|---|
| 大小 | 2,424,069 bytes |
| 词表 | 32,100 |
| 上游 | [`google/t5-v1_1-xxl`](https://huggingface.co/google/t5-v1_1-xxl) |
| `spiece.model` SHA256 | `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86` |

### 为什么不是 Qwen tokenizer

Anima 的文本编码器是 Qwen3-0.6B，这个没错。但提示词要**同时**进两条通道：

- **Qwen3-0.6B** 产出语义隐状态（context）——负责"读懂"
- **T5 tokenizer** 只负责切出 token 位置——负责"数格子"

模型里的 LLM Adapter 按 **T5 的位置序列**生成条件向量：

```python
x = self.in_proj(self.embed(target_input_ids))   # 长度 = T5 切出来多少
context = source_hidden_states                   # 内容 = Qwen 读出来的
```

所以**条件序列有多长由 T5 决定**，而两条通道各自独立截断到 512
（`text_encoding.py` 里 T5 那路 `ids[:512]`、Qwen 那路 `truncation=True, max_length=512`）。
同一段英文 T5 切得比 Qwen 多，**先撞线的是 T5**：

| 同一段文字 | Qwen | T5 |
|---|---|---|
| tag 串 | 53 | 54 |
| 英文自然语言 | 25 | 28 |
| 长提示词 | 181 | 210 |

只数 Qwen 会漏报：报「480/512 安全」的时候，T5 那路已经砍掉了尾巴——而被砍的是
槽位顺序最后的 NL。

### 一个限制

T5 词表只覆盖拉丁字母语言（英/德/法/罗）。中文、日文等会落到 `<unk>`，
**内容在这一路直接丢失**。校验器检测到 `<unk>` 会标 `+unk` 并报警。
Anima 的提示词应当是英文。

详见 [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## wd-eva02-tagger-2026-canary-onnx-v2/

下载落点。获取方式：

```powershell
python tools/setup.py
```

国内网络先设镜像 `$env:HF_ENDPOINT = "https://hf-mirror.com"`。

**不要换成别的 tagger 模型。** 词表、槽位冲突表、`0.35 / 0.85` 阈值全是照这一份标定的。

### 钉死的版本

| 文件 | 大小 | SHA256 |
|---|---|---|
| `model.onnx` | 1,309,248,037 bytes | `fd78fbdf9390cbd163e4dd28f754a5bbf83bc7a111c4d20270f22415a0f66c95` |
| `selected_tags.csv` | 467,782 bytes | `3f78c28ee0d50779edb320733f76aeaf4184694cbd09c631deef6889865f9178` |

上游：`Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2`
@ `0a86acfa093b33b8818667820e52fc5eccf27ff8`

### 自己核对

```powershell
python tools/setup.py --check                 # 查存在性 + 大小 + 词表行数（应为 16473）
python tools/setup.py --check --verify-hash   # 再逐字节核对 SHA256（读 1.3 GiB，慢）
```

### 为什么必须同版本

`model.onnx` 的输出维度必须等于 `selected_tags.csv` 的行数。两者错配时
`tools/wd_tagger.py` 只会报一句「输出维度 != 标签数」，很难自己 debug 出来，
所以 `setup.py` 把 revision 和 SHA256 都钉死了。

### LICENSE

模型本体是 `ashen-sensored/wd-eva02-tagger-2026-canary`，**Apache-2.0**，该仓库随附 `LICENSE`
文件，`setup.py` 会把它一并取回落进本目录：

| 项 | 值 |
|---|---|
| 来源 | `ashen-sensored/wd-eva02-tagger-2026-canary` |
| revision | `c45a59a3f17c0ca6066072b1c213e0c12a90e242` |
| 大小 | 11,358 bytes |
| SHA256 | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

ONNX 仓库（`Misaka41Z/…`）只做了格式转换，自身没有放 `LICENSE` 文件，所以授权以模型作者那份为准。
授权链的完整描述见 [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
