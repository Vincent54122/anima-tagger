# models/

放模型资产。**权重不入库**——它们由 `tools/setup.py` 从 Hugging Face 拉取。

```
models/
├── t5_tokenizer/tokenizer.json   # bundled prompt tokenizer
└── pixai-tagger-v1.0-original/   # downloaded, ignored by Git
    ├── model.safetensors        # 1,945,425,796 bytes (1.81 GiB)
    ├── config.json              # ordered tags + category splits
    ├── preprocessor_config.json
    ├── tagger_pipeline.py       # pinned official model/processor code
    └── README.md               # upstream model card
```

## t5_tokenizer/tokenizer.json

入库。本项目针对的 Anima 配置按这条通道核算 512 预算。下述架构说明适用于本项目的目标配置；使用其他模型版本或前端时，应核对实际编码方式和截断设置。

来源 `google/t5-v1_1-xxl` 的 `spiece.model`（SentencePiece），转成 `tokenizer.json`
格式，使用 `tokenizers` 依赖进行计数，无需额外安装 `sentencepiece`。

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
两条通道的 token 数可能不同。预算检查应使用目标通道的分词器，不能用 Qwen 的计数代替 T5；实际截断设置以所用模型和前端为准。

### 一个限制

当前分词器对无法表示的字符会产生 `<unk>`，不能从该 token 还原原字符；是否可表示以实际分词结果为准。校验器检测到 `<unk>` 会标 `+unk` 并报警。
Anima 的提示词应当是英文。

详见 [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## PixAI 原版模型

运行 `python tools/setup.py` 下载，`--check` 检查 CUDA 环境、词表、代码与配置；`--check --verify-hash` 另核对完整权重 SHA256。

上游：<https://huggingface.co/pixai-labs/pixai-tagger-v1.0>，固定 revision `9fe10addf9326e292da8a85a98ea74cd91b41771`。
全部文件哈希保存在 `tools/pixai_config.py`。推理前校验官方代码与配置，只从本地加载；默认 CUDA BF16，支持 FP32。
使用官方 1008×1008 预处理：等比例缩放、黑色补边、RGB 归一化；透明图先合成白底。EXIF 方向在预处理前纠正。

无 CUDA 时使用 PixAI ONNX CPU FP32 推理。模型卡声明 Apache-2.0，详见第三方归属说明。

## PixAI ONNX CPU 回退

目录 `models/pixai-tagger-v1.0-onnx/`；上游 `noaione/pixai-tagger-v1.0-onnx`，固定 revision `68e8f4f02dd56a5f40c1b7474489fa0f599dec34`。文件为 `model.onnx`、`model.onnx.data`、`tags.json`、`README.md`，哈希见 `tools/pixai_config.py`。ONNX Runtime 1.30.0、CPUExecutionProvider、默认 8 线程、FP32，采用 PIL 双线性缩放与补边。CPU 部署下载这套文件；GPU 部署仅准备原版，打标回退到 CPU 时按需下载 ONNX，并复用已校验的文件。
