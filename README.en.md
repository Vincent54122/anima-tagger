<h1 align="center">anima-tagger</h1>

<p align="center">A specialized Agent Skill and toolchain for local dual-channel anime diffusion models such as Anima.<br>Transforms an image or a creative request into standardized "Danbooru tag stream + natural language" prompts.</p>

<p align="center">
  <a href="./README.en.md">English</a> | <a href="./README.md">简体中文</a>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Target-Anima-purple?style=flat-square" alt="Target: Anima">
</p>

Modern anime diffusion models like Anima use dual-channel text conditioning (Qwen/T5 tokens paired with natural language). Crafting robust prompts requires strictly adhering to vocabulary standards, maintaining precise slot order, and staying within the hardware 512-token limit to avoid tail truncation.

`anima-tagger` operates both as an **Agent Skill** (compliant with the [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills) specification) and an independent CLI toolchain. It provides seamless bidirectional prompt generation:
- **Reverse Branch (Image to Prompt)**: Extracts high-confidence Danbooru tags via local WD EVA02, resolves low-confidence features with vision confirmation, and deterministically normalizes the output.
- **Creation Branch (Text to Prompt)**: Deconstructs abstract requirements into a production breakdown (Scriptwriter → Director → Key Animation → Cinematographer), assembling canonical tags and spatial/lighting NL sentences.

## Highlights

| Highlight | Benefit |
|---|---|
| **Machine-Tagged Accuracy** | Tags are drawn exclusively from a 16,473-label WD EVA02 vocabulary, eliminating LLM hallucination of invalid Danbooru tags. |
| **Tiered Confidence Decisions** | Tags scoring ≥ 0.8 are accepted unconditionally; only tags scoring < 0.8 undergo visual verification. |
| **Deterministic Validation** | `tools/anima_validate.py` handles hypernym folding, underscore cleaning, slot conflict resolution, and canonical ordering without LLM non-determinism. |
| **T5 Token Budgeting** | Anima truncates prompts strictly past 512 tokens. The bundled T5 tokenizer ensures accurate token counts, preventing critical NL descriptions from being clipped. |
| **Lightweight CPU Execution** | Optimized ONNX models run entirely on CPU in seconds, leaving your GPU VRAM untouched for image generation. |

## Architecture

```text
┌─────────────────────────┐               ┌─────────────────────────┐
│       Input: Image      │               │   Input: Text Request   │
└────────────┬────────────┘               └────────────┬────────────┘
             │                                         │
             ▼                                         ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│     Reverse Branch      │               │     Creation Branch     │
│  wd_tagger.py (CPU)     │               │  LLM Scene Breakdown    │
│  16,473 Label Lexicon   │               │  Script → Direct → Lens │
└────────────┬────────────┘               └────────────┬────────────┘
             │ Confidence Scores                       │
             ▼                                         │
┌─────────────────────────┐                            │
│   Agent Visual Check    │                            │
│  ≥0.8 Accepted directly │                            │
│  <0.8 Verified on image │                            │
└────────────┬────────────┘                            │
             │                                         │
             └────────────────────┬────────────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │     anima_validate.py     │
                    │ 1. Normalize & fold terms │
                    │ 2. Resolve slot conflicts │
                    │ 3. Measure T5 token cost  │
                    └─────────────┬─────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │       Final Prompt        │
                    │  Danbooru Tags + Clean NL │
                    └───────────────────────────┘
```

## Prompt Example

Regardless of the generation branch, outputs follow a strict **"tag sequence + spatial/lighting natural language"** format:

```text
general, 1girl, solo,

short hair, blue hair, blue eyes, parted bangs, bare legs,

transparent, translucent, see-through clothes, raincoat, coat, hood, long sleeves, see-through sleeves, blue shirt, pleated skirt, blue skirt, hair ornament, hair flower, white flower, transparent umbrella, white shoes, sneakers,

standing, standing on one leg, leg up, holding, holding umbrella, outstretched arm, looking at viewer, light smile, closed mouth,

full body, from below, dutch angle,

outdoors, day, building, cloudy sky, blue sky, cumulonimbus cloud, sunlight, backlighting, lens flare, rainbow, rain, water, puddle, reflection, water drop, bird, balloon, plant, overgrown,

Place a girl balancing on one leg under a clear umbrella just after rain, one arm outstretched toward the rainbow arcing overhead and the other hand gripping the umbrella handle. Frame the full body from a low angle, overgrown buildings rising on both sides and small balloons drifting in the distant sky. Use strong sunlight bursting from the upper left behind the umbrella, water drops glittering in the backlight and the wet ground mirroring the sky.
```

> **Punctuation**: Tags are always joined with `", "`; each NL sentence is a complete English sentence, separated and terminated by a period `.` rather than a comma.
>
> **Design Principle**: Prohibits generic quality boosters (`masterpiece`, `best quality`), artist tags, and negative prompts, ensuring clean, direct model conditioning.

## Quick Install

Requires Python 3.11 or newer.

```powershell
git clone https://github.com/Vincent54122/anima-tagger.git anima-tagger
cd anima-tagger
python tools/setup.py
```

`setup.py` initializes a self-contained `.venv`, installs dependencies, and downloads the pinned ONNX weights (~1.22 GiB). Verify setup integrity anytime with:

```powershell
python tools/setup.py --check
```

## Quick Start

### 1. As an Agent Skill (Recommended)

Place the repository in your Agent's skills directory. When the agent loads `SKILL.md`, it automatically selects the optimal workflow:
- Image provided ➔ executes the tagger and verifies features.
- Text request provided ➔ crafts structured tags and scene directions.

### 2. Standalone CLI Usage

Drive the local tagger and validator directly:

```powershell
$PY = ".\.venv\Scripts\python.exe"        # Linux / macOS: ./.venv/bin/python

# 1. Piped tagging & validation (Recommended: streaming pipe without temporary files)
& $PY tools\wd_tagger.py photo.png --general 0.35 --character 0.85 --json | & $PY tools\anima_validate.py --tagger-json -

# 2. Or save intermediate artifacts to the managed .work/ temporary directory
New-Item -ItemType Directory -Force -Path ".work" | Out-Null
& $PY tools\wd_tagger.py photo.png --general 0.35 --character 0.85 --json | Out-File .work\run.json -Encoding utf8
& $PY tools\anima_validate.py --tagger-json .work\run.json

# 3. Validate full output including natural language sentences
& $PY tools\anima_validate.py --tags "1girl, solo, ..." --nl "Place the character..."
```

## Commands

| Command | Description |
|---|---|
| `python tools/setup.py` | Sets up venv, installs requirements, downloads model weights |
| `python tools/setup.py --check` | Validates environment and label table (exits 0 if ready) |
| `tools/wd_tagger.py <images...> --json` | Runs local ONNX tagger; emits per-tag confidence scores |
| `tools/anima_validate.py --tagger-json <json>` | Cleans tags, folds hypernyms, checks conflicts, counts tokens |
| `tools/anima_validate.py --tags "..." --nl "..."` | Full check on final prompt ensuring it fits the 512-token limit |

## Repository Layout

```text
anima-tagger/
├── SKILL.md                     # Skill entry point (Agent Skills standard)
├── references/                  # Core prompt guidelines and workflows
│   ├── 格式规范.md              # Format rules, slot orders, and banned tags
│   ├── 反推分支.md              # Image-to-prompt reverse pipeline
│   └── 创作分支.md              # Text-to-prompt creative breakdown
├── tools/                       # Python toolchain
│   ├── setup.py                 # Idempotent setup and weight downloader
│   ├── wd_tagger.py             # ONNX CPU tagging engine
│   ├── anima_validate.py        # Deterministic validator & tokenizer
│   └── requirements.txt
└── models/
    ├── t5_tokenizer/            # Bundled T5 tokenizer assets
    └── wd-eva02-.../            # Downloaded model weights (via setup.py)
```

## Troubleshooting

- **Slow / Interrupted Downloads**: Use a mirror endpoint before running setup:
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python tools/setup.py
  ```
- **Token `+unk` Warning**: Indicates characters outside the T5 vocabulary (e.g., non-Latin or unsupported symbols). Ensure prompts are composed in clean English.
- **Token Budget Exceeded**: When token count exceeds `512`, **do not reduce tags**. Tags provide essential anchors. Shorten or merge the natural language sentences at the end instead.

## License

Released under the [MIT License](./LICENSE).
For third-party model assets and license details, see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
