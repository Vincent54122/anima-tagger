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

Anima supports tags, natural-language captions, and hybrid prompts. This tool organizes tags and English descriptions consistently and uses the bundled T5 tokenizer to measure ordinary prompts against the default 512-token budget. Multi-character prompts group each subject's appearance, clothing, and actions, then use explicit subjects to describe placement and interaction.

`anima-tagger` operates both as an **Agent Skill** (compliant with the [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills) specification) and an independent CLI toolchain. It provides three prompt-generation paths:
- **Reverse Branch (Image to Prompt)**: Extracts high-confidence Danbooru tags via original PixAI on CUDA, reviews general visual features while accepting all emitted character candidates, and deterministically normalizes the output.
- **Creation Branch (Text to Prompt)**: Deconstructs abstract requirements into a production breakdown (Scriptwriter → Director → Key Animation → Cinematographer), assembling canonical tags and spatial/lighting NL sentences.
- **Expansion Branch (Prompt to Detailed Long Draft)**: Rewrites an existing prompt or tag set into a three-part draft (Tag Anchors → Layered Detail → Refined Master) for long-prompt channels such as Krea2 that are not capped at 512 tokens, so **no length validation is performed**.

## Highlights

| Feature | What it helps you do |
|---|---|
| **Image, idea, and prompt workflows** | Extract and organize features from a reference image, turn a written idea into a complete prompt, or enrich an existing prompt. The input determines the workflow. |
| **Tags and natural language working together** | Danbooru tags specify characters, clothing, props, and poses. English prose adds depth, character interactions, and complex lighting, assembled into a consistent prompt structure ready to use. |
| **Complete scenes from brief ideas** | Develop the subject, cast, action, setting, composition, and lighting step by step. Follow detailed requests faithfully, fill in missing scene elements for loose ideas, and offer distinct directions for open-ended requests. |
| **Layered detail that preserves your anchors** | Keep the original key settings while expanding composition and pose, hair and accessories, expression, clothing, props, lighting and color, and background. Deliver a three-part draft—tag anchors, seven detail layers, and an integrated refinement—for long-prompt channels without a 512-token cap. |
| **Descriptions grounded in visible content** | Translate abstract intent into concrete objects, positions, occlusion, materials, and lighting. Keep descriptions focused on the desired frame, avoiding off-frame shooting instructions, invisible psychological explanations, and generic quality boosters. |
| **Cleanup and checks before delivery** | Deduplicate tags, fold broader terms, normalize formatting, and check known count and attribute conflicts, forbidden content, and prose format. Count ordinary prompts precisely with the T5 tokenizer against the default 512-token budget, trimming redundant prose first while protecting key settings. |

## Multi-character prompts

Arrange tag blocks as subject count, global framing/scene/weather/lighting, character A, character B, and further characters as needed, with blank lines between blocks. Each character block groups appearance, clothing, accessories, props, expressions, and poses. End with English sentences assigning placement and actions explicitly. Named characters also receive basic appearance descriptions.

For image reversal, retain emitted character candidates and use the image to determine subject count, appearance, and placement. The validator checks formatting, conflicts, and budget; the assistant reviews subject ownership and counts the actual grouped final text.

See the [multi-character guide](references/多人提示词.md) for detailed instructions, three examples, and panel guidance.

## Architecture

```text
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│     Input: Image    │  │    Input: Text      │  │   Input: Prompt     │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Reverse Branch    │  │   Creation Branch   │  │  Expansion Branch   │
│  pixai_tagger.py GPU │  │  LLM Breakdown      │  │  LLM Expansion      │
│  30,877 Labels      │  │  Script → Lens      │  │  Tags → Layers → NL │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │ Confidence Scores      │                        │
           ▼                        │                        │
┌─────────────────────┐             │                        │
│  Agent Visual Check │             │                        │
│   Characters accepted     │             │                        │
│  General tags reviewed      │             │                        │
└──────────┬──────────┘             │                        │
           │                        │                        │
           └───────────┬────────────┴────────────────────────┘
                       ▼
         ┌───────────────────────────┐
         │     anima_validate.py     │
         │ 1. Normalize & fold terms │
         │ 2. Resolve slot conflicts │
         │ 3. Measure T5 token cost  │
         │ 4. Skip 512-token limit   │
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │        Final Prompt       │
         │  Danbooru Tags + Clean NL │
         │  or 3-part Expansion Draft│
         └───────────────────────────┘
```

## Input Example

<p align="center">
  <img src="./assets/example-input.jpg" alt="Example input: a girl holding a clear umbrella after the rain" width="420">
</p>

## Prompt Example

Reversing the image above yields the same strict format every branch produces — **"tag sequence + spatial/lighting natural language"**:

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

> Use concrete tags and English descriptions for composition, appearance, materials, actions, and lighting.
>
> **WYSIWYG only**: Every word in the prompt is something the model will draw. The prompt's sole source of truth is the final frame — the user's wording, the shooting setup, and camera reasoning are scaffolding to internalize first, never to copy in; no off-frame apparatus, and no negative phrasing such as "no X in frame".

## Quick Install

Requires Python 3.11 or newer. CUDA is preferred; systems without CUDA use the verified PixAI ONNX CPU route, including Intel/AMD GPU machines. GPU setup installs PyTorch 2.8.0 cu128; CPU setup does not require PyTorch.

```powershell
git clone https://github.com/Vincent54122/anima-tagger.git anima-tagger
cd anima-tagger
python tools/setup.py
```

`setup.py` initializes `.venv-pixai-gpu`, installs dependencies for the selected device, and downloads pinned assets. GPU deployment downloads only original weights; CPU deployment downloads only ONNX assets. When tagging falls back to CPU, missing ONNX assets are downloaded automatically and verified files are reused. Use `python tools/setup.py --device cpu` to force CPU deployment. Verify setup integrity anytime with:

```powershell
python tools/setup.py --check
```

## Quick Start

### 1. As an Agent Skill (Recommended)

Place the repository in your Agent's skills directory. When the agent loads `SKILL.md`, it automatically selects the optimal workflow:
- Image provided ➔ executes the tagger and verifies features.
- Text request provided ➔ crafts structured tags and scene directions.
- Existing prompt to enrich ➔ runs the expansion workflow and returns a three-part long draft (no length check).

### 2. Standalone CLI Usage

Drive the local tagger and validator directly:

```powershell
$PY = ".\.venv-pixai-gpu\Scripts\python.exe"        # Linux (CUDA): ./.venv-pixai-gpu/bin/python

# 1. Piped tagging & validation (Recommended: streaming pipe without temporary files)
& $PY tools\pixai_tagger.py photo.png --general 0.17 --character 0.27 --json | & $PY tools\anima_validate.py --tagger-json -

# 2. Or save intermediate artifacts to the managed .work/ temporary directory
New-Item -ItemType Directory -Force -Path ".work" | Out-Null
& $PY tools\pixai_tagger.py photo.png --general 0.17 --character 0.27 --json | Out-File .work\run.json -Encoding utf8
& $PY tools\anima_validate.py --tagger-json .work\run.json

# 3. Validate full output including natural language sentences
& $PY tools\anima_validate.py --tags "1girl, solo, ..." --nl "Place the character..."
```

## Commands

| Command | Description |
|---|---|
| `python tools/setup.py` | Sets up venv, installs requirements, downloads model weights |
| `python tools/setup.py --check` | Validates environment and label table (exits 0 if ready) |
| `tools/pixai_tagger.py <images...> --json` | Runs original PixAI CUDA tagger; emits per-tag confidence scores |
| `tools/anima_validate.py --tagger-json <json>` | Cleans tags, folds hypernyms, checks conflicts, counts tokens |
| `tools/anima_validate.py --tags "..." --nl "..."` | Known format/conflict checks and exact token budget; semantics and slot arrangement require assistant review |
| `tools/anima_validate.py --tags "..." --no-token-limit` | Expansion Branch: still folds terms and checks slot conflicts, but skips the 512-token check |

## Repository Layout

```text
anima-tagger/
├── SKILL.md                     # Skill entry point (Agent Skills standard)
├── references/                  # Core prompt guidelines and workflows
│   ├── 格式规范.md              # Format rules, slot orders, and banned tags
│   ├── 反推分支.md              # Image-to-prompt reverse pipeline
│   ├── 创作分支.md              # Text-to-prompt creative breakdown
│   ├── 多人提示词.md            # Grouped tags, subject ownership, and examples
│   └── 扩写分支.md              # Prompt-to-draft expansion (no 512-token cap)
├── tools/                       # Python toolchain
│   ├── setup.py                 # Idempotent setup and weight downloader
│   ├── pixai_tagger.py             # PyTorch CUDA tagging engine
│   ├── anima_validate.py        # Deterministic validator & tokenizer
│   └── requirements.txt
├── assets/                      # README example images
└── models/
    ├── t5_tokenizer/            # Bundled T5 tokenizer assets
    └── pixai-tagger-v1.0-original/            # Downloaded model weights (via setup.py)
```

## Troubleshooting

- **Slow / Interrupted Downloads**: Use a mirror endpoint before running setup:
  ```powershell
  $env:HF_ENDPOINT = "https://hf-mirror.com"
  python tools/setup.py
  ```
- **Token `+unk` Warning**: Indicates characters outside the T5 vocabulary (e.g., non-Latin or unsupported symbols). Ensure prompts are composed in clean English.
- **Token Budget Exceeded**: When token count exceeds `512`, shorten NL first. If tags alone exceed the budget or the prompt still does not fit, reduce redundant and secondary tags while protecting explicit user anchors. (The Expansion Branch is exempt: add `--no-token-limit` and shorten nothing.)

## License

Released under the [MIT License](./LICENSE).
For third-party model assets and license details, see [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

Rating scores stay in analysis metadata. The validator checks formatting, conflicts, and budget; the assistant reviews semantics and slot arrangement. Restore the exact tokenizer before validating the ordinary prompt budget.

Default `--device auto` uses original CUDA BF16 inference (optional FP32) or falls back to ONNX CPU FP32 when CUDA is absent. Use `--device cpu` to force CPU. Retain emitted character candidates. Process multiple images together to share model loading. JSON `performance` separates model loading from per-image timing and reports PyTorch memory allocations.
