#!/usr/bin/env python
"""Anima 提示词确定性校验器（不依赖 LLM）。

设计原则：**只报不猜**。
  * 机械可判定的（大小写 / 下划线 / 分隔符 / 上位词折叠 / token 预算）→ 直接给修正结果
  * 需要看图的语义冲突（发饰颜色到底是蓝还是紫）→ **只标记**，把决定权交回看图的那一步

用法:
    # 直接吃 wd_tagger.py 的 --json 输出
    python anima_validate.py --tagger-json run.json
    # 或给一个逗号分隔的 tag 串
    python anima_validate.py --tags "1girl, solo, long_hair"
    # NL 段也一起算 token
    python anima_validate.py --tagger-json x.json --nl "Place the subject slightly right of center."

退出码: 0=无问题, 1=有需要人工处理的冲突/超预算, 2=输入错误
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEP = ", "
TOKEN_LIMIT = 512  # Anima 硬编码上限。两路文本通道都是 512，先到线的是 T5，见 count_tokens

# T5 sentencepiece 的 <unk> id。T5 词表只覆盖拉丁字母语言，
# 中文、日文、带变音符号的字母都会落到这个 id 上——**内容在这一路已被丢弃**，
# 此时 token 数不再代表那段文字的真实占用。
T5_UNK_ID = 2

# ---------------------------------------------------------------- token 计数

def count_tokens(text: str) -> tuple[int | None, str]:
    """按 T5 通道计数；tokenizer 不可用则退回启发式估算。

    返回 (token 数, 计数方式)。tokenizer 来源优先级：
      ① 环境变量 ANIMA_TOKENIZER 指定的 tokenizer.json
      ② skill 自带的 models/t5_tokenizer/tokenizer.json

    **为什么数 T5 而不是 Qwen**：Anima 的提示词同时进两条通道——Qwen3-0.6B
    产出语义隐状态（context），T5 只负责切出 token 位置。模型里的 LLM Adapter
    按 T5 的位置序列生成条件向量：

        x = self.in_proj(self.embed(target_input_ids))   # 长度 = T5 切出多少
        context = source_hidden_states                   # 内容 = Qwen 读出来的

    所以**条件序列有多长由 T5 决定**，而两条通道各自独立截断到 512。同一段英文，
    T5 切出来总比 Qwen 多（实测 tag 串 +2%、自然语言 +12%、长提示词 +16%），
    即 T5 先撞线。只数 Qwen 会低估，报「480/512 安全」时 T5 那路已经砍掉尾巴。
    """
    cand: list[Path] = []
    if env := os.environ.get("ANIMA_TOKENIZER"):
        cand.append(Path(env))
    cand.append(HERE.parent / "models" / "t5_tokenizer" / "tokenizer.json")
    # 支持泛化用户缓存路径 ~/.cache/anima-tagger/models/...
    cand.append(Path.home() / ".cache" / "anima-tagger" / "models" / "t5_tokenizer" / "tokenizer.json")
    try:
        from tokenizers import Tokenizer  # type: ignore
        for p in cand:
            if p.exists():
                tok = Tokenizer.from_file(str(p))
                ids = tok.encode(text, add_special_tokens=False).ids
                tag = "exact(t5-sp)+unk" if T5_UNK_ID in ids else "exact(t5-sp)"
                return len(ids), tag
    except Exception:
        pass
    # 启发式：按"逗号分隔的词组数 + 单词数"粗估，宁可高估
    rough = len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text))
    return rough, "estimate(rough)"

# ---------------------------------------------------------------- 规范化

def normalize_tag(t: str) -> str:
    """下划线→空格、小写、折叠空白。括号保留（Anima 链路不做转义）。"""
    t = t.replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

# ---------------------------------------------------------------- 上位词折叠

_WORD_RE = re.compile(r"[a-z0-9()'\-]+")

def _words(t: str) -> set[str]:
    return set(_WORD_RE.findall(t))

# 修饰词白名单：只有当超串比它**多出来的词全部是修饰词**时才允许折叠。
#
# 判据：把"物体"折进"动作/别的物体"是语义错误（book←holding book、gun←gun sling），
# 黑名单永远列不全，所以用中心名词 + 非动词短语两道闸。
COLORS_MOD = {
    "black", "blonde", "brown", "blue", "white", "pink", "grey", "gray", "purple",
    "red", "green", "orange", "silver", "aqua", "platinum", "gold", "golden", "navy",
    "light", "dark", "pale", "bright", "deep", "pastel",
}
MODIFIERS = COLORS_MOD | {
    # 尺寸/程度
    "long", "short", "very", "absurdly", "medium", "large", "small", "huge", "gigantic",
    "flat", "big", "little", "oversized", "baggy", "tight", "wide", "narrow", "full",
    "half", "high", "low", "single", "double", "twin", "mini", "maxi", "micro",
    # 形制/材质/状态
    "pleated", "frilled", "striped", "plaid", "checkered", "polka", "dotted", "torn",
    "open", "closed", "wet", "sleeveless", "detached", "buttoned", "unbuttoned",
    "layered", "trimmed", "fur", "lace", "latex", "satin", "velvet", "sheer",
    "denim", "leather", "knit", "wool", "cotton", "silk",
    # 归属用名词（限定性用法）
    "hair", "neck", "arm", "leg", "ear", "eye", "head", "hand", "shoulder",
}


# 动作/关系动词：作为 keeper 的首词时，说明 keeper 是动词短语而非名词短语
ACTION_WORDS = {
    "holding", "wearing", "carrying", "using", "grabbing", "touching", "eating",
    "drinking", "reading", "riding", "hugging", "kissing", "licking", "sucking",
    "pulling", "pushing", "lifting", "throwing", "catching", "drawing", "aiming",
    "playing", "opening", "covering", "reaching", "pointing", "resting", "pouring",
    "adjusting", "twirling", "soaking", "sitting", "standing", "lying",
}


def _can_fold(victim: str, keeper: str) -> bool:
    """受害者能否被 keeper 折叠掉。

    判据：
      ① 词集包含：victim 的所有词都在 keeper 里
      ② **中心名词相同**：两者的最后一个词必须一样（hat ⊂ witch hat ✓；gun ⊂ gun sling ✗）
      ③ **keeper 不是动词短语**：keeper 的第一个词不能是动作动词
         （book ⊂ holding book ✗ —— 把"物体"折进"拿物体的动作"是语义错误）

    """
    vw, kw = _words(victim), _words(keeper)
    if not vw or not vw <= kw:
        return False
    vl, kl = victim.split(), keeper.split()
    if not vl or not kl:
        return False
    if vl[-1] != kl[-1]:            # ② 中心名词必须一致
        return False
    if kl[0] in ACTION_WORDS:       # ③ 不能是动词短语
        return False
    return True


def fold_redundant(tags: list[str], scores: dict[str, float] | None = None):
    """包含式冗余折叠（中心名词 + 非动词短语双闸）。

    多个候选都能覆盖受害者时，保留**置信度最高**的那个作为"因谁被删"。

    返回 (保留, [(被删, 因谁被删)], [(被闸拦下, 差点因谁被删)])
    """
    sc = scores or {}
    ordered = sorted(range(len(tags)), key=lambda i: (-len(tags[i]), i))
    kept: list[int] = []
    dropped: list[tuple[str, str]] = []
    blocked: list[tuple[str, str]] = []
    for i in ordered:
        t = tags[i]
        coverers: list[str] = []      # 能覆盖 t 且允许折叠
        blockers: list[str] = []      # 能覆盖 t 但不允许折叠
        for j in kept:
            k = tags[j]
            if t != k and t in k and _words(t) <= _words(k):
                (coverers if _can_fold(t, k) else blockers).append(k)
        if coverers:
            best = max(coverers, key=lambda k: sc.get(k, 0.0))
            dropped.append((t, best))
        else:
            if blockers:
                blocked.append((t, max(blockers, key=lambda k: sc.get(k, 0.0))))
            kept.append(i)
    kept.sort()
    return [tags[i] for i in kept], dropped, blocked

# ---------------------------------------------------------------- 槽位冲突（自建表）

def _suffix_slot(prefixes: list[str], suffix: str) -> set[str]:
    return {f"{p}_{suffix}" for p in prefixes} | {f"{p} {suffix}" for p in prefixes}

COLORS = ["black", "blonde", "brown", "blue", "white", "pink", "grey", "gray",
          "purple", "red", "green", "orange", "silver", "aqua", "blonde", "platinum"]

_C = [c for c in COLORS if c != "gray"]

# 角色级槽位：**多主体图整组跳过**——两个角色当然可以有不同的发色/瞳色。
# （实测踩坑：图 04 是两个角色，`black hair` + `white hair` 被误报成冲突。）
#
# ⚠️ **属性轴必须拆开**：颜色 / 图案 / 构造 / 长度 是不同的轴，跨轴的值可以并存。
# 合成一个槽位会把颜色、长度这类可复现信息静默抹掉。
CHARACTER_SLOTS: dict[str, set[str]] = {
    # —— 颜色轴
    "hair_color": _suffix_slot(_C, "hair"),
    "eye_color": _suffix_slot(_C, "eyes"),
    "hairband_color": _suffix_slot(_C, "hairband"),
    "neckerchief_color": _suffix_slot(_C, "neckerchief"),
    "cardigan_color": _suffix_slot(_C, "cardigan"),
    "sweater_color": _suffix_slot(_C, "sweater"),
    "dress_color": _suffix_slot(_C, "dress"),
    "skirt_color": _suffix_slot(_C, "skirt"),
    "shirt_color": _suffix_slot(_C, "shirt"),
    # —— 长度/尺寸轴
    "hair_length": {"long hair", "very long hair", "absurdly long hair", "short hair",
                    "medium hair", "very short hair", "bald"},
    "breast_size": {"flat chest", "small breasts", "medium breasts", "large breasts",
                    "huge breasts", "gigantic breasts"},
    "skirt_length": {"long skirt", "short skirt", "microskirt", "miniskirt", "maxi skirt"},
    "dress_length": {"short dress", "long dress", "minidress", "maxi dress"},
    # —— 图案轴
    "skirt_pattern": {"plaid skirt", "checkered skirt", "striped skirt", "polka dot skirt",
                      "argyle skirt", "houndstooth skirt", "floral skirt"},
    "dress_pattern": {"floral print dress", "plaid dress", "striped dress",
                      "polka dot dress"},
    # —— 构造/形制轴
    "skirt_form": {"pleated skirt", "frilled skirt", "denim skirt", "pencil skirt",
                   "tight skirt", "tiered skirt", "wrap skirt", "suspender skirt"},
    "dress_form": {"frilled dress", "sundress", "sailor dress", "pinafore dress",
                   "sweater dress", "shirt dress"},
    # —— 腿袜轴
    "legwear": {"pantyhose", "black pantyhose", "thighhighs", "black thighhighs",
                "white thighhighs", "socks", "kneehighs", "bare legs", "black legwear",
                "white socks", "black socks", "loose socks", "single sock"},
    # —— 姿势轴（多主体时跳过；单主体只能有一个主姿势）
    "pose": {"standing", "sitting", "squatting", "kneeling", "lying", "on back",
             "on stomach", "on side", "all fours", "crouching", "seiza", "wariza",
             "indian style", "bending over"},
}

# 画面级槽位：任何情况都要查（它们描述的是"这张图怎么拍的"，与人数无关）
IMAGE_SLOTS: dict[str, set[str]] = {
    "background_form": {"simple background", "blurry background", "white background",
                        "detailed background", "transparent background", "dark background"},
    "background_type": {"gradient background", "checkered background", "two-tone background",
                        "spotlight", "vignette"},
    "background_color": _suffix_slot(_C, "background"),
    "gaze": {"looking at viewer", "looking away", "looking down", "looking up",
             "looking back", "looking to the side", "looking at another"},
    "framing": {"full body", "upper body", "cowboy shot", "close-up", "portrait",
                "wide shot", "feet out of frame"},
    "orientation": {"from above", "from below", "from side", "from behind", "from front"},
}

# 分类词豁免表：这些 tag 的存在说明"同槽多值"是**真实**的（双色发、左右不对称），应当放行。
SLOT_SUPPRESSORS: dict[str, set[str]] = {
    "hair_color": {"colored inner hair", "streaked hair", "two-tone hair",
                   "multicolored hair", "gradient hair", "colored tips",
                   "multicolored hair", "hair streaks"},
    "eye_color": {"heterochromia", "multicolored eyes", "two-tone eyes"},
    "legwear": {"asymmetrical legwear", "uneven legwear", "mismatched legwear",
                "single thighhigh", "single sock", "single legwear", "asymmetrical legwear"},
    "background_form": {"gradient background", "two-tone background"},
    "pose": {"sitting", "squatting"},   # 坐姿与蹲姿可并存，不互斥
}

# 多主体标记
MULTI_MARKERS = {
    "2girls", "3girls", "4girls", "5girls", "6girls", "6+girls", "7+girls", "8+girls",
    "9+girls", "2boys", "3boys", "4boys", "5boys", "6+boys", "multiple girls",
    "multiple boys", "multiple others", "couple", "group", "duo", "trio", "threesome",
    "group sex", "twins", "siblings",
}


def detect_multi(tags: list[str]) -> tuple[bool, list[str]]:
    """自动判定是否多主体。**必须自动**——手工传 --multi 迟早会忘。"""
    present = set(tags)
    hit = sorted(present & MULTI_MARKERS)
    if "1girl" in present and "1boy" in present:
        hit.append("1girl+1boy")
    return bool(hit), hit


def find_conflicts(tags: list[str], is_multi: bool = False) -> list[dict]:
    present = set(tags)
    slots = dict(IMAGE_SLOTS)
    if not is_multi:
        slots.update(CHARACTER_SLOTS)
    out: list[dict] = []

    # ① 结构 tag 互斥：solo 与任何"多人"标记不可能同时成立（1girl 不算多人）
    multi_present = present & MULTI_MARKERS
    if "solo" in present and multi_present:
        out.append({"slot": "subject_count", "scope": "image",
                    "candidates": sorted({"solo"} | multi_present),
                    "needs": "solo 与多人标记不可能同时成立，必须删一边"})
    if "1girl" in present and multi_present and not is_multi:
        out.append({"slot": "subject_count_conflict", "scope": "image",
                    "candidates": sorted({"1girl"} | multi_present),
                    "needs": "1girl 与多人标记矛盾"})

    # ② 槽位多值（带分类词豁免）
    # 已知可叠的例外：`wide shot` 可以与 `full body` 同时成立（nai5 §3.9 明确列出）
    framing_ok = frozenset({"full body", "wide shot"})
    for name, members in slots.items():
        hit = sorted(present & members)
        if len(hit) <= 1:
            continue
        if name == "framing" and frozenset(hit) == framing_ok:
            continue
        sup = SLOT_SUPPRESSORS.get(name, set())
        if present & sup:
            continue
        out.append({
            "slot": name,
            "scope": "image" if name in IMAGE_SLOTS else "character",
            "candidates": hit,
            "needs": "看图或按意图定夺",
        })
    return out

# ---------------------------------------------------------------- 读文件

def read_text_sniff(path: Path | str) -> str:
    """读文本或标准输入，容忍不同终端写出的几种编码。

    若 path 为 "-" 或等价的标准输入指示，直接从 sys.stdin 读取；
    Windows PowerShell 5.1 的 `>` 写出 UTF-16LE，`Out-File -Encoding utf8` 带 BOM；
    pwsh 7 的 `>` 是 UTF-8 无 BOM。三种都吃，免得使用者先撞一次编码错误。
    """
    if str(path) == "-":
        if hasattr(sys.stdin, "buffer"):
            raw = sys.stdin.buffer.read()
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                return raw.decode("utf-16")
            return raw.decode("utf-8-sig")
        return sys.stdin.read()

    p = Path(path)
    raw = p.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


# ---------------------------------------------------------------- 主流程

def load_tagger_json(path: Path | str) -> tuple[list[str], dict[str, float]]:
    data = json.loads(read_text_sniff(path))
    recs = data if isinstance(data, list) else [data]
    tags: list[str] = []
    scores: dict[str, float] = {}
    for rec in recs:
        for key in ("character", "general"):
            for name, s in (rec.get(key) or {}).items():
                tags.append(name)
                scores[name] = max(scores.get(name, 0.0), float(s))
        rating = rec.get("rating") or {}
        if rating:
            best = max(rating, key=rating.get)
            tags.append(best)
            scores[best] = max(scores.get(best, 0.0), float(rating[best]))
    return tags, scores


def per_record(args) -> int:
    """对多图 JSON 逐张校验，输出一张汇总表（E2E 用）。"""
    data = json.loads(read_text_sniff(args.tagger_json))
    data = data if isinstance(data, list) else [data]
    rows = []
    rc = 0
    for rec in data:
        tags = list(rec.get("character", {})) + list(rec.get("general", {}))
        rating = rec.get("rating") or {}
        if rating:
            tags.append(max(rating, key=rating.get))
        norm = [normalize_tag(t) for t in tags if normalize_tag(t)]
        folded, drops, blocked = fold_redundant(norm, {normalize_tag(k): v for k, v in
                                                       {**rec.get("character", {}), **rec.get("general", {})}.items()})
        is_multi, markers = detect_multi(folded)
        conflicts = find_conflicts(folded, is_multi=is_multi)
        text = SEP.join(folded)
        n_tok, how = count_tokens(text)
        if conflicts or (n_tok or 0) > TOKEN_LIMIT:
            rc = 1
        rows.append({
            "image": Path(rec.get("image", "?")).name,
            "in": len(tags), "folded": len(folded), "dropped": len(drops),
            "tokens": n_tok, "token_method": how,
            "is_multi": is_multi, "markers": markers,
            "conflicts": [f"{c['slot']}: " + " | ".join(c["candidates"]) for c in conflicts],
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rc


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tagger-json", type=str, help="wd_tagger.py --json 的输出文件路径，支持 '-' 表示从标准输入 stdin 读取")
    src.add_argument("--tags", help="逗号分隔的 tag 串")
    ap.add_argument("--nl", default="", help="自然语言段（一起算 token）")
    ap.add_argument("--multi", action="store_true", help="强制多人模式（正常会自动探测）")
    ap.add_argument("--per-record", action="store_true",
                    help="输入是 wd_tagger 的多图 JSON 时，逐张图分别校验（E2E 用）")
    ap.add_argument("--proposed", type=Path,
                    help="审计一个 LLM 给出的 tag 列表（逗号分隔或每行一个）：查发明 / 查过度删除")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    if args.per_record and args.tagger_json:
        return per_record(args)
    if args.tagger_json:
        raw, raw_scores = load_tagger_json(args.tagger_json)
    else:
        raw = [t for t in (x.strip() for x in args.tags.split(",")) if t]
        raw_scores = {}

    problems: list[str] = []

    # 1) 规范化（分数跟着规范化后的名字走，折叠时按分数挑保留项）
    norm = [normalize_tag(t) for t in raw]
    norm = [t for t in norm if t]
    score_by_norm: dict[str, float] = {}
    for src, dst in zip(raw, [normalize_tag(t) for t in raw]):
        if dst:
            score_by_norm[dst] = max(score_by_norm.get(dst, 0.0), raw_scores.get(src, 0.0))
    changed = [(a, b) for a, b in zip(raw, norm) if a.strip().lower() != b]

    # 2) 去重（保序）
    seen, deduped = set(), []
    for t in norm:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    # 3) 上位词折叠
    folded, drops, blocked_folds = fold_redundant(deduped, score_by_norm)

    # 3b) 审计 LLM 给出的清单：输出必须 ⊆ 输入；差异也暴露"过度删除"
    audit: dict | None = None
    if args.proposed:
        txt = read_text_sniff(args.proposed)
        prop_raw = [t for t in re.split(r"[,\n]", txt) if t.strip()]
        prop = [normalize_tag(t) for t in prop_raw if normalize_tag(t)]
        prop_set, input_set = set(prop), set(deduped)
        invented = sorted(prop_set - input_set)
        dropped_vs_input = sorted(input_set - prop_set)
        audit = {
            "proposed_count": len(prop),
            "invented": invented,                     # 输入里没有 → 违反"不许发明"
            "dropped_vs_input": dropped_vs_input,     # 输入有、它删了 → 可能是过度删除，需人工裁
        }
        if invented:
            problems.append(f"审计：LLM 发明了 {len(invented)} 个输入里没有的 tag")

    # 4) 槽位冲突
    is_multi, multi_hits = detect_multi(folded)
    if args.multi:
        is_multi = True
    conflicts = find_conflicts(folded, is_multi=is_multi)

    # 5) 组装 + token
    #    tag 层单独也数一遍：它是固定开支，剩下的全是 NL 预算，这个数直接决定 NL 还能写多少。
    tag_text = SEP.join(folded)
    text = f"{tag_text}. {args.nl}" if (tag_text and args.nl) else (tag_text or args.nl)
    n_tok, how = count_tokens(text)
    tag_tok: int | None = None
    if tag_text and args.nl:
        tag_tok, _ = count_tokens(tag_text)
    if n_tok is not None and n_tok > TOKEN_LIMIT:
        msg = f"token 超预算：{n_tok} > {TOKEN_LIMIT}（超出 {n_tok - TOKEN_LIMIT}，{how}）"
        if tag_tok is not None:
            # 实测（0.35 阈值、24 张图）：最复杂的图 tag 层也只到 291，离 512 有 221 余量。
            # 所以要撑满 512 得把 tag 数翻近一倍——正常情况超预算一定是 NL 太长。
            msg += f"　→　精简 NL（tag 层占 {tag_tok}，NL 预算 {TOKEN_LIMIT - tag_tok}）"
            if tag_tok > TOKEN_LIMIT:
                msg += ("\n      ⚠ tag 层单独就超了 512，这不正常——"
                        "先查 tagger 阈值/模型是不是被换过，别靠删 tag 硬凑")
        problems.append(msg)
    if "+unk" in how:
        problems.append(
            "文本里有 T5 词表表示不了的字符（中文、日文等）——这一段在 T5 通道会被丢成 <unk>，"
            "token 数不代表它的真实占用，那段内容也进不了模型。Anima 的提示词应当是英文"
        )
    if conflicts:
        problems.append(f"{len(conflicts)} 个槽位有多值，需人工/看图定夺")

    result = {
        "input_count": len(raw),
        "normalized_changed": changed,
        "deduped_removed": len(norm) - len(deduped),
        "folded": [{"dropped": d, "kept_because": k} for d, k in drops],
        "fold_blocked_by_action_word": [{"kept": k, "would_have_folded": v} for v, k in blocked_folds],
        "is_multi": is_multi,
        "multi_markers": multi_hits,
        "conflicts": conflicts,
        "final_tags": folded,
        "final_text": text,
        "tokens": {"count": n_tok, "method": how, "limit": TOKEN_LIMIT,
                   "over_by": max(0, (n_tok or 0) - TOKEN_LIMIT),
                   "tag_layer_only": tag_tok},
        "audit": audit,
        "problems": problems,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"输入 {len(raw)} → 规范化后 {len(norm)} → 去重 {len(deduped)} → 折叠后 {len(folded)}")
        if changed:
            print("\n[规范化改动] " + ", ".join(f"{a!r}→{b!r}" for a, b in changed[:20]))
        if drops:
            print("\n[上位词折叠]")
            for d, k in drops:
                print(f"  - {d}  （被 {k} 覆盖）")
        if blocked_folds:
            print("\n[闸拦下：动作词超串，不折]")
            for v, k in blocked_folds:
                print(f"  - {v}  虽然 {k} 词集覆盖它，但多出来的是动作词")
        if conflicts:
            print("\n[槽位冲突 · 需人工定夺]")
            for c in conflicts:
                print(f"  * {c['slot']}: {' | '.join(c['candidates'])}")
        tok_line = f"\n[token] {n_tok} （{how}） / 上限 {TOKEN_LIMIT}"
        if tag_tok is not None:
            tok_line += f"　·　tag 层 {tag_tok}　→　NL 最多 {TOKEN_LIMIT - tag_tok}"
        print(tok_line)
        print("\n[最终 tag] " + SEP.join(folded))
        if args.nl:
            print("\n[最终文本]\n" + text)
        if audit:
            print(f"\n[审计 LLM 输出] 提出 {audit['proposed_count']} 条")
            if audit["invented"]:
                print("  ! 发明（输入里没有）：" + ", ".join(audit["invented"]))
            if audit["dropped_vs_input"]:
                print(f"  它删掉了 {len(audit['dropped_vs_input'])} 条输入里有的 tag（需人工裁）")
        if problems:
            print("\n[待处理]")
            for p in problems:
                print("  ! " + p)

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
