#!/usr/bin/env python
"""WD EVA02 Tagger 2026 Canary —— ONNX 反推（danbooru 风格 tag）。

模型: ashen-sensored/wd-eva02-tagger-2026-canary （16,473 标签，含 rating/general/character）
ONNX: Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2 （与上游逐字节相同的 selected_tags.csv）

用法:
    python wd_tagger.py IMAGE [IMAGE...] [--model-dir DIR] [--general 0.35] [--character 0.85]
                               [--top N] [--json] [--csv PATH]

设计要点（照抄 canonical 实现，不自行发明）:
  * 预处理: EXIF 转正 -> 非 RGB 先转(RGBA 用白底合成) -> 白边补方 -> resize 448(bicubic)
            -> /127.5-1 (mean=std=0.5) -> RGB 通道翻转成 BGR
  * 输入布局自适应: 从 onnxruntime 的 input shape 判断 NHWC / NCHW
  * 输出: sigmoid（若模型已带 sigmoid 则原样使用）, 按 category 9/0/4 分流
  * 阈值: general 默认 0.35 / character 默认 0.85；作者验证口径 P=R 点为 0.6094
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

def get_default_model_dir() -> Path:
    if env := os.environ.get("ANIMA_MODEL_DIR"):
        return Path(env)
    local_path = Path(__file__).resolve().parent.parent / "models" / "wd-eva02-tagger-2026-canary-onnx-v2"
    if (local_path / "model.onnx").exists():
        return local_path
    cache_path = Path.home() / ".cache" / "anima-tagger" / "models" / "wd-eva02-tagger-2026-canary-onnx-v2"
    if (cache_path / "model.onnx").exists():
        return cache_path
    return local_path

DEFAULT_MODEL_DIR = get_default_model_dir()
RATING, GENERAL, CHARACTER = 9, 0, 4


def load_tags(csv_path: Path) -> tuple[list[str], list[int]]:
    names: list[str] = []
    cats: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            cats.append(int(row["category"]))
    return names, cats


def preprocess(path: Path, size: int) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) or img
    if img.mode != "RGB":
        if img.mode == "RGBA" or "transparency" in img.info:
            img = img.convert("RGBA")
            canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img)
            img = canvas.convert("RGB")
        else:
            img = img.convert("RGB")
    w, h = img.size
    px = max(w, h)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(img, ((px - w) // 2, (px - h) // 2))
    return canvas.resize((size, size), Image.Resampling.BICUBIC)


def to_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # mean=std=0.5
    arr = arr[..., ::-1]                                   # RGB -> BGR
    return np.ascontiguousarray(arr)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--csv", default=None, help="selected_tags.csv 路径（默认取 model-dir 下的）")
    ap.add_argument("--general", type=float, default=0.35)
    ap.add_argument("--character", type=float, default=0.85)
    ap.add_argument("--top", type=int, default=0, help="只输出前 N 个 general tag（0=全部）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort

    model_dir = Path(args.model_dir)
    onnx_path = model_dir / "model.onnx"
    csv_path = Path(args.csv) if args.csv else model_dir / "selected_tags.csv"
    for p in (onnx_path, csv_path):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 1

    names, cats = load_tags(csv_path)
    cat_arr = np.asarray(cats)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    # 输入布局自适应：
    #   NHWC [N,H,W,3] —— SmilingWolf 官方 ONNX 导出的惯例
    #   NCHW [N,3,H,W] —— 本仓库这份 ONNX（Misaka41Z 导出）实测是这种
    nhwc = len(shape) == 4 and isinstance(shape[-1], int) and shape[-1] in (1, 3)
    size = 448
    if len(shape) == 4:
        cand = shape[1] if nhwc else shape[2]
        if isinstance(cand, int) and cand > 0:
            size = cand

    results = []
    for image in args.images:
        img = preprocess(Path(image), size)
        arr = to_array(img)
        x = arr[None, ...] if nhwc else np.transpose(arr, (2, 0, 1))[None, ...]
        y = sess.run(None, {inp.name: np.ascontiguousarray(x)})[0]
        scores = np.asarray(y, dtype=np.float32).reshape(-1)
        if scores.size != len(names):
            print(f"!! 输出维度 {scores.size} != 标签数 {len(names)}", file=sys.stderr)
            return 2
        already_sigmoid = bool(scores.min() >= 0.0 and scores.max() <= 1.0)
        probs = scores if already_sigmoid else 1.0 / (1.0 + np.exp(-scores))

        rating = {names[i]: float(probs[i]) for i in np.where(cat_arr == RATING)[0]}
        gen = {names[i]: float(probs[i]) for i in np.where(cat_arr == GENERAL)[0] if probs[i] >= args.general}
        char = {names[i]: float(probs[i]) for i in np.where(cat_arr == CHARACTER)[0] if probs[i] >= args.character}
        gen = dict(sorted(gen.items(), key=lambda kv: -kv[1]))
        char = dict(sorted(char.items(), key=lambda kv: -kv[1]))
        if args.top:
            gen = dict(list(gen.items())[: args.top])

        results.append({
            "image": str(image),
            "input_layout": "NHWC" if nhwc else "NCHW",
            "size": size,
            "activation": "model-native" if already_sigmoid else "sigmoid-applied",
            "rating": dict(sorted(rating.items(), key=lambda kv: -kv[1])),
            "character": char,
            "general": gen,
        })

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for r in results:
        print("=" * 70)
        print(f"image: {r['image']}   ({r['input_layout']} {r['size']}px, {r['activation']})")
        print("-" * 70)
        print("rating   : " + ", ".join(f"{k}={v:.3f}" for k, v in r["rating"].items()))
        print("-" * 70)
        print(f"character(>={args.character}): " + ", ".join(f"{k}({v:.3f})" for k, v in r["character"].items()))
        print("-" * 70)
        print(f"general(>={args.general}, {len(r['general'])}): " + ", ".join(f"{k}({v:.3f})" for k, v in r["general"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
