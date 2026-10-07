"""PixAI: original CUDA inference, verified ONNX CPU fallback when CUDA is absent."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

try:
    from .pixai_config import MODEL_DIR, MODEL_REPO, MODEL_REVISION, THRESHOLDS, CATEGORY_COUNTS, verify_assets, CPU_MODEL_DIR, CPU_MODEL_REPO, CPU_MODEL_REVISION
except ImportError:
    from pixai_config import MODEL_DIR, MODEL_REPO, MODEL_REVISION, THRESHOLDS, CATEGORY_COUNTS, verify_assets, CPU_MODEL_DIR, CPU_MODEL_REPO, CPU_MODEL_REVISION


def cuda_available():
    # CPU-only users need neither PyTorch nor its GPU dependencies.
    try:
        import torch
        return torch.cuda.is_available()
    except (ImportError, OSError):
        return False


def resolve_device(requested, available=None):
    if requested != "auto":
        return requested
    return "cuda:0" if (cuda_available() if available is None else available) else "cpu"


def cpu_preprocess(path):
    """Use the PIL bilinear preprocessing validated in the ONNX comparison."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        original_size = list(image.size)
        if image.mode != "RGB":
            image = image.convert("RGBA")
            canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
            canvas.alpha_composite(image)
            image = canvas.convert("RGB")
        width, height = image.size
        if image.size != (1008, 1008):
            scale = min(1008 / width, 1008 / height)
            resized = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(resized, Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (1008, 1008), (0, 0, 0))
            canvas.paste(image, ((1008 - resized[0]) // 2, (1008 - resized[1]) // 2))
            image = canvas
        array = (np.asarray(image, dtype=np.float32) / 255.0 - .5) / .5
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None]), original_size


class PixAICPUTagger:
    def __init__(self, model_dir=CPU_MODEL_DIR, threads=8):
        problems = verify_assets(model_dir, cpu=True)
        if problems:
            raise RuntimeError("\n".join(problems) + "\nRun python tools/setup.py --device cpu")
        import onnxruntime as ort
        categories = json.loads((Path(model_dir) / "tags.json").read_text(encoding="utf-8"))["categories"]
        self.tags, self.splits = [], []
        for category in categories:
            if category["offset"] != len(self.tags) or len(category["tags"]) != category["count"]:
                raise ValueError("Invalid ONNX vocabulary offsets or counts")
            self.tags.extend(category["tags"])
            self.splits.append((category["name"], category["count"]))
        select_tags([0.] * len(self.tags), self.tags, self.splits, THRESHOLDS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(model_dir) / "model.onnx"),
                                            sess_options=options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 1008, 1008] or inp.type != "tensor(float)":
            raise ValueError(f"Unexpected ONNX input: {inp.shape} {inp.type}")
        self.input_name, self.threads = inp.name, threads

    def tag(self, path, thresholds, top=0):
        import numpy as np
        started = time.perf_counter()
        pixels, original_size = cpu_preprocess(path)
        inference_started = time.perf_counter()
        logits = np.asarray(self.session.run(None, {self.input_name: pixels})[0], dtype=np.float32).reshape(-1)
        inference_seconds = time.perf_counter() - inference_started
        positive = logits >= 0
        scores = np.empty_like(logits)
        scores[positive] = 1 / (1 + np.exp(-logits[positive]))
        exp = np.exp(logits[~positive])
        scores[~positive] = exp / (1 + exp)
        categories = select_tags(scores, self.tags, self.splits, thresholds, top)
        return {"image": str(path), "model": CPU_MODEL_REPO, "revision": CPU_MODEL_REVISION,
                "device": "cpu", "backend": "onnxruntime", "provider": "CPUExecutionProvider",
                "precision": "fp32", "size": 1008, "original_size": original_size,
                "activation": "sigmoid", "thresholds": thresholds, **categories,
                "performance": {"inference_seconds": inference_seconds,
                                "total_seconds": time.perf_counter() - started, "threads": self.threads}}


def select_tags(scores, tags, splits, thresholds, top=0):
    if len(scores) != len(tags) or sum(count for _, count in splits) != len(tags):
        raise ValueError("Model output, vocabulary and category dimensions do not match")
    if dict(splits) != CATEGORY_COUNTS or len(splits) != len(CATEGORY_COUNTS):
        raise ValueError("Unexpected PixAI category layout")
    result, offset = {}, 0
    for category, count in splits:
        pairs = [(tags[i], float(scores[i])) for i in range(offset, offset + count)]
        if any(not math.isfinite(score) or not 0 <= score <= 1 for _, score in pairs):
            raise ValueError("Invalid model probabilities")
        pairs = sorted(((name, score) for name, score in pairs if score >= thresholds[category]),
                       key=lambda pair: pair[1], reverse=True)
        if category == "general" and top:
            pairs = pairs[:top]
        result[category] = dict(pairs)
        offset += count
    return result


def require_cuda(torch, precision, device):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the explicitly requested GPU. Use --device auto or --device cpu for ONNX CPU inference.")
    index = int(device.partition(":")[2] or 0)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device does not exist: {device}")
    torch.cuda.set_device(index)
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16. Retry with --precision fp32.")
    return torch.bfloat16 if precision == "bf16" else torch.float32


class PixAITagger:
    def __init__(self, model_dir=MODEL_DIR, precision="bf16", device="cuda:0"):
        problems = verify_assets(model_dir)
        if problems:
            raise RuntimeError("\n".join(problems))
        import torch
        self.torch = torch
        self.dtype = require_cuda(torch, precision, device)
        self.device, self.precision = device, precision
        spec = importlib.util.spec_from_file_location("anima_pixai_upstream", Path(model_dir) / "tagger_pipeline.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.model = module.ViTDetCls.from_pretrained(
            str(model_dir), dtype=self.dtype, local_files_only=True).eval().to(device)
        self.processor = module.RescalePadProcessor(size=1008)
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        self.tags, self.splits = config["tags"], config["tags_split"]
        select_tags([0.] * len(self.tags), self.tags, self.splits, THRESHOLDS)

    def tag(self, path, thresholds, top=0):
        from PIL import Image, ImageOps
        torch = self.torch
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            original_size = list(image.size)
            pixels = self.processor(image, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        torch.cuda.synchronize(self.device)
        gpu_started = time.perf_counter()
        with torch.inference_mode():
            scores = self.model(pixels).float().sigmoid()[0].cpu().tolist()
        torch.cuda.synchronize(self.device)
        gpu_seconds = time.perf_counter() - gpu_started
        categories = select_tags(scores, self.tags, self.splits, thresholds, top)
        return {"image": str(path), "model": MODEL_REPO, "revision": MODEL_REVISION,
                "device": self.device, "backend": "pytorch", "precision": self.precision, "size": 1008,
                "original_size": original_size, "activation": "sigmoid",
                "thresholds": thresholds, **categories,
                "performance": {"gpu_seconds": gpu_seconds,
                                "total_seconds": time.perf_counter() - started,
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
                                "reserved_mib": torch.cuda.memory_reserved(self.device) / 2**20}}


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return value


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("ANIMA_MODEL_DIR", MODEL_DIR)))
    parser.add_argument("--cpu-model-dir", type=Path, default=CPU_MODEL_DIR)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"] + [f"cuda:{i}" for i in range(16)], default="auto")
    parser.add_argument("--cpu-threads", type=int, default=8, help="ONNX CPU threads (default: verified 8-thread configuration)")
    for name, threshold in THRESHOLDS.items():
        parser.add_argument(f"--{name}", type=probability, default=threshold)
    parser.add_argument("--top", type=int, default=0, help="Limit general tags only (0: unlimited); never limits characters")
    parser.add_argument("--json", action="store_true", help="Emit validator-compatible scored records")
    args = parser.parse_args(argv)
    if args.top < 0:
        parser.error("--top must be nonnegative")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
    thresholds = {name: getattr(args, name) for name in THRESHOLDS}
    try:
        started = time.perf_counter()
        device = resolve_device(args.device)
        if device == "cpu":
            print("PixAI: using verified ONNX CPU FP32 inference (CUDA unavailable or CPU requested).", file=sys.stderr)
            tagger = PixAICPUTagger(args.cpu_model_dir, args.cpu_threads)
        else:
            tagger = PixAITagger(args.model_dir, args.precision, device)
        load_seconds = time.perf_counter() - started
        records = [tagger.tag(path, thresholds, args.top) for path in args.images]
        for record in records:
            record["performance"]["model_load_seconds"] = load_seconds
        if args.json:
            print(json.dumps(records, ensure_ascii=False, indent=2))
        else:
            for record in records:
                print(record["image"])
                print(", ".join([*record["character"], *record["general"]]))
        return 0
    except Exception as exc:
        print(f"PixAI inference failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
