"""Pinned upstream assets and inference defaults (no heavy imports)."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "pixai-tagger-v1.0-original"
MODEL_REPO = "pixai-labs/pixai-tagger-v1.0"
MODEL_REVISION = "9fe10addf9326e292da8a85a98ea74cd91b41771"
CPU_MODEL_DIR = ROOT / "models" / "pixai-tagger-v1.0-onnx"
CPU_MODEL_REPO = "noaione/pixai-tagger-v1.0-onnx"
CPU_MODEL_REVISION = "68e8f4f02dd56a5f40c1b7474489fa0f599dec34"
CPU_MODEL_HASHES = {
    "model.onnx": "563f4576c2668560c20f403b957f0ec4a7bd6a2275da2c2aa7f82f898ad34e5c",
    "model.onnx.data": "4de1c25a38d1f2a2172fbcb0d2485b5f02a058b6e67bedd7bbbcfdf4de9329dc",
    "tags.json": "0d34f2078016798808dc066dc206b18fb6ce7622f64241002ecd24172a4da068",
    "README.md": "4bf01095d3d9f25322b7ab1927ab9da92386d801a5dfeec6a4b9d60e63619861",
}
MODEL_HASHES = {
    "model.safetensors": "f29e475205cbcbc25b52a075840c7809d6215be15f45375113ba09c49bf90292",
    "config.json": "f8a19b38661c37dc5fd519f2137be70f6f897972021b4ebc17b7d35a0e27b9dd",
    "preprocessor_config.json": "20451575e627950993de7b00641d5b263eceb3e7436e4ad518fea2411b60e49d",
    "tagger_pipeline.py": "3431279fdfff234f62664999fc1fe6bb20f1e8756f7435d0e165ebd4456f7fcd",
    "README.md": "67b6e11967974b2646275da42f3a1149ef77ea2eed5c6d93cd6a258d495c5a38",
}
THRESHOLDS = {"general": .17, "character": .27, "copyright": .24,
              "style": .15, "meta": .17, "rating": .41}
CATEGORY_COUNTS = {"general": 15043, "character": 8308, "copyright": 2460,
                   "style": 4917, "meta": 145, "rating": 4}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory, *, weights=False, cpu=False):
    """Check all assets; hash executable code/config each time, large weights on demand."""
    problems = []
    hashes = CPU_MODEL_HASHES if cpu else MODEL_HASHES
    weight_name = "model.onnx.data" if cpu else "model.safetensors"
    weight_size = 1955123200 if cpu else 1945425796
    for name, expected in hashes.items():
        path = Path(directory) / name
        if not path.is_file():
            problems.append(f"Missing model asset: {path}")
        elif name == weight_name and not weights:
            if path.stat().st_size != weight_size:
                problems.append(f"Incomplete model weights: {path}")
        elif sha256_file(path) != expected:
            problems.append(f"SHA256 mismatch: {path}; rerun tools/setup.py")
    return problems
