#!/usr/bin/env python
"""anima-tagger 一键部署：建 venv → 装依赖 → 按设备安装 PixAI 模型与依赖 → 校验。

用法:
    python tools/setup.py              全自动（推荐）
    python tools/setup.py --check      只体检，不装不下载；一切就绪返回 0
    python tools/setup.py --no-venv    不建 venv，装进当前解释器
    python tools/setup.py --skip-deps  跳过装依赖（依赖已装好时重复跑更快）

国内网络先设镜像再跑:
    PowerShell :  $env:HF_ENDPOINT = "https://hf-mirror.com"
    bash/zsh   :  export HF_ENDPOINT=https://hf-mirror.com

"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 常量

try:
    from .pixai_config import ROOT, MODEL_DIR, MODEL_REPO, MODEL_REVISION, MODEL_HASHES, CATEGORY_COUNTS, verify_assets, CPU_MODEL_DIR, CPU_MODEL_REPO, CPU_MODEL_REVISION, CPU_MODEL_HASHES
except ImportError:
    from pixai_config import ROOT, MODEL_DIR, MODEL_REPO, MODEL_REVISION, MODEL_HASHES, CATEGORY_COUNTS, verify_assets, CPU_MODEL_DIR, CPU_MODEL_REPO, CPU_MODEL_REVISION, CPU_MODEL_HASHES

VENV = ROOT / ".venv-pixai-gpu"
TOKENIZER = ROOT / "models" / "t5_tokenizer" / "tokenizer.json"
REQUIREMENTS = ROOT / "tools" / "requirements.txt"
MODEL_FILES = tuple(MODEL_HASHES)
MIN_TOKENIZER_BYTES = 1_000_000
INSTALL_EXTRA = "huggingface_hub==0.36.2"
REQUIRED_MODULES = ("torch", "torchvision", "transformers", "timm", "onnxruntime", "PIL", "numpy", "tokenizers")
BACKEND = "cuda"
GPU_MODULES = REQUIRED_MODULES
GPU_ASSETS = (MODEL_DIR, MODEL_REPO, MODEL_REVISION, MODEL_HASHES)
CPU_MODULES = ("onnxruntime", "PIL", "numpy", "tokenizers")


def select_backend(py, requested="auto", *, installing=False):
    if requested != "auto":
        return requested
    if py.exists():
        probe = subprocess.run([str(py), "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"],
                               capture_output=True, timeout=60)
        if probe.returncode == 0:
            return "cuda"
    if installing:
        try:
            probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, timeout=15)
            if probe.returncode == 0 and probe.stdout.strip():
                return "cuda"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "cpu"


def configure_backend(backend):
    global BACKEND, MODEL_DIR, MODEL_REPO, MODEL_REVISION, MODEL_FILES, REQUIRED_MODULES, REQUIREMENTS
    BACKEND = backend
    if backend == "cpu":
        MODEL_DIR, MODEL_REPO, MODEL_REVISION = CPU_MODEL_DIR, CPU_MODEL_REPO, CPU_MODEL_REVISION
        MODEL_FILES = tuple(CPU_MODEL_HASHES)
        REQUIRED_MODULES = CPU_MODULES
        REQUIREMENTS = ROOT / "tools" / "requirements-cpu.txt"
    else:
        MODEL_DIR, MODEL_REPO, MODEL_REVISION, hashes = GPU_ASSETS
        MODEL_FILES = tuple(hashes)
        REQUIRED_MODULES = GPU_MODULES
        REQUIREMENTS = ROOT / "tools" / "requirements.txt"


def asset_problems(weights=False):
    return verify_assets(MODEL_DIR, weights=weights, cpu=BACKEND == "cpu")

RULE = "=" * 68


def _fix_console() -> None:
    """统一按 UTF-8 输出。

    Windows 上 Python 默认用 cp936 等本地编码写 stdout，被管道 / 编辑器按
    UTF-8 读时中文会变乱码；被 agent 读取时尤其明显。errors="replace" 兜底，
    保证任何情况下都不会抛 UnicodeEncodeError 打断部署。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass


# ---------------------------------------------------------------- 工具

def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GiB" if n >= 1 << 30 else f"{n / 1024 ** 2:.1f} MiB"


def verify_hashes() -> list[str]:
    return asset_problems(weights=True)


# ---------------------------------------------------------------- 体检

def broken_modules(py: Path) -> list[str]:
    """在目标解释器里真的 import 一遍运行时依赖，返回失败的项。

    不能用"文件在不在"代替：包装上了但一 import 就崩（缺 DLL、版本错配）的情况，
    只有真的 import 才看得出来——而那正好是打标时会炸的形态。
    """
    code = (
        "import importlib\n"
        f"mods = {list(REQUIRED_MODULES)!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m} ({type(e).__name__})')\n"
        "print('; '.join(bad))\n"
    )
    try:
        out = subprocess.run(
            [str(py), "-c", code], capture_output=True, text=True, timeout=180
        )
    except Exception as exc:  # 超时 / 解释器起不来
        return [f"<无法在 {py.name} 里检查依赖：{type(exc).__name__}>"]
    if out.returncode != 0:
        lines = (out.stderr or "").strip().splitlines()
        return [f"<检查依赖时解释器异常：{lines[-1] if lines else f'退出码 {out.returncode}'}>"]
    return [item for item in out.stdout.strip().split(";") if item]


def collect_problems(py: Path, require_venv: bool) -> list[str]:
    problems: list[str] = []

    if require_venv and not venv_python(VENV).exists():
        problems.append(f"没有虚拟环境：{venv_python(VENV)}（跑 `python tools/setup.py` 即可建）")
    elif py.exists():
        # 装了但 import 不进来，比"没装"更常见也更隐蔽，必须一起查
        broken = broken_modules(py)
        if broken:
            problems.append(
                "运行时依赖无法导入：" + "；".join(broken)
                + "——跑 `python tools/setup.py` 重装（**别加 --skip-deps**）"
            )

    problems.extend(asset_problems())
    if BACKEND == "cuda" and py.exists():
        probe = subprocess.run([str(py), "-c", "import torch; "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'); "
            "raise SystemExit(0 if torch.cuda.is_available() else 1)"],
            capture_output=True, text=True, timeout=60)
        print(probe.stdout.strip())
        if probe.returncode:
            problems.append("Requested CUDA is unavailable; use --device cpu or --device auto")

    if not TOKENIZER.exists():
        problems.append(
            f"缺 tokenizer：{TOKENIZER.relative_to(ROOT)}（本该随 clone 一起下来）——"
            f"缺它就只能粗估 token，512 预算会失准"
        )
    elif TOKENIZER.stat().st_size < MIN_TOKENIZER_BYTES:
        problems.append(f"{TOKENIZER.relative_to(ROOT)} 大小可疑，请重新 clone 仓库")

    return problems


def print_report(py: Path, problems: list[str], *, show_next_step: bool) -> int:
    print(RULE)
    print("anima-tagger 部署状态")
    print(RULE)
    print(f"skill 根目录 : {ROOT}")
    print(f"python       : {py}")
    print(f"模型目录     : {MODEL_DIR}")
    print(f"运行路线     : {BACKEND}")
    print(f"HF 端点      : {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(RULE)

    if problems:
        print(f"未就绪（{len(problems)} 项）：")
        for p in problems:
            print(f"  ! {p}")
        print(RULE)
        print("下一步：python tools/setup.py")
        return 1

    print("就绪。")
    print(RULE)
    if show_next_step:
        print("打标 + 校验：")
        print(f'  & "{py}" "{ROOT / "tools" / "pixai_tagger.py"}" <图片> `')
        print(f"      --device {BACKEND} --general 0.17 --character 0.27 --json > .work/run.json")
        print(f'  & "{py}" "{ROOT / "tools" / "anima_validate.py"}" --tagger-json .work/run.json')
    return 0


# ---------------------------------------------------------------- 步骤

def ensure_venv() -> Path:
    py = venv_python(VENV)
    if py.exists():
        print(f"[1/4] venv 已存在：{py}")
        return py
    print(f"[1/4] 建 venv：{VENV}")
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    if not py.exists():
        raise SystemExit(f"建 venv 失败：没有生成 {py}")
    return py


def install_deps(py: Path) -> None:
    print(f"[2/4] 装依赖：{REQUIREMENTS.name} + {INSTALL_EXTRA}")
    sys.stdout.flush()  # 否则 pip 的输出会插到上面这行之前
    if not REQUIREMENTS.exists():
        raise SystemExit(f"找不到 {REQUIREMENTS}")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip", "-q"], check=False)
    if BACKEND == "cuda":
        subprocess.run([str(py), "-m", "pip", "install", "torch==2.8.0+cu128", "torchvision==0.23.0+cu128",
                        "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS), INSTALL_EXTRA], check=True)



# ------------------------------------------------- 下载前的代理环境体检

_BRACKET_IPV6 = re.compile(r"^\[[0-9a-fA-F:.]+\]$")


def _strip_bracketed_ipv6() -> list[str]:
    """把 no_proxy 里的 `[::1]` 这类方括号 IPv6 字面量去括号，返回被改动的条目。

    Clash 一类代理工具会写成 `NO_PROXY=localhost,127.0.0.1,::1,[::1]`。httpx 把
    no_proxy 的每一项转成 URLPattern，方括号那项被它拼成 `all://*[::1]`，解析时抛
    `InvalidURL: Invalid port: ':1]'`——**在发出第一个请求之前**就炸，报错还完全
    看不出跟代理有关。去掉括号后与已有的 `::1` 等价（顺带去重）。
    """
    changed: list[str] = []
    for name in ("NO_PROXY", "no_proxy"):
        raw = os.environ.get(name)
        if not raw:
            continue
        kept: list[str] = []
        for item in (part.strip() for part in raw.split(",")):
            if item and _BRACKET_IPV6.match(item):
                changed.append(item)
                item = item[1:-1]
            if item and item not in kept:
                kept.append(item)
        os.environ[name] = ",".join(kept)
    return changed


def _httpx_error() -> str | None:
    """试着建一个 httpx.Client：成功返回 None，失败返回错误描述。"""
    try:
        import httpx
    except ImportError:
        return None  # 没装 httpx 就不会有这个问题
    try:
        httpx.Client().close()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def ensure_download_env() -> None:
    """确认 httpx 能建客户端；不行就先抢救，再不行把话说清楚。

    huggingface_hub 走 httpx，而 httpx 在**构造客户端时**就解析 NO_PROXY。
    配置里只要有一个它解析不了的条目，后面所有下载都会失败。这里先探一次。
    """
    problem = _httpx_error()
    if problem is None:
        return

    fixed = _strip_bracketed_ipv6()
    if fixed and _httpx_error() is None:
        print(f"[3/4] 已临时去掉 NO_PROXY 里的方括号条目 {fixed}（httpx 解析不了它们）")
        return

    print(
        f"\n!! httpx 无法创建 HTTP 客户端：{problem}\n"
        "   这会让所有下载失败，通常与代理配置有关。请检查 NO_PROXY / no_proxy：\n"
        "   - 去掉形如 [::1] 的方括号写法（写成 ::1 即可）\n"
        "   - 或者临时清空后再跑：$env:NO_PROXY = \"\"; $env:no_proxy = \"\"\n"
        f"   当前 NO_PROXY = {os.environ.get('NO_PROXY')!r}",
        file=sys.stderr,
    )


def download_assets(directory, repo, revision, hashes, *, cpu=False) -> None:
    if not verify_assets(directory, weights=True, cpu=cpu):
        print(f"[3/4] Verified assets already available: {repo}")
        return
    ensure_download_env()
    from huggingface_hub import snapshot_download
    print(f"[3/4] Downloading pinned {repo}@{revision[:12]} (about 1.9 GiB)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(directory), allow_patterns=list(hashes))
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}. Retry setup, or set HF_ENDPOINT to a reachable mirror.") from exc
    problems = verify_assets(directory, weights=True, cpu=cpu)
    if problems:
        raise RuntimeError("\n".join(problems))


def fetch_cpu_model(directory=CPU_MODEL_DIR) -> None:
    download_assets(directory, CPU_MODEL_REPO, CPU_MODEL_REVISION, CPU_MODEL_HASHES, cpu=True)


def fetch_model() -> None:
    hashes = CPU_MODEL_HASHES if BACKEND == "cpu" else GPU_ASSETS[3]
    try:
        download_assets(MODEL_DIR, MODEL_REPO, MODEL_REVISION, hashes, cpu=BACKEND == "cpu")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


# ---------------------------------------------------------------- 主流程

def main() -> int:
    _fix_console()
    ap = argparse.ArgumentParser(
        description="anima-tagger 一键部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="CUDA first; no CUDA uses validated ONNX CPU")
    ap.add_argument("--check", action="store_true", help="只体检，不装不下载")
    ap.add_argument("--no-venv", action="store_true", help="不建 venv，装进当前解释器")
    ap.add_argument("--skip-deps", action="store_true", help="跳过装依赖")
    ap.add_argument("--verify-hash", action="store_true",
                    help="额外逐字节核对 SHA256（要读 1.9 GiB，慢）")
    ap.add_argument("--_in-venv", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    target_py = Path(sys.executable) if args.no_venv else venv_python(VENV)
    backend = select_backend(target_py, args.device, installing=not args.check and not args._in_venv)
    configure_backend(backend)
    print(f"Selected inference route: {backend}")
    require_venv = not args.no_venv

    def full_report(py: Path, *, show_next_step: bool) -> int:
        problems = collect_problems(py, require_venv)
        if args.verify_hash and not problems:
            problems += verify_hashes()
        return print_report(py, problems, show_next_step=show_next_step)

    if args._in_venv:
        # 子进程：此时解释器就是 venv，huggingface_hub 已可用
        fetch_model()
        print("[4/4] 校验")
        return full_report(Path(sys.executable), show_next_step=True)

    if args.check:
        py = Path(sys.executable) if args.no_venv else venv_python(VENV)
        return full_report(py, show_next_step=True)

    py = Path(sys.executable) if args.no_venv else ensure_venv()

    if args.skip_deps:
        print("[2/4] 跳过装依赖")
    else:
        install_deps(py)

    if py.resolve() == Path(sys.executable).resolve():
        # 当前解释器就是要用的那个（--no-venv，或用 venv python 直接跑本脚本）
        fetch_model()
        print("[4/4] 校验")
        return full_report(py, show_next_step=True)

    # 换 venv 解释器重新执行自己，在装好 huggingface_hub 的那一侧下载并出报告
    child = [str(py), str(Path(__file__).resolve()), "--_in-venv", "--device", backend]
    if args.verify_hash:
        child.append("--verify-hash")
    sys.stdout.flush()  # 否则子进程的输出会插到上面几行之前
    return subprocess.run(child).returncode


if __name__ == "__main__":
    raise SystemExit(main())
