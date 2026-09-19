#!/usr/bin/env python
"""anima-tagger 一键部署：建 venv → 装依赖 → 下载模型与许可证 → 校验。

用法:
    python tools/setup.py              全自动（推荐）
    python tools/setup.py --check      只体检，不装不下载；一切就绪返回 0
    python tools/setup.py --no-venv    不建 venv，装进当前解释器
    python tools/setup.py --skip-deps  跳过装依赖（依赖已装好时重复跑更快）

国内网络先设镜像再跑:
    PowerShell :  $env:HF_ENDPOINT = "https://hf-mirror.com"
    bash/zsh   :  export HF_ENDPOINT=https://hf-mirror.com

为什么模型不入库：单个 model.onnx 约 1.22 GiB，超过 GitHub 单文件上限，
必须走 LFS 或 Release。本脚本按**钉死的 revision** 拉取，保证
selected_tags.csv 与 model.onnx 是同一版本——两者错配时 wd_tagger.py
只会报"输出维度 != 标签数"，很难自己 debug 出来。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 常量

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
MODEL_DIR = ROOT / "models" / "wd-eva02-tagger-2026-canary-onnx-v2"
TOKENIZER = ROOT / "models" / "t5_tokenizer" / "tokenizer.json"
REQUIREMENTS = ROOT / "tools" / "requirements.txt"

MODEL_REPO = "Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2"
MODEL_REVISION = "0a86acfa093b33b8818667820e52fc5eccf27ff8"
MODEL_FILES = ("model.onnx", "selected_tags.csv")

# 模型本体（作者）那条线：授权与 LICENSE 文本的来源。ONNX 仓库只做格式转换、未附 LICENSE，
# 所以许可按这一份走。
LICENSE_REPO = "ashen-sensored/wd-eva02-tagger-2026-canary"
LICENSE_REVISION = "c45a59a3f17c0ca6066072b1c213e0c12a90e242"

EXPECTED_TAGS = 16473           # 词表行数；必须等于 model.onnx 的输出维度
MODEL_SHA256 = "fd78fbdf9390cbd163e4dd28f754a5bbf83bc7a111c4d20270f22415a0f66c95"
TAGS_SHA256 = "3f78c28ee0d50779edb320733f76aeaf4184694cbd09c631deef6889865f9178"
MIN_ONNX_BYTES = 1_000_000_000  # 约 1.22 GiB，只用来发现"下了半个文件"
MIN_TOKENIZER_BYTES = 1_000_000
INSTALL_EXTRA = "huggingface_hub>=0.23"   # 只有本脚本用，运行时不需要
REQUIRED_MODULES = ("onnxruntime", "PIL", "numpy", "tokenizers")  # 打标/校验真正要 import 的

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


def count_tag_rows(csv_path: Path) -> int:
    """数 selected_tags.csv 的数据行（不含表头）。"""
    with open(csv_path, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_hashes() -> list[str]:
    """逐字节核对 revision。比"大小 + 行数"更强，但要读完 1.3 GiB，故默认不做。"""
    problems: list[str] = []
    pairs = (
        (MODEL_DIR / "model.onnx", MODEL_SHA256),
        (MODEL_DIR / "selected_tags.csv", TAGS_SHA256),
    )
    for path, expected in pairs:
        if not path.exists():
            continue  # 缺失已由 collect_problems 报过，别重复
        print(f"      算 {path.name} 的 SHA256 …")
        got = sha256_file(path)
        if got.lower() != expected.lower():
            problems.append(
                f"{path.name} SHA256 不符：期望 {expected[:12]}…，实际 {got[:12]}…"
                f"——文件不是钉死的那个 revision"
            )
    return problems


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

    onnx = MODEL_DIR / "model.onnx"
    if not onnx.exists():
        problems.append(f"缺模型权重：{onnx.relative_to(ROOT)}")
    elif onnx.stat().st_size < MIN_ONNX_BYTES:
        problems.append(
            f"model.onnx 只有 {human(onnx.stat().st_size)}，疑似没下完"
            f"（应为约 1.22 GiB）——删掉整个模型目录重跑"
        )

    tags = MODEL_DIR / "selected_tags.csv"
    if not tags.exists():
        problems.append(f"缺词表：{tags.relative_to(ROOT)}")
    else:
        rows = count_tag_rows(tags)
        if rows != EXPECTED_TAGS:
            problems.append(
                f"selected_tags.csv 有 {rows} 行，应为 {EXPECTED_TAGS} 行——"
                f"模型与词表版本错配，必须删掉重下（不要改这个期望值去迁就）"
            )

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
        print(f'  & "{py}" "{ROOT / "tools" / "wd_tagger.py"}" <图片> `')
        print("      --general 0.35 --character 0.85 --json > run.json")
        print(f'  & "{py}" "{ROOT / "tools" / "anima_validate.py"}" --tagger-json run.json')
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
    subprocess.run(
        [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS), INSTALL_EXTRA],
        check=True,
    )


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


def fetch_model() -> None:
    """必须在装好 huggingface_hub 的解释器里执行。"""
    ensure_download_env()
    onnx = MODEL_DIR / "model.onnx"
    if onnx.exists() and onnx.stat().st_size >= MIN_ONNX_BYTES:
        print(f"[3/4] 权重已在位（{human(onnx.stat().st_size)}），跳过下载")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "缺 huggingface_hub。请按正常流程跑 setup.py（会自动装），"
            f"或手动：pip install \"{INSTALL_EXTRA}\""
        )

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] 下载 {MODEL_REPO}")
    print(f"      revision {MODEL_REVISION[:12]}   源 {endpoint}")
    print("      约 1.22 GiB，支持断点续传；慢是正常的，别中断它")

    try:
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(MODEL_DIR),
            allow_patterns=list(MODEL_FILES),
        )
    except Exception as exc:  # 网络 / 镜像 / 磁盘都走这里
        print(f"\n!! 下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\n请依次尝试：\n"
            '  1) 设镜像后重跑：$env:HF_ENDPOINT = "https://hf-mirror.com"; python tools/setup.py\n'
            "  2) 检查磁盘剩余空间（需要约 1.3 GiB）\n"
            f"  3) 手动下载 https://huggingface.co/{MODEL_REPO}/tree/{MODEL_REVISION}\n"
            f"     把 model.onnx 与 selected_tags.csv 放进：{MODEL_DIR}\n"
            "  4) 不要改用别的 tagger 模型——词表和 0.35/0.85 阈值都是照这份标定的",
            file=sys.stderr,
        )
        raise SystemExit(3)


def fetch_license() -> None:
    """取回模型作者随附的 Apache-2.0 文本。

    许可的来源是模型本体那条线（ashen-sensored），不是 ONNX 转换仓库——后者没放 LICENSE。
    这一步只影响授权文件的完整性，取不到也能正常打标，所以失败只警告、不中断。
    """
    dest = MODEL_DIR / "LICENSE"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[3/4] LICENSE 已在位（{dest.name}），跳过")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return  # fetch_model 已经报过这个错，别重复喊

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[3/4] 取回 {LICENSE_REPO} 的 LICENSE　源 {endpoint}")
    try:
        snapshot_download(
            repo_id=LICENSE_REPO,
            revision=LICENSE_REVISION,
            local_dir=str(MODEL_DIR),
            allow_patterns=["LICENSE"],
        )
    except Exception as exc:  # 网络 / 镜像问题都不该挡住打标
        print(
            f"[3/4] LICENSE 取回失败（不影响使用）：{type(exc).__name__}: {exc}\n"
            f"      可稍后重跑本脚本，或手动从 https://huggingface.co/{LICENSE_REPO} 取。",
            file=sys.stderr,
        )


# ---------------------------------------------------------------- 主流程

def main() -> int:
    _fix_console()
    ap = argparse.ArgumentParser(
        description="anima-tagger 一键部署",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--check", action="store_true", help="只体检，不装不下载")
    ap.add_argument("--no-venv", action="store_true", help="不建 venv，装进当前解释器")
    ap.add_argument("--skip-deps", action="store_true", help="跳过装依赖")
    ap.add_argument("--verify-hash", action="store_true",
                    help="额外逐字节核对 SHA256（要读 1.3 GiB，慢）")
    ap.add_argument("--_in-venv", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    require_venv = not args.no_venv

    def full_report(py: Path, *, show_next_step: bool) -> int:
        problems = collect_problems(py, require_venv)
        if args.verify_hash and not problems:
            problems += verify_hashes()
        return print_report(py, problems, show_next_step=show_next_step)

    if args._in_venv:
        # 子进程：此时解释器就是 venv，huggingface_hub 已可用
        fetch_model()
        fetch_license()
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
        fetch_license()
        print("[4/4] 校验")
        return full_report(py, show_next_step=True)

    # 换 venv 解释器重新执行自己，在装好 huggingface_hub 的那一侧下载并出报告
    child = [str(py), str(Path(__file__).resolve()), "--_in-venv"]
    if args.verify_hash:
        child.append("--verify-hash")
    sys.stdout.flush()  # 否则子进程的输出会插到上面几行之前
    return subprocess.run(child).returncode


if __name__ == "__main__":
    raise SystemExit(main())
