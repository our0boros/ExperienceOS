"""环境信息收集器。

收集当前运行环境的详细信息（OS、内核、Python、包版本、硬件、模型等），
保存为结构化 JSON，为未来开发提供接口。

收集的信息包括：
  - OS 与内核：发行版、内核版本、架构
  - Python 运行时：版本、路径、虚拟环境
  - 关键包版本：openai、litellm、sentence-transformers 等
  - 硬件：CPU 核数、内存、GPU（如可用）
  - LLM 后端：ollama 模型列表、DeepInfra 连通性
  - 本地模型：models/ 目录下的模型
  - τ-bench：安装状态、可用域

使用方式::

    from experience_os.env_info import collect_env_info, save_env_info
    info = collect_env_info()
    save_env_info(info, path=".experience_os_data/env_info.json")

    # 或通过 CLI
    # experience-os env-info
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _run_cmd(cmd: list[str], timeout: int = 10) -> str:
    """安全运行命令，返回 stdout 去尾空白。"""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return r.stdout.strip()
    except Exception as exc:
        return f"error: {exc}"


def _get_package_versions() -> dict[str, str]:
    """获取关键 Python 包的版本。"""
    pkgs = [
        "openai", "litellm", "sentence_transformers", "torch",
        "transformers", "numpy", "pydantic", "click",
        "tau2", "experience_os",
    ]
    versions: dict[str, str] = {}
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _get_gpu_info() -> list[dict[str, str]]:
    """获取 GPU 信息（如可用）。"""
    gpus: list[dict[str, str]] = []
    # 尝试 nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        out = _run_cmd([
            nvidia_smi,
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "memory": parts[2],
                    "driver": parts[3],
                })
    # 尝试 torch CUDA
    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "index": str(i),
                        "name": torch.cuda.get_device_name(i),
                        "memory": f"{torch.cuda.get_device_properties(i).total_mem / 1e9:.1f}GB",
                        "driver": "CUDA",
                    })
        except Exception:
            pass
    return gpus


def _get_local_models(models_dir: Path) -> list[dict[str, str]]:
    """扫描 models/ 目录下的本地模型。"""
    models: list[dict[str, str]] = []
    if not models_dir.exists():
        return models
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        info: dict[str, str] = {"name": d.name, "path": str(d)}
        # 检查是否有 config.json 来判断模型类型
        config = d / "config.json"
        if config.exists():
            try:
                cfg = json.loads(config.read_text())
                info["type"] = cfg.get("model_type", cfg.get("architectures", ["unknown"])[0])
                info["params"] = cfg.get("params", {}).get("_name_or_path", "")
            except Exception:
                info["type"] = "unknown"
        # 检查是否有 safetensors
        safetensors = list(d.glob("*.safetensors"))
        if safetensors:
            info["format"] = "safetensors"
            info["files"] = len(safetensors)
        gguf = list(d.glob("*.gguf"))
        if gguf:
            info["format"] = "gguf"
            info["files"] = len(gguf)
        models.append(info)
    return models


def _get_ollama_models() -> list[str]:
    """获取 ollama 中已安装的模型列表。"""
    ollama = shutil.which("ollama")
    if not ollama:
        return []
    out = _run_cmd([ollama, "list"])
    lines = out.splitlines()[1:]  # 跳过表头
    return [l.split()[0] for l in lines if l.strip()]


def _get_tau2_info() -> dict[str, Any]:
    """获取 τ-bench 安装和域信息。"""
    info: dict[str, Any] = {"installed": False}
    try:
        import tau2
        info["installed"] = True
        info["version"] = getattr(tau2, "__version__", "unknown")
        from tau2.registry import list_domains
        info["domains"] = list_domains()
    except ImportError:
        pass
    except Exception as exc:
        info["error"] = str(exc)
    return info


def collect_env_info(project_dir: Path | None = None) -> dict[str, Any]:
    """收集完整的环境信息。

    Args:
        project_dir: 项目根目录，默认为当前目录

    Returns:
        结构化环境信息字典
    """
    if project_dir is None:
        project_dir = Path.cwd()

    models_dir = project_dir / "models"

    import multiprocessing
    mem_bytes = 0
    try:
        import psutil
        mem = psutil.virtual_memory()
        mem_bytes = mem.total
    except ImportError:
        # 回退到 /proc/meminfo
        try:
            meminfo = Path("/proc/meminfo").read_text()
            for line in meminfo.splitlines():
                if line.startswith("MemTotal:"):
                    mem_bytes = int(line.split()[1]) * 1024
                    break
        except Exception:
            pass

    info: dict[str, Any] = {
        "collected_at": _run_cmd(["date", "-Iseconds"]),
        "collector_version": "1.0",

        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "distro": _get_distro(),
        },

        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "in_venv": hasattr(sys, "real_prefix") or (
                sys.prefix != sys.base_prefix if hasattr(sys, "base_prefix") else False
            ),
            "path": sys.path[:5],
        },

        "packages": _get_package_versions(),

        "hardware": {
            "cpu_count": multiprocessing.cpu_count(),
            "memory_bytes": mem_bytes,
            "memory_gb": round(mem_bytes / 1e9, 1) if mem_bytes else 0,
            "gpus": _get_gpu_info(),
        },

        "llm_backends": {
            "ollama": {
                "available": shutil.which("ollama") is not None,
                "models": _get_ollama_models(),
                "base_url": os.environ.get("EOS_OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            },
            "deepinfra": {
                "token_set": bool(os.environ.get("DEEPINFRA_TOKEN") or os.environ.get("DEEPINFRA_API_KEY")),
                "base_url": os.environ.get("EOS_DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai"),
            },
        },

        "local_models": _get_local_models(models_dir),

        "tau2": _get_tau2_info(),

        "environment_variables": {
            k: os.environ.get(k, "")
            for k in [
                "EOS_LLM_BACKEND",
                "EOS_OLLAMA_MODEL",
                "EOS_DEEPINFRA_MODEL",
                "EOS_EMBEDDING_MODEL",
                "EOS_EMBEDDING_DIM",
                "EOS_MIN_SUPPORT",
                "EOS_DATA_DIR",
            ]
        },
    }

    return info


def _get_distro() -> str:
    """获取 Linux 发行版名称。"""
    try:
        r = _run_cmd(["cat", "/etc/os-release"])
        for line in r.splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    return platform.platform()


def save_env_info(
    info: dict[str, Any] | None = None,
    path: str | Path = ".experience_os_data/env_info.json",
) -> Path:
    """保存环境信息到 JSON 文件。

    Args:
        info: 环境信息字典，None 则自动收集
        path: 输出文件路径

    Returns:
        保存的文件路径
    """
    if info is None:
        info = collect_env_info()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    log.info("Environment info saved to %s", p)
    return p


def print_env_info(info: dict[str, Any] | None = None) -> None:
    """格式化打印环境信息摘要。"""
    if info is None:
        info = collect_env_info()

    print("=" * 60)
    print("  ExperienceOS 环境信息")
    print("=" * 60)

    print(f"\n[OS]")
    print(f"  {info['os']['distro']}")
    print(f"  Kernel: {info['os']['release']} ({info['os']['machine']})")

    print(f"\n[Python]")
    print(f"  {info['python']['version'].split()[0]}")
    print(f"  venv: {info['python']['in_venv']}")

    print(f"\n[Hardware]")
    print(f"  CPU: {info['hardware']['cpu_count']} cores")
    print(f"  RAM: {info['hardware']['memory_gb']} GB")
    for gpu in info["hardware"]["gpus"]:
        print(f"  GPU: {gpu['name']} ({gpu['memory']})")

    print(f"\n[Ollama Models]")
    for m in info["llm_backends"]["ollama"]["models"]:
        print(f"  {m}")

    print(f"\n[Local Models]")
    for m in info["local_models"]:
        t = m.get("type", "?")
        f = m.get("format", "?")
        print(f"  {m['name']} ({t}, {f})")

    print(f"\n[τ-bench]")
    tau2 = info["tau2"]
    print(f"  installed: {tau2.get('installed', False)}")
    if tau2.get("domains"):
        print(f"  domains: {', '.join(tau2['domains'])}")

    print(f"\n[Key Packages]")
    for pkg, ver in sorted(info["packages"].items()):
        if ver != "not installed":
            print(f"  {pkg}: {ver}")


if __name__ == "__main__":
    print_env_info()
    save_env_info()
