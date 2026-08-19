"""
Collect reproducibility metadata for a single run.

Usage:
    python collect_env.py --tag local_pre_step1
    python collect_env.py --tag local_pre_step2

Outputs:
    _run_logs/run_env_<tag>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "_run_logs"


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        return (out.stdout or "") + (("\n[stderr]\n" + out.stderr) if out.stderr else "")
    except FileNotFoundError:
        return f"(not found: {cmd[0]})"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _file_info(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
    }


def collect(tag: str) -> dict:
    repo_root = HERE
    while repo_root != repo_root.parent and not (repo_root / ".git").exists():
        repo_root = repo_root.parent
    has_git = (repo_root / ".git").exists()

    info: dict = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "git": {},
        "gpu": {},
        "key_packages": {},
        "raw_inputs": {},
        "notebooks": {},
    }

    if has_git:
        info["git"] = {
            "repo_root": str(repo_root),
            "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo_root).strip(),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).strip(),
            "status_short": _run(["git", "status", "--short"], cwd=repo_root),
        }
    else:
        info["git"] = {"available": False}

    info["gpu"] = {
        "nvidia_smi": _run(["nvidia-smi"]) if shutil.which("nvidia-smi") else "(no nvidia-smi)",
    }

    pkgs = [
        "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
        "tensorflow", "tensorflow-cpu", "xgboost", "lightgbm",
        "matplotlib", "seaborn", "joblib", "pyarrow", "openpyxl",
    ]
    pip_freeze_full = _run([sys.executable, "-m", "pip", "freeze"])
    info["pip_freeze"] = pip_freeze_full
    by_name: dict[str, str] = {}
    for line in pip_freeze_full.splitlines():
        if "==" in line:
            name, _, ver = line.partition("==")
            by_name[name.strip().lower()] = ver.strip()
    for p in pkgs:
        info["key_packages"][p] = by_name.get(p.lower(), "(not installed)")

    raw_dir = HERE.parent / "3. 외부 및 내부 데이터 관련 데이터와 수집 및 가공 코드"
    for fname in [
        "df_통합_LIMS_기상정보_일조포함_결합_60일_gzip_X.pkl",
        "df_통합_LIMS_기상정보_일조포함_결합_60일_gzip_y.pkl",
    ]:
        info["raw_inputs"][fname] = _file_info(raw_dir / fname)

    for nb in [
        "1.데이터 전처리+2.EDA.executed.ipynb",
        "2. 특성공학.executed.ipynb",
        "3.모델링.executed.ipynb",
    ]:
        info["notebooks"][nb] = _file_info(HERE / nb)

    for out_name in ["df_fixed.pkl.gz", "df_enhanced.pkl.gz", "df_enhanced.parquet"]:
        info.setdefault("intermediate_outputs", {})[out_name] = _file_info(HERE / out_name)

    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="e.g. local_pre_step1, local_pre_step2, local_post_step2")
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    info = collect(args.tag)
    out_path = LOG_DIR / f"run_env_{args.tag}.json"
    out_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
