"""
Modal app: GPU L4에서 `notebooks/3.모델링.executed.ipynb` 를 실행하고
결과를 experiments/<EXP_ID>/ 폴더에 자동 저장한다.

준비:
1) 로컬에서 1·2번 노트북을 실행해 data/processed/df_enhanced.parquet 를 만든다.
2) `python -m modal token new` 로 인증.

실행 예 (PowerShell):
    $env:EXP_ID="EXP-002_focal_loss"
    $env:CHANGED="loss: BCE -> Focal(gamma=2); epochs: 30 -> 50"
    $env:NOTES="Focal loss로 양성 가중 강화"
    $env:BASE_EXP="EXP-001_baseline"
    python -m modal run src/modal_run.py::run_notebook

실행 예 (bash):
    EXP_ID=EXP-002_focal_loss \
    CHANGED="loss: BCE -> Focal(gamma=2); epochs: 30 -> 50" \
    NOTES="Focal loss로 양성 가중 강화" \
    BASE_EXP=EXP-001_baseline \
    python -m modal run src/modal_run.py::run_notebook

환경변수:
- EXP_ID   : 실험 식별자. 없으면 EXP-YYYYMMDD-HHMMSS_adhoc 으로 자동 생성.
- CHANGED  : 이번 실험에서 baseline 대비 바꾼 점들. 세미콜론(;) 으로 구분.
- NOTES    : 자유 메모.
- BASE_EXP : 비교 기준이 된 실험 ID.

산출물 (experiments/<EXP_ID>/):
- notebook.executed.ipynb : 실행된 노트북 (셀 출력 포함)
- execute.log             : nbconvert 실행 로그
- run_metrics.json        : 노트북이 dump 한 메트릭 (있을 때만)
- manifest.json           : 본 스크립트가 자동 생성 (메타 + 메트릭 요약)
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone

import modal

APP_NAME = "aflatoxin-modeling"
VOLUME_NAME = "aflatoxin-runs"

# src/ 안에 있으므로 부모의 부모가 프로젝트 루트
SRC_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "df_enhanced.parquet"
FONT_HELPER_PATH = SRC_DIR / "font_helper.py"
REQ_PATH = SRC_DIR / "requirements.txt"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def _resolve_exp_id() -> str:
    exp_id = os.environ.get("EXP_ID", "").strip()
    if exp_id:
        return exp_id
    return f"EXP-{datetime.now():%Y%m%d-%H%M%S}_adhoc"


# Modal Image: requirements.txt 그대로 사용 + 한글 폰트 OS 패키지
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("fonts-nanum", "fontconfig")
    .pip_install_from_requirements(str(REQ_PATH))
    # 로컬 pandas 3.0/numpy 2.x 로 만든 pickle 의 pyarrow 의존 객체 unpickle 호환용
    .pip_install("pyarrow==16.1.0")
    # 한글 폰트 캐시 갱신 + matplotlib 캐시 무효화
    .run_commands(
        "fc-cache -fv",
        "rm -rf /root/.cache/matplotlib || true",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    gpu="L4",
    volumes={"/work": volume},
    timeout=60 * 60 * 3,  # 3시간 상한
)
def run_notebook_remote(
    notebook_bytes: bytes,
    data_bytes: bytes,
    font_helper_bytes: bytes,
    exp_id: str,
):
    """
    원격(GPU L4)에서 노트북 실행. 결과 bytes 를 직접 반환해
    로컬에서 `modal volume get` 없이 저장 가능.
    """
    import os
    import subprocess
    import time
    from pathlib import Path

    work = Path("/work")
    run_dir = work / "run"
    out_dir = work / "outputs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb_in = run_dir / "notebook.ipynb"
    nb_out = out_dir / "notebook.executed.ipynb"
    log_out = out_dir / "execute.log"

    nb_in.write_bytes(notebook_bytes)
    parquet_path = run_dir / "df_enhanced.parquet"
    parquet_path.write_bytes(data_bytes)
    (run_dir / "font_helper.py").write_bytes(font_helper_bytes)

    # 컨테이너의 pandas/numpy ABI 로 pkl.gz 재생성 (노트북은 read_pickle 사용)
    import pandas as pd
    df_compat = pd.read_parquet(parquet_path, engine="pyarrow")
    pkl_path = run_dir / "df_enhanced.pkl.gz"
    df_compat.to_pickle(pkl_path, compression="gzip")
    print(
        f"[compat] parquet -> pkl.gz: shape={df_compat.shape}, "
        f"pkl_size={pkl_path.stat().st_size:,}B"
    )

    try:
        nvidia = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        nvidia = "(nvidia-smi unavailable)"

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb_in),
        "--output",
        str(nb_out),
        "--ExecutePreprocessor.timeout=3600",
        "--ExecutePreprocessor.kernel_name=python3",
    ]
    # [DEFECT-015 fix] TF 결정성 환경변수를 subprocess 환경에 주입
    sub_env = os.environ.copy()
    sub_env['PYTHONHASHSEED'] = '42'
    sub_env['TF_DETERMINISTIC_OPS'] = '1'
    sub_env['TF_CUDNN_DETERMINISTIC'] = '1'
    # 노트북의 metric dump 셀이 결과를 outputs/<EXP_ID>/ 로 직접 떨어뜨리도록.
    sub_env['RUN_LOG_DIR'] = str(out_dir)
    sub_env['EXP_ID'] = exp_id

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        env=sub_env,
    )
    elapsed = time.time() - t0

    log = (
        f"=== exp_id={exp_id} ===\n"
        f"=== nvidia-smi ===\n{nvidia}\n"
        f"=== cmd ===\n{' '.join(cmd)}\n"
        f"=== returncode={proc.returncode} elapsed={elapsed:.1f}s ===\n"
        f"=== stdout ===\n{proc.stdout}\n"
        f"=== stderr ===\n{proc.stderr}\n"
    )
    log_out.write_text(log, encoding="utf-8")
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"nbconvert failed (rc={proc.returncode}). See {log_out}"
        )

    metrics_path = out_dir / "run_metrics.json"
    metrics_bytes = metrics_path.read_bytes() if metrics_path.exists() else None

    return {
        "exp_id": exp_id,
        "elapsed_sec": elapsed,
        "notebook_bytes": nb_out.read_bytes(),
        "log_bytes": log.encode("utf-8"),
        "metrics_bytes": metrics_bytes,
        "remote_out_dir": str(out_dir),
    }


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        return None


def _extract_summary(metrics_path: pathlib.Path) -> dict:
    """run_metrics.json 에서 요약 메트릭만 뽑아낸다."""
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"failed to parse run_metrics.json: {e}"}

    test = data.get("metrics", {}).get("test", {})
    cm = test.get("confusion_matrix") or [[None, None], [None, None]]
    return {
        "best_threshold": data.get("metrics", {}).get("best_th_f2_from_val"),
        "roc_auc": test.get("roc_auc"),
        "pr_auc": test.get("pr_auc"),
        "f2": test.get("f2_at_best_th"),
        "f1": test.get("f1_at_best_th"),
        "precision": test.get("precision_at_best_th"),
        "recall": test.get("recall_at_best_th"),
        "accuracy": test.get("accuracy_at_best_th"),
        "confusion_matrix": {
            "tn": cm[0][0], "fp": cm[0][1],
            "fn": cm[1][0], "tp": cm[1][1],
        },
    }


def _write_manifest(local_dir: pathlib.Path, exp_id: str, elapsed: float) -> dict:
    summary = _extract_summary(local_dir / "run_metrics.json")
    changed = [
        s.strip() for s in os.environ.get("CHANGED", "").split(";") if s.strip()
    ]
    manifest = {
        "exp_id": exp_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": elapsed,
        "git_commit": _git_head(),
        "base_exp": os.environ.get("BASE_EXP") or None,
        "changed": changed,
        "runtime": {"platform": "Modal GPU L4", "python": "3.11"},
        "summary_metrics": summary,
        "notes": os.environ.get("NOTES", ""),
    }
    (local_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


@app.local_entrypoint()
def run_notebook():
    """
    로컬 entrypoint.
    experiments/<EXP_ID>/notebook.ipynb 를 업로드해 GPU에서 실행한다.
    결과는 같은 폴더에 notebook.executed.ipynb / execute.log / run_metrics.json /
    manifest.json 으로 저장된다.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"입력 데이터가 없습니다: {DATA_PATH}\n"
            "1·2번 노트북을 먼저 실행해 data/processed/df_enhanced.parquet 를 생성해 주세요."
        )

    exp_id = _resolve_exp_id()
    local_out = EXPERIMENTS_DIR / exp_id
    notebook_path = local_out / "notebook.ipynb"

    if not notebook_path.exists():
        raise FileNotFoundError(
            f"실험 노트북이 없습니다: {notebook_path}\n"
            f"먼저 다음을 실행해 실험 폴더를 생성하세요:\n"
            f"  python src/new_experiment.py {exp_id}"
        )

    executed_out = local_out / "notebook.executed.ipynb"
    if executed_out.exists():
        raise FileExistsError(
            f"이미 실행된 결과가 있습니다: {executed_out}\n"
            f"다른 EXP_ID 로 실행하거나, 이전 결과를 백업 후 삭제하세요."
        )

    print(f"[exp_id] {exp_id}")
    print(f"[upload] notebook: {notebook_path} ({notebook_path.stat().st_size:,} B)")
    print(f"[upload] data    : {DATA_PATH} ({DATA_PATH.stat().st_size:,} B)")
    print(f"[upload] helper  : {FONT_HELPER_PATH}")

    result = run_notebook_remote.remote(
        notebook_path.read_bytes(),
        DATA_PATH.read_bytes(),
        FONT_HELPER_PATH.read_bytes(),
        exp_id,
    )

    (local_out / "notebook.executed.ipynb").write_bytes(result["notebook_bytes"])
    (local_out / "execute.log").write_bytes(result["log_bytes"])
    if result.get("metrics_bytes"):
        (local_out / "run_metrics.json").write_bytes(result["metrics_bytes"])

    manifest = _write_manifest(local_out, exp_id, result["elapsed_sec"])
    print(f"[done] elapsed={result['elapsed_sec']:.1f}s")
    print(f"[saved] {local_out}")
    sm = manifest.get("summary_metrics") or {}
    if sm and "error" not in sm:
        print(
            f"[metrics] roc_auc={sm.get('roc_auc')} "
            f"pr_auc={sm.get('pr_auc')} "
            f"f2={sm.get('f2')} "
            f"recall={sm.get('recall')} "
            f"precision={sm.get('precision')}"
        )
    print("[next] leaderboard 갱신: python src/update_leaderboard.py")
