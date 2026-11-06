import sys
import json
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. CẤU HÌNH CONSOLE UTF-8 & SYS.PATH CHO MỌI MODULE
# -----------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# -----------------------------------------------------------------------------
# 2. CẤU HÌNH MATPLOTLIB KHÔNG CẦN GUI (AGG) & PHONG CÁCH NOTEBOOK
# -----------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.autolayout"] = True
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------------------------------------------------------
# 3. HẰNG SỐ ĐƯỜNG DẪN & SEED DỰ ÁN
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

RAW_CSV = ROOT_DIR / "data" / "raw" / "CEAS_08.csv"
CLEAN_CSV = ROOT_DIR / "data" / "processed" / "clean_emails.csv"
TRAIN_TEST_JOBLIB = ROOT_DIR / "data" / "processed" / "train_test_data.joblib"
PREPROCESSOR_JOBLIB = ROOT_DIR / "models" / "saved_models" / "preprocessor.joblib"
SAVED_MODELS_DIR = ROOT_DIR / "models" / "saved_models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
IMAGES_DIR = ROOT_DIR / "docs" / "images"
REPORT_DIR = ROOT_DIR / "docs" / "report"
LOGS_DIR = ROOT_DIR / "logs"

# Đảm bảo các thư mục đích luôn tồn tại
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 4. HÀM CHUYỂN ĐỔI KIỂU DỮ LIỆU SANG JSON THUẦN PYTHON
# -----------------------------------------------------------------------------
def _json_serializable(obj):
    """Chuyển đổi các kiểu dữ liệu NumPy/Pandas/Path sang kiểu tương thích JSON."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {str(k): _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)


# -----------------------------------------------------------------------------
# 5. QUẢN LÝ PHONG BÌ KẾT QUẢ TỪNG CHẶNG (STAGE ENVELOPE)
# -----------------------------------------------------------------------------
def save_stage_result(stage_id: str, data: dict) -> Path:
    """
    Lưu kết quả chặng vào outputs/stage_{stage_id}.json theo phong bì chuẩn:
    {
      stage, status, started_at, finished_at, duration_sec, quick_mode,
      log_file, config, metrics, tables, figures, timings
    }
    """
    envelope = {
        "stage": stage_id,
        "status": data.get("status", "success"),
        "started_at": data.get("started_at", datetime.now().isoformat()),
        "finished_at": data.get("finished_at", datetime.now().isoformat()),
        "duration_sec": round(float(data.get("duration_sec", 0.0)), 2),
        "quick_mode": bool(data.get("quick_mode", False)),
        "log_file": str(data.get("log_file", "")),
        "config": data.get("config", {}),
        "metrics": data.get("metrics", {}),
        "tables": data.get("tables", {}),
        "figures": data.get("figures", []),
        "timings": data.get("timings", {})
    }

    target_file = OUTPUTS_DIR / f"stage_{stage_id}.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2, default=_json_serializable)

    return target_file


def load_stage_result(stage_id: str) -> dict:
    """Nạp dữ liệu phong bì kết quả chặng từ outputs/stage_{stage_id}.json."""
    target_file = OUTPUTS_DIR / f"stage_{stage_id}.json"
    if not target_file.exists():
        return None
    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# 6. KIỂM THỬ ĐIỀU KIỆN TIÊN QUYẾT & LƯU HÌNH ẢNH
# -----------------------------------------------------------------------------
def require_files(logger, paths: list, hint: str = "") -> None:
    """
    Kiểm tra danh sách các file bắt buộc phải tồn tại.
    Nếu thiếu, ghi log lỗi cảnh báo dễ hiểu và dừng chương trình với mã thoát khác 0.
    """
    missing = [Path(p) for p in paths if not Path(p).exists()]
    if missing:
        msg = f"Chưa có tệp: {', '.join(p.name for p in missing)}."
        if hint:
            msg += f" Gợi ý: {hint}"
        logger.error(f"✖ ĐIỀU KIỆN TIÊN QUYẾT THẤT BẠI: {msg}")
        sys.exit(1)


def save_figure(fig, filename: str, dpi: int = 150) -> Path:
    """
    Lưu biểu đồ Matplotlib vào docs/images/{filename} với DPI chuẩn,
    tự động giải phóng bộ nhớ RAM.
    """
    dest = IMAGES_DIR / filename
    fig.savefig(str(dest), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return dest
