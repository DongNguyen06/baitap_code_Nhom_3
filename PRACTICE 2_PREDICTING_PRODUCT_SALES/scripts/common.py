import json
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN TẬP TRUNG (CENTRALIZED PATHS)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = BASE_DIR / "docs"
REPORT_DIR = DOCS_DIR / "report"
IMAGES_DIR = DOCS_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Tự động khởi tạo các thư mục đầu ra nếu chưa tồn tại
for directory in [PROCESSED_DIR, REPORT_DIR, IMAGES_DIR, SAVED_MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. CÁC TỆP DỮ LIỆU & HIỆN VẬT CỐT LÕI
# -----------------------------------------------------------------------------
RAW_CSV = RAW_DIR / "Amazon Sale Report.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_amazon_sales.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
TRAIN_TEST_JOBLIB = PROCESSED_DIR / "train_test_data.joblib"
PREPROCESSOR_JOBLIB = SAVED_MODELS_DIR / "preprocessor.joblib"

RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# 3. QUẢN LÝ DỮ LIỆU PHONG BÌ (ENVELOPE PATTERN)
# -----------------------------------------------------------------------------
def _json_serializable(obj):
    """Bộ chuyển đổi an toàn cho các kiểu dữ liệu NumPy / Pandas sang JSON."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return str(obj)


def save_stage_result(stage_id: str, data: dict) -> Path:
    """
    Lưu phong bì dữ liệu chặng vào outputs/stage_{stage_id}.json
    Định dạng chuẩn phong bì:
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
# 4. KIỂM THỬ ĐIỀU KIỆN TIÊN QUYẾT & LƯU HÌNH ẢNH
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
